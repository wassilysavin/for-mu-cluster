import assert from 'node:assert/strict';
import { open, readFile, statfs } from 'node:fs/promises';
import { dirname } from 'node:path';
import { createInterface } from 'node:readline';
import { createActualHistoryHarness, hash } from './evaluation/actual-history.mjs';
import { encodeCalls } from './evaluation/functiongemma.mjs';
import { renderPrompt } from './evaluation/history-train.mjs';
import { buildModelInput, pinnedTokenizerSha256 } from './evaluation/model-input.mjs';
import { canonical } from './evaluation/schema.mjs';
import { applyModelOutput, createSession, recordIncompleteModelOutput, recordProviderFailure,
  sessionHistory } from './evaluation/session.mjs';
import { compactObservation } from './diagnose-data.mjs';
import { gradeTransfer, transferRegistry } from './transfer_contract.mjs';

const panels = ['closed_loop', 'challenge'];
const assetSha256 = '9df60f65e30c8b0d54966b2a2076af1318810ac1cb3cfecd77457dc299c0ff7f';
const api = { applyModelOutput, createSession, sessionHistory, encodeCalls, canonical, buildModelInput, pinnedTokenizerSha256 };
const harness = createActualHistoryHarness(api);

async function load(registryPath, bundlePath) {
  const source = JSON.parse(await readFile(registryPath, 'utf8'));
  const registry = await transferRegistry(source);
  const raw = await readFile(bundlePath, 'utf8');
  assert.equal(hash(raw), assetSha256, 'Fresh DEV asset differs from the frozen version');
  const data = JSON.parse(raw);
  assert.equal(data.schema, 'semantic-transfer-dev-v1');
  assert.equal(data.source_registry_sha256, source.fingerprint);
  const registryHash = await registry.fingerprint();
  const bundleHash = hash(canonical({ asset_sha256: assetSha256, registry_sha256: registryHash, version: data.schema }));
  const ids = new Set(), turns = new Map();
  for (const panel of panels) {
    for (const row of data.panels[panel]) {
      assert.ok(!ids.has(row.case_id) && row.panel === panel && row.split === 'dev');
      assert.ok(row.family_id && row.bootstrap_family_id && row.question?.trim());
      assert.ok(['supported', 'outside_scope'].includes(row.cohort));
      assert.equal(typeof row.reference_eligible, 'boolean');
      ids.add(row.case_id);
      if (row.dialogue_id) {
        assert.ok(!row.initial_prefix && panel === 'closed_loop');
        assert.equal(row.turn_index, (turns.get(row.dialogue_id) ?? 0) + 1);
        turns.set(row.dialogue_id, row.turn_index);
      } else {
        assert.equal(row.turn_index, 1);
        assert.equal(Boolean(row.initial_prefix?.length), panel === 'challenge');
      }
    }
  }
  assert.equal(turns.size, 24);
  assert.ok([...turns.values()].every(n => n === 4));
  assert.equal(ids.size, 233);
  const plan = panels.flatMap(panel => data.panels[panel]);
  return { registry, data, plan, bundleHash, registryHash };
}

async function score(registry, row, before, after, resolved) {
  const frozen = await harness.grade(registry, row, before, after, resolved);
  const result = await gradeTransfer(frozen, resolved, before, after);
  return row.initial_prefix ? { ...result, standalone: false } : result;
}

async function materialize(registry, before, question, generation) {
  if (generation.error) {
    assert.ok(['context_overflow', 'inference_failed'].includes(generation.error));
    return recordProviderFailure(registry, before, question, generation.error);
  }
  const ids = generation.raw_ids;
  assert.ok(Array.isArray(ids) && ids.length > 0 && ids.length <= 512);
  assert.ok(ids.every(id => Number.isInteger(id) && id >= 0 && id < 262144));
  assert.equal(typeof generation.raw_text, 'string');
  if (ids.at(-1) === 50) {
    assert.ok(!ids.slice(0, -1).includes(50));
    return applyModelOutput(registry, before, question, generation.raw_text);
  }
  assert.equal(ids.length, 512);
  assert.ok(!ids.includes(50));
  return recordIncompleteModelOutput(registry, before, question, generation.raw_text, ids, 'token_limit');
}

async function selftest({ registry, plan }) {
  const sessions = new Map(), counters = { alternatives: 0, unavailable_answers: 0, retry_after_error: 0,
    retry_after_success: 0, wrong_scope: 0, clarified: 0, failed_live_predecessors: 0, retry_quoted_instead: 0, partial_compound_answers: 0,
    necessary_clarifications: 0, unnecessary_clarifications: 0, ambiguous_guesses_retained: 0 };
  for (const row of plan) {
    const before = sessions.get(row.dialogue_id) ?? await harness.initial(registry, row);
    const resolved = harness.resolve(row, before);
    let continuation;
    for (const batch of resolved.allowed_batches) {
      const after = await applyModelOutput(registry, before, resolved.question, encodeCalls(batch));
      const verdict = await score(registry, row, before, after, resolved);
      assert.equal(verdict.success, true, row.case_id);
      continuation ??= after;
      counters.alternatives++;
    }
    if (row.dialogue_id) sessions.set(row.dialogue_id, continuation);
    if (row.policy.kind === 'retry') {
      counters[resolved.branch === 'failed' ? 'retry_after_error' : 'retry_after_success']++;
    }
    if (row.policy.kind === 'ambiguous') counters.clarified++;
    if (row.panel === 'challenge' && row.policy.kind === 'ambiguous' && row.policy.object_kind === 'answer') {
      const once = before.semantic.turns[0].answers[0].id;
      const guess = encodeCalls([{ name: 'repeat_answer', arguments: { target: once } }]);
      const after = await applyModelOutput(registry, before, resolved.question, guess);
      const verdict = await score(registry, row, before, after, resolved);
      assert.equal(verdict.success, false);
      assert.equal(verdict.failure_category, 'semantic_unresolved');
      assert.equal(verdict.critical_status, 'review_required');
      counters.ambiguous_guesses_retained++;
    }
    if (row.panel === 'challenge' && row.policy.kind === 'repeat_answer') {
      const clarify = encodeCalls([{ name: 'clarify_reference', arguments: { kind: 'answer' } }]);
      const next = await applyModelOutput(registry, before, resolved.question, clarify);
      const clarification = await score(registry, row, before, next, resolved);
      assert.equal(clarification.critical_status, 'none_observed');
      if (resolved.branch === 'absent') {
        assert.equal(clarification.success, true);
        assert.equal(clarification.failure_category, null);
        counters.necessary_clarifications++;
      } else {
        assert.equal(clarification.success, false);
        assert.equal(clarification.failure_category, 'unnecessary_clarification');
        counters.unnecessary_clarifications++;
      }
      const selected = before.semantic.turns.at(-1).answers[0].id;
      const wrong = [{ name: 'repeat_answer', arguments: { target: selected } }];
      if (resolved.reference_target_alternatives.some(refs => refs.includes(selected))) continue;
      const after = await applyModelOutput(registry, before, resolved.question, encodeCalls(wrong));
      const verdict = await score(registry, row, before, after, resolved);
      assert.equal(verdict.success, false);
      assert.equal(verdict.critical_category, 'wrong_reference', row.case_id);
      counters[resolved.branch === 'absent' ? 'unavailable_answers' : 'wrong_scope']++;
    }
    if (row.panel === 'challenge' && row.policy.kind === 'retry') {
      const prior = before.exchanges[row.policy.target_exchange - 1];
      const quote = [{ name: 'repeat_exchange', arguments: { target: prior.id } }];
      const after = await applyModelOutput(registry, before, resolved.question, encodeCalls(quote));
      assert.equal((await score(registry, row, before, after, resolved)).success, false, row.case_id);
      counters.retry_quoted_instead++;
    }
    if (row.dialogue_id && row.turn_index === 1 && row.policy.calls?.length === 2) {
      const partial = encodeCalls([row.policy.calls[0]]);
      const after = await applyModelOutput(registry, before, resolved.question, partial);
      assert.equal((await score(registry, row, before, after, resolved)).success, false, row.case_id);
      counters.partial_compound_answers++;
    }
    if (row.dialogue_id && row.turn_index > 1 && ['retry','quote','repeat_answer','pending_cancel','pending_resolve','replace'].includes(row.policy.kind)) {
      let failed = await createSession(registry, { dialogueVersion: 2, sessionId: 'failed-prefix-control' });
      for (let n = 0; n < row.turn_index - 1; n++) {
        const original = before.exchanges[n];
        failed = await applyModelOutput(registry, failed, original.question, 'invalid output');
      }
      const actual = harness.resolve(row, failed);
      const after = await applyModelOutput(registry, failed, actual.question, encodeCalls(actual.allowed_batches[0]));
      assert.equal((await score(registry, row, failed, after, actual)).success, true, row.case_id);
      counters.failed_live_predecessors++;
    }
  }
  assert.ok(Object.values(counters).every(count => count > 0));
  return counters;
}

async function savedPrefix(path) {
  let bytes;
  try { bytes = await readFile(path); } catch (error) { if (error.code === 'ENOENT') return []; throw error; }
  if (bytes.length === 0) return [];
  const boundary = bytes.lastIndexOf(10) + 1;
  const lines = bytes.subarray(0, boundary).toString('utf8').split('\n').slice(0, -1).map(line => JSON.parse(line));
  if (boundary === bytes.length) return lines;
  const fragment = bytes.subarray(boundary).toString('utf8');
  let complete = false;
  try { lines.push(JSON.parse(fragment)); complete = true; } catch (error) { if (!(error instanceof SyntaxError)) throw error; }
  const handle = await open(path, 'r+');
  try {
    if (complete) await handle.write('\n', bytes.length, 'utf8');
    else await handle.truncate(boundary);
    await handle.datasync();
  } finally { await handle.close(); }
  return lines;
}

const emit = message => new Promise((resolve, reject) =>
  process.stdout.write(JSON.stringify(message) + '\n', error => error ? reject(error) : resolve()));

async function evaluate({ registry, plan, data, bundleHash }, path, reserve, binding) {
  const saved = await savedPrefix(path), sessions = new Map(), records = [];
  assert.ok(saved.length <= plan.length, 'Fresh DEV receipts extend past the frozen panel');
  const input = createInterface({ input: process.stdin, crlfDelay: Infinity });
  const replies = input[Symbol.asyncIterator](), file = await open(path, 'a');
  try {
    for (const [i, row] of plan.entries()) {
      const before = sessions.get(row.dialogue_id) ?? await harness.initial(registry, row);
      const resolved = harness.resolve(row, before);
      const prompt = await renderPrompt(registry, api, before, resolved.question);
      const promptHash = hash(prompt), previous = saved[i];
      let generation;
      if (previous !== undefined) {
        assert.equal(previous.case_id, row.case_id);
        assert.equal(previous.bundle_sha256, bundleHash);
        assert.equal(previous.prompt_sha256, promptHash);
        generation = previous.generation;
      } else {
        await emit({ kind: 'generate', case_id: row.case_id, prompt });
        const reply = await replies.next();
        assert.ok(!reply.done, 'Model provider disconnected; the saved prefix can be resumed');
        generation = JSON.parse(reply.value);
      }
      if (binding) assert.equal(generation.diagnostic_worker_sha256, binding, 'Provider or checkpoint identity changed');
      const after = await materialize(registry, before, resolved.question, generation);
      const graded = await score(registry, row, before, after, resolved);
      const receipt = compactObservation({ ...graded, bundle_sha256: bundleHash, prompt_sha256: promptHash,
        before_visible_history_sha256: resolved.before_visible_history_sha256, question: resolved.question,
        policy_branch: resolved.branch, expected_model_call_batches: resolved.allowed_batches, generation, before, after });
      if (previous !== undefined) {
        assert.equal(canonical(previous), canonical(receipt), 'Fresh DEV replay differs from saved observation');
      } else {
        const line = JSON.stringify(receipt) + '\n';
        const { bavail, bsize } = await statfs(dirname(path), { bigint: true });
        assert.ok(bavail * bsize >= reserve + BigInt(Buffer.byteLength(line)), 'Fresh DEV disk reserve exhausted');
        await file.appendFile(line); await file.datasync();
      }
      if (row.dialogue_id) sessions.set(row.dialogue_id, after);
      records.push(receipt);
    }
    const panelsScore = Object.fromEntries(panels.map(panel => {
      const scored = records.filter(r => r.panel === panel);
      const summary = harness.summarize(data.panels[panel], scored);
      const available = scored.filter(r => r.reference_eligible && r.resolved_reference_available);
      const passed = available.filter(r => r.reference_success).length;
      summary.exact_reference_when_available = { planned: available.length, passed,
        accuracy: available.length ? passed / available.length : null };
      return [panel, summary];
    }));
    panelsScore.challenge.standalone = { planned: 0, passed: 0, accuracy: null };
    const summary = panelsScore.closed_loop;
    const failed = records.filter(r => r.generation.error === 'inference_failed').map(r => r.case_id);
    await emit({ kind: 'result', score: { ...summary, summary, panels: panelsScore, observed_records: records.length,
      bundle_sha256: bundleHash, infrastructure_failures: failed.length, infrastructure_failure_case_ids: failed,
      quality_valid: !failed.length, review_status: 'automatic_conservative_unresolved_retained', release_ready: false } });
  } finally { input.close(); await file.close(); }
}

const [registryPath, dataPath, output, config = '{}', reserve = '0'] = process.argv.slice(2);
assert.ok(registryPath && dataPath && output, 'Expected registry, DEV asset and output file or --check/--selftest');
const context = await load(registryPath, dataPath);
const header = { cases: context.plan.length, bundle_sha256: context.bundleHash, registry_sha256: context.registryHash,
  panels: Object.fromEntries(panels.map(panel => [panel, context.data.panels[panel].length])) };
if (output === '--check') await emit({ kind: 'check', ...header });
else if (output === '--selftest') await emit({ kind: 'selftest', ...header, ...(await selftest(context)) });
else {
  const expected = JSON.parse(config);
  if (expected.bundle_sha256) assert.equal(expected.bundle_sha256, context.bundleHash);
  assert.ok(BigInt(reserve) >= 0n);
  const binding = expected.diagnostic_worker_sha256;
  if (binding !== undefined) assert.match(binding, /^[0-9a-f]{64}$/);
  await evaluate(context, output, BigInt(reserve), binding);
}
