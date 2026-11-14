import assert from 'node:assert/strict';
import { readFile, open, statfs } from 'node:fs/promises';
import { dirname } from 'node:path';
import { createInterface } from 'node:readline';
import { createRegistry } from './evaluation/registry.mjs';
import { canonical } from './evaluation/schema.mjs';
import { createSession, applyModelOutput, recordProviderFailure, recordIncompleteModelOutput, sessionHistory } from './evaluation/session.mjs';
import { encodeCalls } from './evaluation/functiongemma.mjs';
import { buildModelInput, pinnedTokenizerSha256 } from './evaluation/model-input.mjs';
import { createActualHistoryHarness, hash } from './evaluation/actual-history.mjs';

const LIMIT = 512, STOP = 50;
const api = { createSession, applyModelOutput, sessionHistory, encodeCalls, canonical };
const harness = createActualHistoryHarness(api);
const emit = value => new Promise((resolve, reject) => process.stdout.write(`${JSON.stringify(value)}\n`, e => e ? reject(e) : resolve()));
const [registryPath, dataPath, output, configText = '{}', reserveText = '0'] = process.argv.slice(2);
assert.ok(registryPath && dataPath && output, 'Expected registry, DEV bundle, and observations path or --check/--selftest');
const config = JSON.parse(configText);
const source = JSON.parse(await readFile(registryPath, 'utf8'));
const registry = createRegistry(source.knowledge, source.contracts);
assert.equal(await registry.fingerprint(), source.fingerprint);
const data = JSON.parse(await readFile(dataPath, 'utf8'));
assert.equal(data.schema, 'actual-history-v3-dev-1');
assert.equal(data.registry_sha256, source.fingerprint);
const bundleHash = hash(canonical({ ...data, review: null }));
if (output !== '--selftest') {
  assert.equal(data.review.status, 'approved', 'DEV semantic review is not approved');
  assert.notEqual(data.review.reviewer, data.author);
  assert.equal(data.review.bundle_sha256, bundleHash, 'DEV changed after independent review');
}
const panels = ['fixed_prefix', 'closed_loop', ...(config.challenge === false ? [] : ['challenge'])];
const plan = panels.flatMap(panel => data.panels[panel]);
assert.equal(new Set(plan.map(row => row.case_id)).size, plan.length);
assert.equal(data.panels.fixed_prefix.length, 288);
assert.equal(data.panels.closed_loop.length, 898);
assert.equal(data.panels.closed_loop.filter(r => r.cohort === 'supported').length, 834);
assert.equal(data.panels.closed_loop.filter(r => r.cohort === 'outside_scope').length, 64);
assert.equal(data.panels.closed_loop.filter(r => !r.dialogue_id && r.cohort === 'supported').length, 194);
assert.equal(new Set(data.panels.closed_loop.filter(r => r.dialogue_id).map(r => r.dialogue_id)).size, 128);
for (const panel of panels) {
  const seen = new Map();
  for (const row of data.panels[panel]) {
    assert.equal(row.panel, panel);
    assert.equal(row.split, 'dev');
    assert.ok(row.family_id && row.semantic_obligations.length);
    assert.equal(typeof row.reference_eligible, 'boolean');
    if (row.dialogue_id) {
      assert.equal(row.turn_index, (seen.get(row.dialogue_id) ?? 0) + 1);
      seen.set(row.dialogue_id, row.turn_index);
    }
  }
  for (const length of seen.values()) assert.equal(length, panel === 'challenge' ? 8 : 5);
}
async function promptFor(before, question) {
  const stop = new Error('captured'); let prompt;
  try {
    await buildModelInput(registry, before, question, { tokenizer: { sha256: pinnedTokenizerSha256,
      encode: async (text, options) => { assert.equal(options.truncation, false); prompt = text; throw stop; } }, contextWindow: 32768, completionTokens: LIMIT });
  } catch (error) { if (error !== stop) throw error; }
  assert.equal(typeof prompt, 'string');
  return prompt;
}
async function materialize(before, resolved, generation) {
  if (generation.error) {
    assert.ok(['context_overflow','inference_failed'].includes(generation.error));
    return recordProviderFailure(registry, before, resolved.question, generation.error);
  }
  assert.equal(typeof generation.raw_text, 'string');
  const ids = generation.raw_ids;
  assert.ok(Array.isArray(ids) && ids.length > 0 && ids.length <= LIMIT);
  assert.ok(ids.every(id => Number.isInteger(id) && id >= 0 && id < 262144));
  if (ids.at(-1) !== STOP) {
    assert.equal(ids.length, LIMIT); assert.ok(!ids.includes(STOP));
    return recordIncompleteModelOutput(registry, before, resolved.question, generation.raw_text, ids, 'token_limit');
  }
  assert.ok(!ids.slice(0, -1).includes(STOP));
  return applyModelOutput(registry, before, resolved.question, generation.raw_text);
}
async function selftest() {
  const sessions = new Map(), records = [];
  let alternatives = 0, absentBranches = 0, wrongTargets = 0, atomicFailures = 0, wrongPendingBranches = 0, scopeMutations = 0;
  for (const row of plan) {
    const before = sessions.get(row.dialogue_id) ?? await harness.initial(registry, row);
    const resolved = harness.resolve(row, before);
    for (const calls of resolved.allowed_batches) {
      const after = await applyModelOutput(registry, before, resolved.question, encodeCalls(calls));
      const result = await harness.grade(registry, row, before, after, resolved);
      assert.equal(result.success, true, row.case_id);
      if (row.reference_eligible && resolved.reference_targets.length) assert.equal(result.reference_success, true);
      alternatives++;
    }
    const after = await applyModelOutput(registry, before, resolved.question, encodeCalls(resolved.allowed_batches[0]));
    records.push(await harness.grade(registry, row, before, after, resolved));
    if (row.dialogue_id) sessions.set(row.dialogue_id, after);
    if (row.reference_eligible) {
      const empty = await createSession(registry, { dialogueVersion: 2, sessionId: 'absent-branch' });
      const absent = harness.resolve(row, empty);
      const next = await applyModelOutput(registry, empty, absent.question, encodeCalls(absent.allowed_batches[0]));
      assert.equal((await harness.grade(registry, row, empty, next, absent)).success, true, `${row.case_id}/absent`);
      absentBranches++;
    }
    if (row.policy.kind.startsWith('pending_')) {
      let wrongPending = await createSession(registry, { dialogueVersion: 2, sessionId: 'wrong-pending-branch' });
      const operation = row.policy.function === 'get_units' ? 'get_default_value' : 'get_units';
      for (let n = 0; n < row.policy.target_exchange; n++) wrongPending = await applyModelOutput(registry, wrongPending, 'Please ask which parameter I mean.', encodeCalls([{ name: operation, arguments: { _clarify: true } }]));
      const alternate = harness.resolve(row, wrongPending);
      assert.equal(alternate.branch, 'absent');
      const next = await applyModelOutput(registry, wrongPending, alternate.question, encodeCalls(alternate.allowed_batches[0]));
      assert.equal((await harness.grade(registry, row, wrongPending, next, alternate)).success, true);
      assert.equal(next.semantic.pending.length, wrongPending.semantic.pending.length);
      wrongPendingBranches++;
    }
    const completeAnswers = resolved.allowed_batches.find(batch => batch.length > 1 && batch.every(c => c.name === 'repeat_answer'));
    if (completeAnswers) {
      const partial = await applyModelOutput(registry, before, resolved.question, encodeCalls([completeAnswers[0]]));
      const verdict = await harness.grade(registry, row, before, partial, resolved);
      assert.equal(verdict.success, false); assert.equal(verdict.critical_status, 'detected'); scopeMutations++;
    }
    if (row.policy.kind === 'repeat_answer') {
      const exchange = before.exchanges[row.policy.target_exchange - 1];
      const turn = before.semantic.turns.find(t => t.id === exchange?.semantic_turn_id);
      if (turn?.answers.length > 1) {
        const broad = await applyModelOutput(registry, before, resolved.question, encodeCalls([{ name: 'repeat_exchange', arguments: { target: exchange.id } }]));
        const verdict = await harness.grade(registry, row, before, broad, resolved);
        assert.equal(verdict.success, false); assert.equal(verdict.critical_status, 'detected'); scopeMutations++;
      }
    }
    if (row.panel === 'fixed_prefix' && row.policy.kind === 'quote') {
      const wrong = before.exchanges.find(ex => ex.id !== resolved.reference_targets[0]);
      const next = await applyModelOutput(registry, before, resolved.question, encodeCalls([{ name: 'repeat_exchange', arguments: { target: wrong.id } }]));
      const verdict = await harness.grade(registry, row, before, next, resolved);
      assert.equal(verdict.success, false); assert.equal(verdict.critical_status, 'detected'); wrongTargets++;
    }
    if (row.panel === 'fixed_prefix' && row.variant === 1) {
      const next = await applyModelOutput(registry, before, resolved.question, 'invalid candidate output');
      const verdict = await harness.grade(registry, row, before, next, resolved);
      assert.equal(verdict.success, false); assert.equal(verdict.failure_category, 'execution_error'); atomicFailures++;
    }
  }
  const duplicate = data.panels.fixed_prefix.find(row => row.group === 'ambiguous_address');
  const before = await harness.initial(registry, duplicate);
  assert.equal(before.exchanges[0].assistant, before.exchanges[1].assistant);
  const occurrence = { ...duplicate, question: 'Quote exactly the first reply.', policy: { kind: 'quote', target_exchange: 1 }, reference_eligible: true };
  const resolved = harness.resolve(occurrence, before);
  const wrongAfter = await applyModelOutput(registry, before, occurrence.question, encodeCalls([{ name: 'repeat_exchange', arguments: { target: 'e2' } }]));
  assert.equal(wrongAfter.exchanges.at(-1).assistant, before.exchanges[0].assistant);
  const wrongVerdict = await harness.grade(registry, occurrence, before, wrongAfter, resolved);
  assert.equal(wrongVerdict.success, false); assert.equal(wrongVerdict.critical_status, 'detected');
  const scores = Object.fromEntries(panels.map(panel => [panel, harness.summarize(data.panels[panel], records.filter(r => r.panel === panel))]));
  return { bundle_sha256: bundleHash, cases: plan.length, alternatives, absent_branches: absentBranches,
    wrong_target_mutations: wrongTargets, identical_text_wrong_target_rejected: true, wrong_pending_branches: wrongPendingBranches, incomplete_or_excess_quote_scope_rejected: scopeMutations, atomic_failure_mutations: atomicFailures, panels: scores, model_inference_performed: false };
}
async function readSaved(path) {
  let text;
  try { text = await readFile(path, 'utf8'); } catch (error) { if (error.code === 'ENOENT') return []; throw error; }
  const newline = text.lastIndexOf('\n');
  if (!text.endsWith('\n')) {
    const tail = text.slice(newline + 1);
    let valid = false;
    try { JSON.parse(tail); valid = true; } catch {}
    const handle = await open(path, 'r+');
    try {
      if (valid) { await handle.write('\n', Buffer.byteLength(text), 'utf8'); text += '\n'; }
      else { await handle.truncate(Buffer.byteLength(text.slice(0, newline + 1))); text = text.slice(0, newline + 1); }
      await handle.datasync();
    } finally { await handle.close(); }
  }
  return text.trim() ? text.trimEnd().split('\n').map(line => JSON.parse(line)) : [];
}
async function evaluate() {
  const saved = await readSaved(output), input = createInterface({ input: process.stdin, crlfDelay: Infinity }), replies = input[Symbol.asyncIterator]();
  assert.ok(saved.length <= plan.length, 'Observation file has extra rows');
  const handle = await open(output, 'a');
  const sessions = new Map(), results = [];
  try {
    for (let index = 0; index < plan.length; index++) {
      const row = plan[index];
      const before = sessions.get(row.dialogue_id) ?? await harness.initial(registry, row);
      const resolved = harness.resolve(row, before);
      const prompt = await promptFor(before, resolved.question);
      const promptHash = hash(prompt);
      let generation, previous = saved[index];
      if (previous) {
        assert.equal(previous.case_id, row.case_id, 'Observation order differs from frozen DEV');
        assert.equal(previous.bundle_sha256, bundleHash, 'Observation belongs to another DEV bundle');
        assert.equal(previous.prompt_sha256, promptHash, 'Saved prompt differs from actual resumed history');
        assert.equal(previous.before_visible_history_sha256, resolved.before_visible_history_sha256);
        generation = previous.generation;
      } else {
        await emit({ kind: 'generate', case_id: row.case_id, prompt });
        const reply = await replies.next();
        if (reply.done) throw new Error('Model provider disconnected; saved observations remain resumable');
        generation = JSON.parse(reply.value);
      }
      const after = await materialize(before, resolved, generation);
      const result = await harness.grade(registry, row, before, after, resolved);
      const observation = { ...result, bundle_sha256: bundleHash, prompt_sha256: promptHash,
        before_visible_history_sha256: resolved.before_visible_history_sha256, question: resolved.question,
        policy_branch: resolved.branch, expected_model_call_batches: resolved.allowed_batches,
        generation, before, after };
      if (previous) assert.equal(canonical(previous), canonical(observation), 'Saved observation is not exactly reproducible');
      else {
        const bytes = Buffer.from(`${JSON.stringify(observation)}\n`);
        const reserve = BigInt(reserveText);
        if (reserve > 0n) {
          const { bavail, bsize } = await statfs(dirname(output), { bigint: true });
          if (bavail * bsize < reserve + BigInt(bytes.length)) throw new Error('DEV free-space reserve exhausted; resume from saved observations');
        }
        await handle.write(bytes); await handle.datasync();
      }
      if (row.dialogue_id) sessions.set(row.dialogue_id, after);
      results.push(result);
    }
    const panelScores = Object.fromEntries(panels.map(panel => [panel, harness.summarize(data.panels[panel], results.filter(r => r.panel === panel))]));
    const summary = panelScores.closed_loop;
    await emit({ kind: 'result', score: { ...summary, summary, panels: panelScores, observed_records: results.length,
      bundle_sha256: bundleHash, diagnostic_reference_panel: 'fixed_prefix', review_status: 'automatic_conservative_unresolved_retained', release_ready: false } });
  } finally { input.close(); await handle.close(); }
}
if (output === '--selftest') await emit({ kind: 'selftest', ...(await selftest()) });
else if (output === '--check') await emit({ kind: 'check', cases: plan.length, panels: Object.fromEntries(panels.map(p => [p, data.panels[p].length])), bundle_sha256: bundleHash });
else await evaluate();
