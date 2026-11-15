import assert from 'node:assert/strict';
import { execFileSync, spawn } from 'node:child_process';
import { once } from 'node:events';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createInterface } from 'node:readline';
import test from 'node:test';
import { expandRecords, loadPanel, shardOf, summarize } from './diagnose-dev.mjs';
import { compactObservation, materializeGeneration } from './diagnose-data.mjs';
import { canonical } from './evaluation/schema.mjs';
import { createSession, applyModelOutput, sessionHistory } from './evaluation/session.mjs';
import { encodeCalls } from './evaluation/functiongemma.mjs';
import { buildModelInput, pinnedTokenizerSha256 } from './evaluation/model-input.mjs';
import { renderPrompt } from './evaluation/history-train.mjs';
import { createActualHistoryHarness, hash } from './evaluation/actual-history.mjs';

const root = dirname(fileURLToPath(import.meta.url));
const registryPath = join(root, 'assets/registry.json');
const panelPath = join(root, 'assets/actual-history-v3-dev.json');
const context = await loadPanel(registryPath, panelPath);
const api = { createSession, applyModelOutput, sessionHistory, encodeCalls, canonical,
  buildModelInput, pinnedTokenizerSha256 };
const harness = createActualHistoryHarness(api);

test('four shards cover all fixed and own-history DEV cases once while retaining every whole dialogue', () => {
  const partitions = Array.from({ length: 4 }, (_, shard) => context.plan.filter(row => shardOf(row) === shard));
  const allIds = partitions.flat().map(row => row.case_id);
  assert.equal(allIds.length, 1250);
  assert.equal(new Set(allIds).size, 1250);
  assert.deepEqual([...allIds].sort(), context.plan.map(row => row.case_id).sort());
  assert.ok(partitions.every(part => part.length > 0));
  const dialogues = new Map();
  for (const row of context.plan.filter(row => row.dialogue_id)) {
    const key = `${row.panel}:${row.dialogue_id}`;
    const group = dialogues.get(key) ?? [];
    group.push(row); dialogues.set(key, group);
  }
  assert.equal(dialogues.size, 136);
  for (const group of dialogues.values()) {
    const shard = shardOf(group[0]);
    assert.ok(group.every(row => shardOf(row) === shard));
    assert.deepEqual(group.map(row => row.turn_index), group[0].panel === 'challenge' ? [1, 2, 3, 4, 5, 6, 7, 8] : [1, 2, 3, 4, 5]);
    assert.deepEqual(partitions[shard].filter(row => row.dialogue_id === group[0].dialogue_id), group);
  }
  assert.ok(context.plan.every(row => shardOf(structuredClone(row), 4) === shardOf(row)));
});

test('summaries retain separate planned denominators, whole dialogues and conservative unresolved failures', () => {
  const records = context.plan.map((row, index) => ({ ...row, success: index % 17 !== 0,
    reference_success: index % 17 !== 0 && row.reference_eligible,
    critical_status: index % 17 !== 0 ? 'none_observed' : (index % 34 === 0 ? 'detected' : 'review_required'),
    critical_category: index % 34 === 0 ? 'wrong_reference' : null }));
  const actual = summarize(context.data, records);
  for (const panel of ['fixed_prefix', 'closed_loop', 'challenge']) {
    const selected = records.filter(row => row.panel === panel);
    assert.deepEqual(actual.panels[panel], harness.summarize(context.data.panels[panel], selected));
  }
  assert.equal(actual.summary.supported.planned, 834);
  assert.equal(actual.summary.outside_scope.planned, 64);
  assert.equal(actual.summary.whole_dialogues.planned, 128);
  assert.equal(actual.panels.fixed_prefix.supported.planned, 288);
  assert.equal(actual.panels.challenge.whole_dialogues.planned, 8);
  assert.equal(actual.observed_records, 1250);
  assert.equal(actual.release_ready, false);
  const reduced = summarize(context.data, records.filter(row => row.case_id !== context.plan[0].case_id));
  assert.equal(reduced.panels.fixed_prefix.missing, 1);
  assert.equal(reduced.panels.fixed_prefix.supported.planned, 288);
});

test('own-history continuation sees and quotes the actual incomplete reference reply', async () => {
  const rows = context.data.panels.closed_loop.filter(row => row.dialogue_id === 'dev-ah3-loop-quote_displayed-01');
  let state = await harness.initial(context.registry, rows[0]);
  const results = [];
  const foreignIds = [410000, 410001];
  for (const [index, row] of rows.entries()) {
    const before = state, resolved = harness.resolve(row, before);
    const generation = index === 1 ? { provider_kind: 'reference', complete: false,
      raw_text: '<unfinished teacher output>', raw_ids: foreignIds, finish_reason: 'length' } :
      { provider_kind: 'reference', complete: true, raw_text: encodeCalls(resolved.allowed_batches[0]), raw_ids: [900000] };
    if (index === 2) {
      const prompt = await renderPrompt(context.registry, api, before, resolved.question);
      assert.ok(prompt.includes('reliable answer to that request'));
      assert.equal(before.exchanges[1].outcome, 'error');
      assert.equal(before.exchanges[1].raw_output, '<unfinished teacher output>');
      assert.deepEqual(before.exchanges[1].generation.token_ids, foreignIds);
      assert.deepEqual(resolved.reference_targets, ['e2']);
    }
    state = await materializeGeneration(context.registry, before, resolved.question, generation);
    const result = await harness.grade(context.registry, row, before, state, resolved);
    results.push(result);
    if (index === 1) {
      assert.equal(result.success, false);
      assert.deepEqual(state.semantic, before.semantic);
    } else assert.equal(result.success, true);
    if (index === 2) assert.equal(state.exchanges[2].assistant, state.exchanges[1].assistant);
  }
  const score = harness.summarize(rows, results);
  assert.equal(score.supported.passed, 4);
  assert.equal(score.whole_dialogues.passed, 0);
  assert.equal(score.missing, 0);
});

test('CLI plan is identical and summary rejects a reordered or incomplete merged stream', async () => {
  const script = join(root, 'diagnose-dev.mjs');
  const planned = JSON.parse(execFileSync(process.execPath, [script, registryPath, panelPath, '--plan'],
    { encoding: 'utf8', maxBuffer: 4 * 1024 * 1024 }));
  assert.equal(planned.rows.length, 1250);
  assert.deepEqual(planned.rows.map(row => row.case_id), context.plan.map(row => row.case_id));
  assert.ok(planned.rows.every((row, index) => row.shard === shardOf(context.plan[index])));
  const temporary = process.env.TMPDIR;
  assert.ok(temporary, 'Run tests through ./cluster to confine temporary files');
  const where = relative(join(root, '.runtime'), resolve(temporary));
  assert.ok(where && !where.startsWith('..') && !where.startsWith('/'), 'Tests require the private runtime temporary directory');
  const directory = await mkdtemp(join(temporary, 'diagnose-dev-test-'));
  try {
    const output = join(directory, 'records.jsonl');
    const records = context.plan.map(row => ({ case_id: row.case_id, panel: row.panel,
      bundle_sha256: context.bundleHash, success: true, reference_success: Boolean(row.reference_eligible),
      critical_status: 'none_observed' }));
    const save = () => writeFile(output, records.map(row => JSON.stringify(row)).join('\n') + '\n');
    await save();
    const score = JSON.parse(execFileSync(process.execPath, [script, registryPath, panelPath, '--summarize', output],
      { encoding: 'utf8', maxBuffer: 4 * 1024 * 1024 }));
    assert.equal(score.score.summary.supported.passed, 834);
    assert.equal(score.score.summary.whole_dialogues.passed, 128);
    [records[0], records[1]] = [records[1], records[0]]; await save();
    assert.throws(() => execFileSync(process.execPath, [script, registryPath, panelPath, '--summarize', output],
      { stdio: 'pipe' }));
    records.pop(); await save();
    assert.throws(() => execFileSync(process.execPath, [script, registryPath, panelPath, '--summarize', output],
      { stdio: 'pipe' }));
    assert.ok((await readFile(output, 'utf8')).endsWith('\n'));
  } finally { await rm(directory, { recursive: true }); }
});

test('compact own-history DEV expands and audits complete dialogues with unchanged raw evidence', async () => {
  const plan = context.data.panels.closed_loop.filter(row => row.dialogue_id === 'dev-ah3-loop-quote_displayed-01');
  let before = await harness.initial(context.registry, plan[0]);
  const full = [], compact = [], binding = 'b'.repeat(64);
  for (const [index, row] of plan.entries()) {
    const resolved = harness.resolve(row, before), prompt = await renderPrompt(context.registry, api, before, resolved.question);
    const generation = index === 1 ? { provider_kind: 'reference', complete: false,
      raw_text: 'unfinished', raw_ids: [420000], finish_reason: 'length', diagnostic_worker_sha256: binding } :
      { provider_kind: 'reference', complete: true, raw_text: encodeCalls(resolved.allowed_batches[0]),
        raw_ids: [420001], diagnostic_worker_sha256: binding };
    const after = await materializeGeneration(context.registry, before, resolved.question, generation);
    const graded = await harness.grade(context.registry, row, before, after, resolved);
    const record = { ...graded, bundle_sha256: context.bundleHash, prompt_sha256: hash(prompt),
      before_visible_history_sha256: resolved.before_visible_history_sha256, question: resolved.question,
      policy_branch: resolved.branch, expected_model_call_batches: resolved.allowed_batches,
      generation, before, after };
    full.push(record); compact.push(compactObservation(record)); before = after;
  }
  const expanded = [];
  for await (const row of expandRecords(context, compact, { binding })) expanded.push(row);
  assert.deepEqual(expanded, full);
  assert.deepEqual(summarize(context.data, compact), summarize(context.data, full));
  assert.ok(JSON.stringify(compact).length < JSON.stringify(full).length);
  assert.equal(compact[1].delivered.outcome, 'error');
  assert.deepEqual(compact[1].generation.raw_ids, [420000]);
  for (const mutate of [
    rows => { rows[1].after_semantic_sha256 = '0'.repeat(64); },
    rows => { rows[2].generation.raw_text = ''; },
    rows => { rows.splice(0, 1); },
    rows => { [rows[0], rows[1]] = [rows[1], rows[0]]; },
  ]) {
    const wrong = structuredClone(compact); mutate(wrong);
    await assert.rejects(async () => { for await (const row of expandRecords(context, wrong)) void row; });
  }
  await assert.rejects(async () => {
    for await (const row of expandRecords(context, compact, { binding: 'c'.repeat(64) })) void row;
  }, /another provider/);
  const temporary = process.env.TMPDIR;
  assert.ok(temporary && !relative(join(root, '.runtime'), resolve(temporary)).startsWith('..'));
  const directory = await mkdtemp(join(temporary, 'diagnose-audit-test-'));
  try {
    const path = join(directory, 'compact.jsonl');
    const bytes = compact.map(row => JSON.stringify(row)).join('\n') + '\n';
    await writeFile(path, bytes);
    const script = join(root, 'diagnose-dev.mjs');
    const audit = JSON.parse(execFileSync(process.execPath, [script, registryPath, panelPath, '--audit', path],
      { encoding: 'utf8' }));
    assert.equal(audit.observations, 5);
    assert.equal(audit.model_inference_performed, false);
    const output = execFileSync(process.execPath, [script, registryPath, panelPath, '--expand', path, compact[2].case_id],
      { encoding: 'utf8', maxBuffer: 4 * 1024 * 1024 }).trimEnd().split('\n').map(JSON.parse);
    assert.equal(output.length, 2);
    assert.deepEqual(output[0].observation, full[2]);
    assert.equal(output[1].kind, 'audit');
    assert.equal(await readFile(path, 'utf8'), bytes);
  } finally { await rm(directory, { recursive: true }); }
});

test('relay persists only through its coordinator and resumes the verified compact prefix', async () => {
  const temporary = process.env.TMPDIR;
  assert.ok(temporary && !relative(join(root, '.runtime'), resolve(temporary)).startsWith('..'));
  const directory = await mkdtemp(join(temporary, 'diagnose-relay-test-'));
  const path = join(directory, 'receipts.jsonl'), binding = 'd'.repeat(64);
  const selected = context.plan.filter(row => shardOf(row) === 0);
  let active;
  const start = () => {
    const child = spawn(process.execPath, [join(root, 'diagnose-dev.mjs'), registryPath, panelPath,
      path, '0', '4', '536870912', binding, '--relay-persist'], { stdio: ['pipe', 'pipe', 'pipe'] });
    const input = createInterface({ input: child.stdout, crlfDelay: Infinity });
    const records = input[Symbol.asyncIterator]();
    let errors = '';
    child.stderr.on('data', bytes => { errors += bytes; });
    const completed = once(child, 'exit');
    return { child, input, completed,
      next: async () => {
        const row = await records.next();
        assert.equal(row.done, false, errors);
        return JSON.parse(row.value);
      },
      send: value => child.stdin.write(`${JSON.stringify(value)}\n`),
      stop: async () => { child.kill('SIGTERM'); await completed; input.close(); } };
  };
  try {
    active = start();
    const request = await active.next();
    assert.equal(request.kind, 'generate');
    assert.equal(request.case_id, selected[0].case_id);
    await assert.rejects(readFile(path), { code: 'ENOENT' });
    const before = await harness.initial(context.registry, selected[0]);
    const resolved = harness.resolve(selected[0], before);
    active.send({ provider_kind: 'reference', complete: true,
      raw_text: encodeCalls(resolved.allowed_batches[0]), raw_ids: [420001], diagnostic_worker_sha256: binding });
    const persist = await active.next();
    assert.equal(persist.kind, 'persist');
    assert.equal(persist.observation.case_id, selected[0].case_id);
    assert.equal(Object.hasOwn(persist.observation, 'after'), false);
    assert.match(persist.observation.delivered_sha256, /^[a-f0-9]{64}$/);
    await assert.rejects(readFile(path), { code: 'ENOENT' });
    const bytes = `${JSON.stringify(persist.observation)}\n`;
    await writeFile(path, bytes);
    active.send({ kind: 'persisted', case_id: request.case_id });
    assert.equal((await active.next()).case_id, selected[1].case_id);
    await active.stop(); active = null;
    active = start();
    const resumed = await active.next();
    assert.equal(resumed.kind, 'generate');
    assert.equal(resumed.case_id, selected[1].case_id);
    assert.equal(await readFile(path, 'utf8'), bytes);
  } finally {
    if (active) await active.stop();
    await rm(directory, { recursive: true });
  }
});
