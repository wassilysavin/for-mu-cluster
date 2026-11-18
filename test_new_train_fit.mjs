import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { createScorer } from './new_train_fit.mjs';
import { continuation, materializeScenario, renderPrompt } from './evaluation/history-train.mjs';
import * as native from './evaluation/functiongemma.mjs';
import * as modelInput from './evaluation/model-input.mjs';
import * as session from './evaluation/session.mjs';

const bundle = JSON.parse(await readFile(new URL('./assets/registry.json', import.meta.url)));
const scorer = await createScorer(bundle);
const api = { ...native, ...modelInput, ...session };
const call = (name, args) => ({ name, arguments: args });
const domain = call('explain_purpose', { subject: 'TR' });
const roots = await lines('transfer-train-roots.jsonl');

async function lines(file) {
  return (await readFile(new URL('./assets/' + file, import.meta.url), 'utf8')).trimEnd().split('\n').map(JSON.parse);
}
async function bind(before, row, options = {}) {
  const target = native.encodeCalls(row.calls);
  const revisedBefore = { ...before, registry_id: await scorer.registry.fingerprint() };
  const prompt = await renderPrompt(scorer.registry, api, revisedBefore, row.question);
  const request = { case_id: row.case_id, prompt, target };
  const config = { section: 'variable', branch: 'frozen_train', ...options };
  const meta = await scorer.prepare(request, row, before, config);
  return { meta, config, request, row, before };
}
function generated(meta, output, complete = true) {
  return { raw_text: output, raw_ids: complete ? [1, 50] : Array(512).fill(1), complete, provider_kind: 'functiongemma',
    max_new_tokens: 512, input_sha256: meta.prompt_sha256, finish_reason: complete ? 'complete_batch' : 'length' };
}
async function scored(meta, calls) {
  return (await scorer.score(meta.case_id, generated(meta, native.encodeCalls(calls)))).observation;
}

test('program equivalence requires both the displayed answer and the correct state', async () => {
  let before = await session.createSession(scorer.registry, { dialogueVersion: 2, sessionId: 'equal-text' });
  for (const q of ['What does TR mean?', 'What does TR mean here?']) {
    before = await session.applyModelOutput(scorer.registry, before, q, native.encodeCalls([domain]));
  }
  const row = { case_id: 'addresses', group: 'references', question: 'Quote my first reply.',
    calls: [call('repeat_exchange', { target: 'e1' })] };
  const { meta, request, config } = await bind(before, row);
  assert.equal((await scored(meta, row.calls)).canonical_call_exact, true);
  const otherProgram = await scored(meta, [call('repeat_answer', { target: 't1.c1' })]);
  assert.equal(otherProgram.success, true);
  assert.equal(otherProgram.canonical_call_exact, false);
  const wrong = await scored(meta, [call('repeat_answer', { target: 't2.c1' })]);
  assert.equal(wrong.success, false);
  assert.equal(wrong.critical_category, 'wrong_reference');
  await assert.rejects(scorer.prepare({ ...request, target: native.encodeCalls([domain]) }, row, before, config), /target differs/);
  await assert.rejects(scorer.prepare({ ...request, prompt: request.prompt.replace(row.question, 'Other question') },
    row, before, config), /prompt differs/);
  await assert.rejects(scorer.score(row.case_id, { ...generated(meta, request.target), input_sha256: 'other' }), /different TRAIN input/);
  const truncated = await scorer.score(row.case_id, generated(meta, request.target, false));
  assert.equal(truncated.observation.success, false);
  assert.equal(truncated.observation.transport_valid, false);
});

test('migrated shared TRAIN scores a real failed history exactly as it was displayed', async () => {
  const pairs = bundle.contracts.bindings.filter(b => b.call.name === 'explain_parameter_data_matching').map(b => b.call.arguments);
  const mismatch = pairs.flatMap(a => pairs.map(b => ({ subject: a.subject, reference: b.reference })))
    .find(v => !pairs.some(p => p.subject === v.subject && p.reference === v.reference));
  const raw = native.encodeCalls([call('explain_parameter_data_matching', mismatch)]);
  const old = await session.createSession(scorer.original, { dialogueVersion: 2, sessionId: 'migrated' });
  const before = await session.applyModelOutput(scorer.original, old, 'Explain the matching.', raw);
  assert.equal(before.exchanges[0].error_code, 'contract_not_implemented');
  const replayed = await session.applyModelOutput(scorer.registry, { ...old, registry_id: await scorer.registry.fingerprint() },
    'Explain the matching.', raw);
  assert.equal(replayed.exchanges[0].error_code, 'invalid_arguments');
  const row = { case_id: 'shared:migrated', group: 'shared_replay', question: 'Quote your first reply.',
    calls: [call('repeat_exchange', { target: 'e1' })] };
  const { meta } = await bind(before, row, { migrated: true, section: 'shared' });
  assert.equal((await scored(meta, row.calls)).success, true);
});

test('all authored TRAIN groups bind to frozen continuations from both arms', async () => {
  for (const arm of ['ordinary', 'contrast']) {
    const seen = new Set();
    for (const row of await lines('transfer-train-' + arm + '.jsonl')) {
      const root = roots.find(r => r.root_id === row.root_id);
      if (root.observed_prefix_id || seen.has(root.group)) continue;
      const { before } = await materializeScenario(scorer.registry, api, { ...root, session_id: root.root_id,
        case_id: row.case_id, continuation: row.continuation });
      const policy = continuation(before, row.continuation);
      const { meta } = await bind(before, { ...row, ...policy, group: root.group }, { branch: policy.branch });
      assert.ok(!Object.hasOwn(meta, 'target') && !Object.hasOwn(meta, 'calls'));
      assert.equal((await scored(meta, policy.calls)).success, true);
      seen.add(root.group);
    }
    assert.deepEqual(seen, new Set(roots.filter(r => !r.observed_prefix_id).map(r => r.group)));
  }
});

test('absent-answer correction applies to an actual missing answer, preserving ambiguous review', async () => {
  let before = await session.createSession(scorer.registry, { dialogueVersion: 2, sessionId: 'absent' });
  before = await session.applyModelOutput(scorer.registry, before, 'A prior error.', 'not a function batch');
  before = await session.applyModelOutput(scorer.registry, before, 'A completed reply.', native.encodeCalls([domain]));
  const row = { group: 'absent', question: 'Repeat the scientific answer from my first request.',
    calls: [call('clarify_reference', { kind: 'answer' })] };
  const candidate = [call('repeat_answer', { target: 't1.c1' })];
  const missing = await bind(before, { ...row, case_id: 'missing' }, { branch: 'requested_answer_missing' });
  assert.equal((await scored(missing.meta, candidate)).critical_category, 'wrong_reference');
  const ambiguous = await bind(before, { ...row, case_id: 'ambiguous', question: 'Repeat that answer.' },
    { branch: 'preauthored_ambiguous_reference' });
  assert.equal((await scored(ambiguous.meta, candidate)).critical_status, 'review_required');
});
