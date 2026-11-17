import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { createCaseInspector } from './inspect_qwen_cases.mjs';
import { compactObservation, createTrainDiagnostics } from './diagnose-data.mjs';
import { createRegistry } from './evaluation/registry.mjs';
import { canonical } from './evaluation/schema.mjs';
import { createSession, applyModelOutput, recordProviderFailure, recordIncompleteModelOutput,
  sessionHistory } from './evaluation/session.mjs';
import { encodeCalls } from './evaluation/functiongemma.mjs';
import { buildModelInput, pinnedTokenizerSha256 } from './evaluation/model-input.mjs';
import { renderPrompt } from './evaluation/history-train.mjs';

const bundle = JSON.parse(await readFile(new URL('./assets/registry.json', import.meta.url), 'utf8'));
const registry = createRegistry(bundle.knowledge, bundle.contracts);
const diagnostic = createTrainDiagnostics(registry);
const inspect = createCaseInspector(registry, bundle.contracts);
const api = { createSession, applyModelOutput, recordProviderFailure, recordIncompleteModelOutput,
  sessionHistory, encodeCalls, canonical, buildModelInput, pinnedTokenizerSha256 };
const call = (name, args) => ({ name, arguments: args });
const defaultTR = call('get_default_value', { subject: 'TR' });
const fresh = () => createSession(registry, { dialogueVersion: 2, sessionId: 'fixture' });

async function fixture(before, expected, actual, question = 'Fixture request for the declared operation.') {
  const row = { case_id: 'inspection-fixture', family_id: 'fixture-family', group: 'fixture',
    question, before, calls: expected,
    after: await applyModelOutput(registry, before, question, encodeCalls(expected)),
    model_input: await renderPrompt(registry, api, before, question) };
  const item = await diagnostic.prepare({ source: 'history', row });
  const full = await diagnostic.score(item, { provider_kind: 'reference', complete: true, raw_text: encodeCalls(actual) });
  return { item, receipt: compactObservation(full), full };
}

test('first failing model call reports the actual dynamic reference schema', async () => {
  const before = await applyModelOutput(registry, await fresh(), 'Default TR?', encodeCalls([defaultTR]));
  const sample = await fixture(before, [defaultTR], [defaultTR,
    call('get_default_value', { subject: 'TR', _replace: 't99.c1' })]);
  const found = await inspect(sample.item, sample.receipt);
  assert.equal(found.trace.stage, 'compile');
  assert.equal(found.trace.call_index, 1);
  assert.equal(found.trace.error.message, '$._replace: value is not in the declared domain');
  const schema = found.schemas.find(entry => entry.source === 'dynamic_model_tool');
  assert.equal(schema.valid, false);
  assert.equal(schema.path, '$._replace');
  assert.deepEqual(schema.fragment.enum, ['t1.c1']);
  assert.equal(found.scores_changed, false);
});

test('domain required arguments remain enforced after the exposed partial schema passes', async () => {
  const sample = await fixture(await fresh(), [defaultTR], [call('get_default_value', {})]);
  const found = await inspect(sample.item, sample.receipt);
  assert.equal(found.trace.error.message, '$: missing subject');
  assert.equal(found.schemas.find(entry => entry.source === 'dynamic_model_tool').valid, true);
  const domain = found.schemas.find(entry => entry.source === 'domain_contract');
  assert.equal(domain.valid, false);
  assert.deepEqual(domain.fragment.required, ['subject']);
  assert.equal(domain.fragment.field, 'subject');
});

test('metadata semantic restrictions preserve the real compiler message', async () => {
  const sample = await fixture(await fresh(), [defaultTR], [call('get_default_value', {
    subject: 'TR', _missing: ['subject'],
  })]);
  const found = await inspect(sample.item, sample.receipt);
  assert.equal(found.trace.error.message, 'Missing-argument metadata requires explicit clarification.');
  assert.equal(found.schemas[0].valid, true);
});

test('same-call replacement is an execution rule violation, not a schema rejection', async () => {
  const before = await applyModelOutput(registry, await fresh(), 'Default TR?', encodeCalls([defaultTR]));
  const sample = await fixture(before, [defaultTR], [call('get_default_value', { subject: 'TR', _replace: 't1.c1' })]);
  const found = await inspect(sample.item, sample.receipt);
  assert.equal(found.trace.stage, 'execute');
  assert.equal(found.trace.call_index, 0);
  assert.equal(found.trace.error.message, 'Replacement does not change the call.');
  assert.ok(found.schemas.every(schema => schema.valid));
});

test('retry full batch exposes the unbound reconstructed domain call after its first operation', async () => {
  const before = await recordProviderFailure(registry, await fresh(), 'Original request.', 'inference_failed');
  const unbound = call('get_value_guidance', { subject: 'TR', preprocessing: 'fMRIPrep' });
  const sample = await fixture(before, [call('retry_exchange', { target: 'e1' }), defaultTR],
    [call('retry_exchange', { target: 'e1' }), unbound]);
  assert.equal(sample.full.error_kind, 'contract_not_implemented');
  const found = await inspect(sample.item, sample.receipt);
  assert.equal(found.trace.stage, 'execute');
  assert.equal(found.trace.call_index, 1);
  assert.deepEqual(found.contract.domain_call, unbound);
  assert.equal(found.contract.schema_valid, true);
  assert.equal(found.contract.exact_binding_present, false);
  assert.ok(found.contract.same_function_count > found.contract.same_function_subject_count);
  assert.ok(found.contract.same_function_subject_count > 0);
  assert.ok(found.contract.same_function_examples.length <= 3);
});

test('change operation reports merged retained arguments in the unbound domain call', async () => {
  const before = await applyModelOutput(registry, await fresh(), 'Choose TR.',
    encodeCalls([call('get_value_guidance', { subject: 'TR' })]));
  const sample = await fixture(before, [defaultTR], [call('change_get_value_guidance', {
    target: 't1.c1', changes: { preprocessing: 'fMRIPrep' },
  })]);
  const found = await inspect(sample.item, sample.receipt);
  assert.equal(found.trace.stage, 'execute');
  assert.deepEqual(found.contract.domain_call, call('get_value_guidance', { subject: 'TR', preprocessing: 'fMRIPrep' }));
  assert.equal(found.schemas[0].name, 'change_get_value_guidance');
  assert.equal(found.schemas[1].name, 'get_value_guidance');
});

test('group completion does not claim an unsafe prefix-based error index', async () => {
  const before = await applyModelOutput(registry, await fresh(), 'Choose values for unspecified parameters.',
    encodeCalls([call('get_value_guidance', { _clarify: true, _each: 'subject' })]));
  assert.equal(before.exchanges.at(-1).outcome, 'committed');
  const sample = await fixture(before, [defaultTR], [call('get_value_guidance', {
    _resolve: 't1.p1', subject: 'TR', preprocessing: 'fMRIPrep',
  })]);
  const found = await inspect(sample.item, sample.receipt);
  assert.equal(found.trace.stage, 'execute');
  assert.equal(found.trace.call_index, null);
  assert.match(found.trace.attribution, /group_finalization/);
  assert.equal(found.trace.error.code, 'invalid_arguments');
});

test('pending completion does not invent a missing domain argument supplied by prior state', async () => {
  const before = await applyModelOutput(registry, await fresh(), 'Choose TR after unspecified preprocessing.',
    encodeCalls([call('get_value_guidance', { subject: 'TR', _clarify: true, _missing: ['preprocessing'] })]));
  const sample = await fixture(before, [defaultTR], [call('get_value_guidance', {
    _resolve: 't1.p1', refinement_enabled: true,
  })]);
  const found = await inspect(sample.item, sample.receipt);
  assert.equal(found.trace.error.message,
    'Completion may supply only missing arguments; changing known arguments requires an explicit replacement.');
  assert.equal(found.schemas.length, 1);
  assert.equal(found.schemas[0].source, 'dynamic_model_tool');
  assert.equal(found.schemas[0].valid, true);
});

test('review compares real delivered answers and state while preserving unresolved status', async () => {
  const expected = call('explain_item_count_limits', { subject: 'concatenate_item', bound: 'maximum' });
  const sample = await fixture(await fresh(), [expected], [call('get_default_value', { subject: 'concatenate_item' })]);
  const saved = canonical(sample.receipt);
  const found = await inspect(sample.item, sample.receipt);
  assert.equal(found.critical_status, 'review_required');
  assert.equal(found.review.adjudicated, false);
  assert.equal(found.trace.stage, 'committed');
  assert.equal(found.review.alternatives[0].delivered.equal, false);
  assert.equal(found.review.alternatives[0].semantic.equal, false);
  assert.ok(found.review.alternatives[0].semantic.entries.length > 0);
  assert.deepEqual(found.review.alternatives[0].calls, [expected]);
  const expectedAfter = await applyModelOutput(registry, sample.item.before, sample.item.question, encodeCalls([expected]));
  assert.ok(expectedAfter.exchanges.at(-1).assistant.length > 240);
  assert.equal(found.review.actual.delivered, sample.full.after.exchanges.at(-1).assistant);
  assert.deepEqual(found.review.actual.semantic, sample.full.after.semantic);
  assert.equal(found.review.alternatives[0].expected.delivered, expectedAfter.exchanges.at(-1).assistant);
  assert.deepEqual(found.review.alternatives[0].expected.semantic, expectedAfter.semantic);
  assert.equal(canonical(sample.receipt), saved);
});

test('identical delivered text does not hide a different reference/action state', async () => {
  const before = await applyModelOutput(registry, await fresh(), 'Default TR?', encodeCalls([defaultTR]));
  const sample = await fixture(before, [defaultTR], [call('repeat_answer', { target: 't1.c1' })], 'Answer afresh.');
  assert.equal(sample.full.critical_status, 'review_required');
  const found = await inspect(sample.item, sample.receipt);
  assert.equal(found.review.alternatives[0].delivered.equal, true);
  assert.equal(found.review.alternatives[0].semantic.equal, false);
  assert.equal(found.review.adjudicated, false);
});

test('altered receipts and cases are rejected before reporting', async () => {
  const sample = await fixture(await fresh(), [defaultTR], [call('get_default_value', {})]);
  const changed = structuredClone(sample.receipt);
  changed.error_kind = 'contract_not_implemented';
  await assert.rejects(() => inspect(sample.item, changed), /receipt does not reproduce/);
  const badCase = structuredClone(sample.item);
  badCase.question += 'changed';
  await assert.rejects(() => inspect(badCase, sample.receipt), /changed/);
});

test('line JSON interface inspects saved evidence without loading a model', async () => {
  const sample = await fixture(await fresh(), [defaultTR], [call('get_default_value', {})]);
  const result = spawnSync(process.execPath, [fileURLToPath(new URL('./inspect_qwen_cases.mjs', import.meta.url)),
    fileURLToPath(new URL('./assets/registry.json', import.meta.url))], {
    input: `${JSON.stringify({ action: 'inspect', case: sample.item, observation: sample.receipt })}\n`,
    encoding: 'utf8', maxBuffer: 1024 * 1024,
  });
  assert.equal(result.status, 0, result.stderr);
  const output = JSON.parse(result.stdout);
  assert.equal(output.kind, 'inspection');
  assert.equal(output.inspection.trace.error.code, 'invalid_arguments');
});
