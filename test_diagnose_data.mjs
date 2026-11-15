import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { createTrainDiagnostics, materializeGeneration } from './diagnose-data.mjs';
import { createRegistry } from './evaluation/registry.mjs';
import { canonical } from './evaluation/schema.mjs';
import { createSession, applyModelOutput, recordProviderFailure, recordIncompleteModelOutput,
  sessionHistory } from './evaluation/session.mjs';
import { encodeCalls } from './evaluation/functiongemma.mjs';
import { buildModelInput, pinnedTokenizerSha256 } from './evaluation/model-input.mjs';
import { materializeScenario, renderPrompt } from './evaluation/history-train.mjs';

const bundle = JSON.parse(await readFile(new URL('./assets/registry.json', import.meta.url), 'utf8'));
const registry = createRegistry(bundle.knowledge, bundle.contracts);
const diagnostic = createTrainDiagnostics(registry);
const api = { createSession, applyModelOutput, recordProviderFailure, recordIncompleteModelOutput,
  sessionHistory, encodeCalls, canonical, buildModelInput, pinnedTokenizerSha256 };
const call = (name, args) => ({ name, arguments: args });
const complete = calls => ({ provider_kind: 'reference', complete: true, raw_text: encodeCalls(calls) });

async function historyCase(before, question, calls, id = 'test-history') {
  const row = { case_id: id, family_id: 'test-family', group: 'test', question, before, calls,
    after: await applyModelOutput(registry, before, question, encodeCalls(calls)),
    model_input: await renderPrompt(registry, api, before, question) };
  return diagnostic.prepare({ source: 'history', row });
}

test('frozen common prompt is reproduced, and native exact differs from canonical call matching', async () => {
  const before = await createSession(registry, { dialogueVersion: 2, sessionId: 'empty' });
  const calls = [call('get_default_value', { subject: 'TR' })];
  const row = { case_id: 'common-one', question: 'What is the default TR?', calls };
  const prompt = await renderPrompt(registry, api, before, row.question);
  const item = await diagnostic.prepare({ source: 'common', row, prompt });
  assert.equal(item.semantic_eligible, true);
  const result = await diagnostic.score(item, { ...complete(calls), raw_text: `\n${encodeCalls(calls)}\n` });
  assert.equal(result.native_literal_exact, false);
  assert.equal(result.canonical_call_exact, true);
  assert.equal(result.semantic_success, true);
});

test('unrestorable historical common input retains native fit but never claims semantic coverage', async () => {
  let before = await createSession(registry, { dialogueVersion: 2, sessionId: 'historical-common' });
  before = await applyModelOutput(registry, before, 'What is TR?', encodeCalls([call('explain_purpose', { subject: 'TR' })]));
  const calls = [call('repeat_answer', { target: 't1.c1' })];
  const row = { case_id: 'common-two', question: 'Repeat the first answer.', calls };
  const prompt = await renderPrompt(registry, api, before, row.question);
  const item = await diagnostic.prepare({ source: 'common', row, prompt });
  assert.equal(item.semantic_eligible, false);
  const scored = await diagnostic.score(item, complete(calls));
  assert.equal(scored.native_literal_exact, true);
  assert.equal(scored.semantic_success, null);
  assert.equal(scored.semantic_result, null);
  const restored = await diagnostic.prepare({ source: 'common', row, prompt, before });
  assert.equal(restored.semantic_eligible, true);
});

test('equivalent whole-reply quoting passes but the same text from another occurrence fails', async () => {
  let before = await createSession(registry, { dialogueVersion: 2, sessionId: 'duplicates' });
  const calls = [call('explain_purpose', { subject: 'TR' })];
  before = await applyModelOutput(registry, before, 'What is TR?', encodeCalls(calls));
  before = await applyModelOutput(registry, before, 'Again, what is TR?', encodeCalls(calls));
  const item = await historyCase(before, 'Quote exactly the first reply.', [call('repeat_exchange', { target: 'e1' })]);
  const equivalent = await diagnostic.score(item, complete([call('repeat_answer', { target: 't1.c1' })]));
  assert.equal(equivalent.native_literal_exact, false);
  assert.equal(equivalent.semantic_success, true);
  const wrong = await diagnostic.score(item, complete([call('repeat_exchange', { target: 'e2' })]));
  assert.equal(wrong.after.exchanges.at(-1).assistant, equivalent.after.exchanges.at(-1).assistant);
  assert.equal(wrong.semantic_success, false);
  assert.equal(wrong.semantic_result.critical_status, 'detected');
  assert.equal(wrong.semantic_result.critical_category, 'wrong_reference');
});

test('whole-reply scope rejects one answer and accepts all answers in order', async () => {
  let before = await createSession(registry, { dialogueVersion: 2, sessionId: 'whole-reply' });
  before = await applyModelOutput(registry, before, 'Explain TR and its default.', encodeCalls([
    call('explain_purpose', { subject: 'TR' }), call('get_default_value', { subject: 'TR' }),
  ]));
  const item = await historyCase(before, 'Quote the complete first reply.', [call('repeat_exchange', { target: 'e1' })]);
  const partial = await diagnostic.score(item, complete([call('repeat_answer', { target: 't1.c1' })]));
  assert.equal(partial.semantic_success, false);
  const full = await diagnostic.score(item, complete([
    call('repeat_answer', { target: 't1.c1' }), call('repeat_answer', { target: 't1.c2' }),
  ]));
  assert.equal(full.semantic_success, true);
});

test('change and explicit replacement are valid alternatives for the same addressed answer', async () => {
  let before = await createSession(registry, { dialogueVersion: 2, sessionId: 'revision' });
  before = await applyModelOutput(registry, before, 'What is TR?', encodeCalls([call('explain_purpose', { subject: 'TR' })]));
  const item = await historyCase(before, 'Change the parameter in that answer to BIDS.basepath.',
    [call('change_explain_purpose', { target: 't1.c1', changes: { subject: 'BIDS.basepath' } })]);
  const result = await diagnostic.score(item, complete([call('explain_purpose', { subject: 'BIDS.basepath', _replace: 't1.c1' })]));
  assert.equal(result.semantic_success, true);
  assert.equal(result.canonical_call_exact, false);
  const wrong = await diagnostic.score(item, complete([call('explain_purpose', { subject: 'BIDS.basepath' })]));
  assert.equal(wrong.semantic_success, false);
  assert.equal(wrong.semantic_result.critical_category, 'omitted_reference');
});

test('tampered state, question or prepared case is rejected', async () => {
  let before = await createSession(registry, { dialogueVersion: 2, sessionId: 'authenticity' });
  before = await applyModelOutput(registry, before, 'What is TR?', encodeCalls([call('explain_purpose', { subject: 'TR' })]));
  const bad = structuredClone(before); bad.exchanges[0].assistant += 'fabricated';
  await assert.rejects(() => historyCase(bad, 'Repeat it.', [call('repeat_exchange', { target: 'e1' })]), /does not reproduce/);
  const item = await historyCase(before, 'Repeat it.', [call('repeat_exchange', { target: 'e1' })]);
  const changed = structuredClone(item); changed.gold_calls[0].arguments.target = 'e2';
  await assert.rejects(() => diagnostic.score(changed, complete(item.gold_calls)), /changed/);
  await assert.rejects(() => diagnostic.prepare({ source: 'common', row: { case_id: 'bad-question',
    question: 'Different question.', calls: item.gold_calls }, prompt: item.prompt }), /question differs/);
});

test('incomplete reference output preserves original tokens without FunctionGemma token assumptions', async () => {
  const before = await createSession(registry, { dialogueVersion: 2, sessionId: 'incomplete' });
  const calls = [call('explain_purpose', { subject: 'TR' })];
  const item = await historyCase(before, 'What is TR?', calls);
  const generation = { provider_kind: 'reference', complete: false, raw_text: encodeCalls(calls),
    raw_ids: [999999, 1000000], finish_reason: 'length' };
  const result = await diagnostic.score(item, generation);
  assert.equal(result.semantic_success, false);
  assert.equal(result.native_literal_exact, false);
  assert.equal(result.after.exchanges.at(-1).error_code, 'incomplete_model_output');
  assert.deepEqual(result.after.exchanges.at(-1).generation.token_ids, generation.raw_ids);
  assert.equal(result.after.exchanges.at(-1).raw_output, generation.raw_text);
  assert.deepEqual(result.after.semantic, before.semantic);
  const failure = await materializeGeneration(registry, before, 'What is TR?', { error: 'inference_failed' });
  assert.equal(failure.exchanges.at(-1).outcome, 'provider_error');
});

test('preauthored scenario groups reproduce canonical gold and enumerate safe alternatives', async () => {
  const text = await readFile(new URL('./assets/history-train-specs.jsonl', import.meta.url), 'utf8');
  const specs = text.trimEnd().split('\n').map(JSON.parse);
  const groups = new Map();
  for (const spec of specs) if (!groups.has(spec.group)) groups.set(spec.group, spec);
  for (const spec of groups.values()) {
    const row = await materializeScenario(registry, api, spec);
    row.model_input = await renderPrompt(registry, api, row.before, row.question);
    const item = await diagnostic.prepare({ source: 'history', row });
    const verdict = await diagnostic.score(item, complete(row.calls));
    assert.equal(verdict.semantic_success, true, spec.case_id);
  }
});
