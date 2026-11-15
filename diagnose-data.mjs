import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { createInterface } from 'node:readline';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { canonical } from './evaluation/schema.mjs';
import { createRegistry } from './evaluation/registry.mjs';
import { createSession, applyModelOutput, recordProviderFailure, recordIncompleteModelOutput,
  sessionHistory } from './evaluation/session.mjs';
import { encodeCalls, decodeCalls } from './evaluation/functiongemma.mjs';
import { buildModelInput, pinnedTokenizerSha256 } from './evaluation/model-input.mjs';
import { renderPrompt } from './evaluation/history-train.mjs';
import { createActualHistoryHarness } from './evaluation/actual-history.mjs';

const api = { createSession, applyModelOutput, recordProviderFailure, recordIncompleteModelOutput,
  sessionHistory, encodeCalls, canonical, buildModelInput, pinnedTokenizerSha256 };
const harness = createActualHistoryHarness(api);
const equal = (a, b) => canonical(a) === canonical(b);
const hash = value => createHash('sha256').update(typeof value === 'string' ? value : canonical(value)).digest('hex');
const call = (name, args) => ({ name, arguments: args });
const refs = calls => calls.flatMap(c => ['target', '_replace', '_resolve', '_supersede']
  .filter(key => typeof c.arguments?.[key] === 'string').map(key => c.arguments[key]));
export const compactObservationSchema = 'diagnostic-observation-v1-compact';

export function compactObservation(observation) {
  assert.ok(Object.hasOwn(observation, 'before') && Object.hasOwn(observation, 'after'));
  const { before, after, ...record } = observation;
  const exchange = after?.exchanges.at(-1);
  const delivered = exchange ? Object.fromEntries(['id', 'question', 'assistant', 'outcome', 'error_code',
    'semantic_turn_id'].filter(key => Object.hasOwn(exchange, key)).map(key => [key, exchange[key]])) : null;
  return { ...record, observation_schema: compactObservationSchema,
    before_sha256: before ? hash(before) : null,
    after_sha256: after ? hash(after) : null,
    after_semantic_sha256: after ? hash(after.semantic) : null,
    delivered_sha256: delivered ? hash(delivered) : null, delivered };
}

function promptData(prompt) {
  const start = '<start_of_turn>user\n', end = '<end_of_turn>\n<start_of_turn>model\n';
  assert.equal(typeof prompt, 'string');
  const parts = prompt.split(start);
  assert.equal(parts.length, 2, 'Expected one frozen user payload');
  assert.ok(parts[1].endsWith(end), 'Frozen prompt has an unexpected generation boundary');
  const data = JSON.parse(parts[1].slice(0, -end.length));
  assert.ok(['gemprf-functiongemma-input-v1', 'gemprf-functiongemma-input-v2'].includes(data.format));
  assert.ok(Array.isArray(data.history?.exchanges));
  return data;
}

async function replayBefore(registry, stored) {
  assert.ok(stored && stored.version === 1 && Array.isArray(stored.exchanges));
  let before = await createSession(registry, { sessionId: stored.session_id,
    dialogueVersion: stored.dialogue_version ?? 1 });
  for (const ex of stored.exchanges) {
    if (ex.outcome === 'provider_error') before = await recordProviderFailure(registry, before, ex.question, ex.error_code);
    else if (ex.error_code === 'incomplete_model_output') {
      before = await recordIncompleteModelOutput(registry, before, ex.question, ex.raw_output,
        ex.generation.token_ids, ex.generation.stop_reason);
    } else before = await applyModelOutput(registry, before, ex.question, ex.raw_output);
  }
  assert.ok(equal(before, stored), 'Frozen before-state does not reproduce from its original observations');
  return before;
}

function answer(before, id) {
  return before.semantic.turns.flatMap(turn => turn.answers).find(item => item.id === id);
}

function answerDisplay(before, exchange) {
  const turn = before.semantic.turns.find(item => item.id === exchange?.semantic_turn_id);
  if (exchange?.outcome !== 'committed' || !turn?.answers.length ||
      turn.response_parts.length !== turn.answers.length ||
      !turn.response_parts.every((part, index) => part.kind === 'answer' && part.id === turn.answers[index].id) ||
      turn.response_parts.map(part => part.text).join('\n\n') !== exchange.assistant) return null;
  return turn;
}

function alternatives(before, original) {
  const options = [structuredClone(original)], args = original.arguments;
  if (original.name === 'repeat_exchange') {
    const exchange = before.exchanges.find(ex => ex.id === args.target);
    const turn = answerDisplay(before, exchange);
    if (turn) return [options, turn.answers.map(item => call('repeat_answer', { target: item.id }))];
  }
  if (original.name === 'repeat_answer') {
    const exchange = before.exchanges.find(ex => answerDisplay(before, ex)?.answers.some(a => a.id === args.target));
    const turn = answerDisplay(before, exchange);
    if (turn?.answers.length === 1) options.push(call('repeat_exchange', { target: exchange.id }));
  }
  if (typeof args._replace === 'string') {
    const previous = answer(before, args._replace);
    const values = { ...args }; delete values._replace;
    if (previous?.call.name === original.name) {
      const changes = Object.fromEntries(Object.entries(values).filter(([key, value]) =>
        !Object.hasOwn(previous.call.arguments, key) || !equal(value, previous.call.arguments[key])));
      const remove = Object.keys(previous.call.arguments).filter(key => !Object.hasOwn(values, key));
      if (Object.keys(changes).length || remove.length) options.push(call(`change_${original.name}`,
        { target: previous.id, changes, ...(remove.length ? { remove } : {}) }));
    }
  }
  if (original.name.startsWith('change_')) {
    const previous = answer(before, args.target);
    if (previous && original.name === `change_${previous.call.name}`) {
      const values = { ...previous.call.arguments, ...args.changes };
      for (const key of args.remove ?? []) delete values[key];
      options.push(call(previous.call.name, { ...values, _replace: previous.id }));
    }
  }
  return options.map(item => [item]);
}

function allowedBatches(before, calls) {
  let batches = [[]];
  for (const item of calls) {
    const choices = alternatives(before, item);
    assert.ok(batches.length * choices.length <= 256, 'Equivalent-program expansion exceeds the declared bound');
    batches = batches.flatMap(prefix => choices.map(choice => [...prefix, ...choice]));
  }
  return [...new Map(batches.map(batch => [canonical(batch), batch])).values()];
}

export async function materializeGeneration(registry, before, question, generation) {
  if (generation.error) {
    assert.ok(['context_overflow', 'inference_failed'].includes(generation.error));
    return recordProviderFailure(registry, before, question, generation.error);
  }
  assert.equal(typeof generation.raw_text, 'string');
  assert.equal(typeof generation.complete, 'boolean', 'Provider must report native-batch completeness');
  if (generation.complete) return applyModelOutput(registry, before, question, generation.raw_text);
  const failed = structuredClone(await applyModelOutput(registry, before, question, ''));
  Object.assign(failed.exchanges.at(-1), { error_code: 'incomplete_model_output', raw_output: generation.raw_text,
    generation: { complete_native_batch: false, provider_kind: generation.provider_kind ?? 'unspecified',
      token_ids: generation.raw_ids ?? null, stop_reason: generation.finish_reason ?? 'provider_stopped_before_native_batch' } });
  return failed;
}

export function createTrainDiagnostics(registry) {
  async function prepare({ source, row, prompt = row.model_input, before: suppliedBefore = null }) {
    assert.ok(['common', 'history'].includes(source));
    assert.equal(typeof row.case_id, 'string');
    assert.equal(typeof row.question, 'string');
    assert.ok(Array.isArray(row.calls) && row.calls.length);
    const data = promptData(prompt);
    assert.equal(data.question, row.question, 'TRAIN question differs from frozen prompt');
    let before = null, reason = null;
    if (source === 'history' || suppliedBefore) before = await replayBefore(registry, suppliedBefore ?? row.before);
    else if (data.history.exchanges.length === 0) before = await createSession(registry,
      { sessionId: data.history.session_id, dialogueVersion: data.format.endsWith('v2') ? 2 : 1 });
    else reason = 'historical_common_row_requires_authenticated_before_state';
    let allowed = null, expected = null;
    if (before) {
      assert.ok(equal(sessionHistory(before), data.history), 'Frozen visible history does not match restored state');
      assert.equal(await renderPrompt(registry, api, before, row.question), prompt,
        'Current prompt contract differs from the exact frozen TRAIN input');
      expected = await applyModelOutput(registry, before, row.question, encodeCalls(row.calls));
      assert.equal(expected.exchanges.at(-1).outcome, 'committed', 'TRAIN gold program failed execution');
      if (row.after) assert.ok(equal(expected, row.after), 'Frozen TRAIN target state is not reproducible');
      allowed = allowedBatches(before, row.calls);
      for (const batch of allowed) {
        const alternative = await applyModelOutput(registry, before, row.question, encodeCalls(batch));
        assert.equal(alternative.exchanges.at(-1).outcome, 'committed', 'Equivalent program failed execution');
        assert.equal(alternative.exchanges.at(-1).assistant, expected.exchanges.at(-1).assistant,
          'Equivalent program changed the complete delivered answer');
        assert.ok(equal(alternative.semantic.pending, expected.semantic.pending));
        assert.ok(equal(alternative.semantic.turns.slice(0, -1), expected.semantic.turns.slice(0, -1)));
      }
    }
    const item = { schema: 'train-fit-case-v1', case_id: row.case_id,
      family_id: row.family_id ?? row.case_id, group: row.group ?? row.history_origin ?? 'common',
      source, cohort: row.calls.every(c => c.name === 'decline_request') ? 'outside_scope' : 'supported',
      question: row.question, prompt, prompt_sha256: hash(prompt), before,
      gold_calls: row.calls, exact_target: encodeCalls(row.calls), allowed_batches: allowed,
      semantic_eligible: Boolean(before), semantic_exclusion_reason: reason,
      label_basis: source === 'history' ? 'frozen_reviewed_continuation_policy' : 'frozen_training_gold',
      label_independently_rechecked_here: false, provenance: row.provenance ?? null };
    return { ...item, case_sha256: hash(item) };
  }

  async function score(item, generation) {
    const { case_sha256, ...content } = item;
    assert.equal(item.schema, 'train-fit-case-v1');
    assert.equal(case_sha256, hash(content), 'Prepared diagnostic case changed');
    assert.ok(generation && typeof generation === 'object');
    if (generation.error) assert.ok(['context_overflow', 'inference_failed'].includes(generation.error));
    else {
      assert.equal(typeof generation.raw_text, 'string');
      assert.equal(typeof generation.complete, 'boolean');
    }
    const complete = generation.complete === true && !generation.error;
    let callExact = false, transportValid = false;
    if (complete) {
      try {
        const calls = decodeCalls(generation.raw_text);
        transportValid = true;
        callExact = equal(calls, item.gold_calls);
      } catch {}
    }
    const literalExact = complete && generation.raw_text === item.exact_target;
    let result = null, after = null;
    if (item.semantic_eligible) {
      const before = item.before;
      after = await materializeGeneration(registry, before, item.question, generation);
      const targets = refs(item.gold_calls);
      const resolved = { question: item.question, branch: 'frozen_train', allowed_batches: item.allowed_batches,
        reference_targets: targets, reference_target_alternatives: item.allowed_batches.map(refs),
        authored_reference_eligible: targets.length > 0 };
      result = await harness.grade(registry, { ...item, panel: 'train_fixed',
        policy: { kind: 'fresh' }, turn_index: null, reference_eligible: targets.length > 0 }, before, after, resolved);
    }
    return { case_id: item.case_id, case_sha256, prompt_sha256: item.prompt_sha256,
      source: item.source, cohort: item.cohort, family_id: item.family_id, group: item.group,
      semantic_eligible: item.semantic_eligible, semantic_exclusion_reason: item.semantic_exclusion_reason,
      native_literal_exact: literalExact, canonical_call_exact: callExact,
      native_exact_match: literalExact, canonical_call_match: callExact, transport_valid: transportValid,
      semantic_success: result?.success ?? null, semantic_result: result,
      success: result?.success ?? null, critical_status: result?.critical_status ?? null,
      error_kind: generation.error ?? after?.exchanges.at(-1).error_code ?? result?.failure_category ?? null,
      generation, before: item.before, after, label_basis: item.label_basis, label_independently_rechecked_here: false };
  }
  async function expand(item, observation) {
    assert.equal(observation.observation_schema, compactObservationSchema);
    const full = await score(item, observation.generation);
    assert.ok(equal(compactObservation(full), observation), 'Compact TRAIN receipt does not reproduce');
    return full;
  }
  return { prepare, score, expand };
}

async function main() {
  const registryPath = process.argv[2];
  assert.ok(registryPath, 'Expected registry JSON path');
  const bundle = JSON.parse(await readFile(registryPath, 'utf8'));
  const registry = createRegistry(bundle.knowledge, bundle.contracts);
  assert.equal(await registry.fingerprint(), bundle.fingerprint);
  const diagnostic = createTrainDiagnostics(registry);
  const reader = createInterface({ input: process.stdin, crlfDelay: Infinity });
  for await (const line of reader) {
    const request = JSON.parse(line);
    let output;
    if (request.action === 'prepare') output = { kind: 'prepared', case: await diagnostic.prepare(request) };
    else if (request.action === 'score') {
      const observation = await diagnostic.score(request.case, request.generation);
      output = { kind: 'score', observation: request.compact === true ? compactObservation(observation) : observation };
    } else if (request.action === 'expand') {
      output = { kind: 'expanded', observation: await diagnostic.expand(request.case, request.observation) };
    }
    else throw new Error(`Unknown diagnostic action: ${request.action}`);
    await new Promise((yes, no) => process.stdout.write(`${JSON.stringify(output)}\n`, error => error ? no(error) : yes()));
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) await main();
