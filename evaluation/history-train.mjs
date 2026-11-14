import assert from 'node:assert/strict';

const ordinal = n => ['first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth'][n - 1] ?? `${n}th`;
const call = (name, args) => ({ name, arguments: args });
const failed = exchange => ['error', 'provider_error'].includes(exchange?.outcome);
const stable = value => JSON.stringify(value, (_, item) => item && typeof item === 'object' && !Array.isArray(item)
  ? Object.fromEntries(Object.keys(item).sort().map(key => [key, item[key]])) : item);

function answerAt(session, exchangeIndex, part = 1) {
  const exchange = session.exchanges[exchangeIndex - 1];
  const turn = session.semantic.turns.find(item => item.id === exchange?.semantic_turn_id);
  return turn?.answers[part - 1];
}

function pendingAt(session, exchangeIndex, part = 1) {
  const exchange = session.exchanges[exchangeIndex - 1];
  return session.semantic.pending.find(item => item.id === `${exchange?.semantic_turn_id}.p${part}`);
}

export function continuation(session, policy) {
  policy = renderPolicy(policy);
  const exchange = session.exchanges[policy.exchange_index - 1];
  if (policy.kind === 'quote_exchange') {
    assert.ok(exchange, 'Predeclared exchange is absent');
    return { question: policy.question, calls: [call('repeat_exchange', { target: exchange.id })],
      branch: 'actual_exchange', reference_targets: [exchange.id] };
  }
  if (policy.kind === 'answer_again') {
    assert.ok(exchange, 'Predeclared exchange is absent');
    return { question: policy.question,
      calls: [...(failed(exchange) ? [call('retry_exchange', { target: exchange.id })] : []), ...policy.calls],
      branch: failed(exchange) ? 'retry_actual_failure' : 'fresh_answer_to_actual_request',
      reference_targets: failed(exchange) ? [exchange.id] : [] };
  }
  if (policy.kind === 'quote_answer') {
    const answer = answerAt(session, policy.exchange_index, policy.part);
    return answer
      ? { question: policy.question, calls: [call('repeat_answer', { target: answer.id })],
        branch: 'actual_answer', reference_targets: [answer.id] }
      : { question: policy.question, calls: [call('clarify_reference', { kind: 'answer' })],
        branch: 'requested_answer_missing', reference_targets: [] };
  }
  if (policy.kind === 'revise_answer' || policy.kind === 'replace_answer') {
    const answer = answerAt(session, policy.exchange_index, policy.part);
    if (!answer) return { question: policy.question, calls: [call('clarify_reference', { kind: 'answer' })],
      branch: 'requested_answer_missing', reference_targets: [] };
    if (stable(answer.call) === stable(policy.replacement)) return { question: policy.already_correct_question,
      calls: [call('repeat_answer', { target: answer.id })], branch: 'requested_content_already_present',
      reference_targets: [answer.id] };
    const same = stable(answer.call) === stable(policy.expected_original);
    if (policy.kind === 'revise_answer' && same) return { question: policy.question,
      calls: [call(`change_${answer.call.name}`, { target: answer.id, changes: policy.changes })],
      branch: 'exact_original_delta', reference_targets: [answer.id] };
    return { question: policy.replacement_question ?? policy.question,
      calls: [call(policy.replacement.name, { ...policy.replacement.arguments, _replace: answer.id })],
      branch: policy.kind === 'replace_answer' ? 'explicit_replacement' : 'wrong_original_explicit_replacement',
      reference_targets: [answer.id] };
  }
  if (policy.kind === 'resolve_pending' || policy.kind === 'cancel_pending') {
    const pending = pendingAt(session, policy.exchange_index, policy.part);
    const matching = pending && pending.function === policy.function &&
      stable(pending.known) === stable(policy.known) &&
      stable(pending.missing) === stable(policy.missing);
    if (!matching) return { question: policy.missing_question,
      calls: [call('clarify_reference', { kind: 'pending' })],
      branch: 'requested_pending_missing_or_different', reference_targets: [] };
    if (policy.kind === 'cancel_pending') return { question: policy.question,
      calls: [call('cancel_pending', { target: pending.id })], branch: 'cancel_actual_pending',
      reference_targets: [pending.id] };
    return { question: policy.question,
      calls: [call(policy.function, { ...policy.values, _resolve: pending.id })],
      branch: 'resolve_actual_pending', reference_targets: [pending.id] };
  }
  if (policy.kind === 'clarify_reference') return { question: policy.question,
    calls: [call('clarify_reference', { kind: policy.reference_kind })],
    branch: 'preauthored_ambiguous_reference', reference_targets: [] };
  if (policy.kind === 'fresh') return { question: policy.question, calls: policy.calls,
    branch: 'fresh', reference_targets: [] };
  throw new Error(`Unknown preauthored continuation kind: ${policy.kind}`);
}

export function renderPolicy(policy, exchangeIndex = policy.exchange_index) {
  const rendered = { ...policy, exchange_index: exchangeIndex };
  for (const key of ['question', 'replacement_question', 'missing_question', 'already_correct_question']) {
    const template = policy.address_templates?.[key];
    if (template) rendered[key] = template.replaceAll('{exchange_ordinal}', ordinal(exchangeIndex))
      .replaceAll('{exchange_number}', String(exchangeIndex));
  }
  return rendered;
}

export async function materializeScenario(registry, api, spec, rawOutputs = null) {
  let before = await api.createSession(registry, { dialogueVersion: 2, sessionId: spec.session_id });
  if (rawOutputs) assert.equal(rawOutputs.length, spec.prefix.length, 'Partial observations cannot be frozen');
  for (const [index, step] of spec.prefix.entries()) {
    if (rawOutputs) {
      const observed = rawOutputs[index];
      assert.equal(observed.question, step.question, 'Observed question differs from frozen prefix');
      before = await applyObservation(registry, api, before, step.question, observed);
    } else if (step.kind === 'provider_error') {
      before = await api.recordProviderFailure(registry, before, step.question, step.error_code);
    } else if (step.kind === 'malformed') {
      before = await api.applyModelOutput(registry, before, step.question, step.raw_output);
    } else {
      const calls = step.policy ? continuation(before, step.policy).calls : step.calls;
      before = await api.applyModelOutput(registry, before, step.question, api.encodeCalls(calls));
      assert.equal(before.exchanges.at(-1).outcome, 'committed', `${spec.case_id}: prefix ${index + 1}`);
    }
  }
  const target = continuation(before, spec.continuation);
  const after = await api.applyModelOutput(registry, before, target.question, api.encodeCalls(target.calls));
  assert.equal(after.exchanges.at(-1).outcome, 'committed', `${spec.case_id}: ${after.exchanges.at(-1).error_code}`);
  assert.deepEqual(after.exchanges.slice(0, -1), before.exchanges, 'Gold continuation rewrote actual history');
  return { case_id: spec.case_id, family_id: spec.family_id, group: spec.group, split: 'train',
    question: target.question, before, actual_before: api.sessionHistory(before), calls: target.calls,
    target: api.encodeCalls(target.calls), after,
    history_origin: rawOutputs ? 'observed_frozen_checkpoint_train_prefix' : 'reconstructed_actual_history_v3',
    reference_targets: target.reference_targets, branch: target.branch,
    match: spec.match, provenance: { ...spec.provenance, actual_prefix_generated: Boolean(rawOutputs) } };
}

export async function applyObservation(registry, api, before, question, observed) {
  if (observed.kind === 'provider_error') return api.recordProviderFailure(registry, before, question, observed.error_code);
  if (observed.kind === 'incomplete') return api.recordIncompleteModelOutput(registry, before, question,
    observed.raw_output, observed.raw_ids, 'token_limit');
  return api.applyModelOutput(registry, before, question, observed.raw_output);
}

export async function renderPrompt(registry, api, before, question) {
  let prompt;
  const captured = new Error('Prompt captured without tokenization');
  try {
    await api.buildModelInput(registry, before, question, { contextWindow: 32768, completionTokens: 512,
      tokenizer: { sha256: api.pinnedTokenizerSha256, encode: async (text, options) => {
        assert.deepEqual(options, { add_special_tokens: false, truncation: false });
        prompt = text;
        throw captured;
      } } });
  } catch (error) { if (error !== captured) throw error; }
  assert.ok(prompt);
  return prompt;
}

export { ordinal };
