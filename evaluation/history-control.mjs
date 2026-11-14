import assert from 'node:assert/strict';
import { continuation, renderPolicy } from './history-train.mjs';

const refKeys = new Set(['target', '_replace', '_resolve', '_supersede']);
const stable = value => JSON.stringify(value, (_, item) => item && typeof item === 'object' && !Array.isArray(item)
  ? Object.fromEntries(Object.keys(item).sort().map(key => [key, item[key]])) : item);

function references(calls) {
  return calls.flatMap(call => Object.entries(call.arguments).filter(([key, value]) =>
    refKeys.has(key) && typeof value === 'string').map(([, value]) => value));
}
function remapCalls(calls, mapping) {
  return calls.map(call => ({ ...call, arguments: Object.fromEntries(Object.entries(call.arguments).map(([key, value]) =>
    [key, refKeys.has(key) && typeof value === 'string' ? mapping.get(value) ?? value : value])) }));
}

export async function matchedControl(registry, api, spec, treatment) {
  const exchanges = treatment.before.exchanges;
  assert.equal(exchanges.length, spec.prefix.length);
  const owner = new Map(exchanges.map((exchange, index) => [exchange.id, index]));
  for (const [index, exchange] of exchanges.entries()) {
    const turn = treatment.before.semantic.turns.find(turn => turn.id === exchange.semantic_turn_id);
    for (const answer of turn?.answers ?? []) owner.set(answer.id, index);
    for (const item of turn?.response_parts ?? []) if (item.kind === 'clarification') owner.set(item.id, index);
    for (const effect of turn?.effects ?? []) if (effect.kind === 'pending_created') owner.set(effect.target, index);
  }
  for (const pending of treatment.before.semantic.pending) {
    const index = exchanges.findIndex(exchange => pending.id.startsWith(`${exchange.semantic_turn_id}.`));
    if (index >= 0) owner.set(pending.id, index);
  }
  const dependencies = exchanges.map((exchange, index) => {
    const deps = new Set(references(exchange.model_calls ?? []).map(id => owner.get(id)).filter(value => value !== undefined && value < index));
    const policy = spec.prefix[index].address_policy;
    if (policy) deps.add(policy.exchange_index - 1);
    return deps;
  });
  const pendingIndices = exchanges.map((exchange, index) => (exchange.model_calls ?? []).some(call =>
    call.name === 'cancel_pending' || Object.keys(call.arguments).some(key => ['_clarify', '_resolve', '_supersede'].includes(key))) ? index : -1)
    .filter(index => index >= 0);
  for (let at = 1; at < pendingIndices.length; at++) dependencies[pendingIndices[at]].add(pendingIndices[at - 1]);
  const wanted = spec.continuation.exchange_index ? spec.continuation.exchange_index - 1 : null;
  const originalOrder = exchanges.map((_, index) => index);
  const orders = [];
  const enumerate = prefix => {
    if (prefix.length === exchanges.length) { orders.push(prefix); return; }
    for (const index of originalOrder) if (!prefix.includes(index) && [...dependencies[index]].every(dep => prefix.includes(dep))) {
      enumerate([...prefix, index]);
    }
  };
  if (wanted === null) orders.push(originalOrder); else enumerate([]);
  const inversions = order => order.reduce((sum, value, index) => sum + order.slice(index + 1).filter(next => next < value).length, 0);
  orders.sort((a, b) => (wanted === null ? 0 : a.indexOf(wanted) - b.indexOf(wanted)) ||
    inversions(a) - inversions(b) || a.join(',').localeCompare(b.join(',')));
  const rejected = [];
  for (const order of orders) {
    try {
      const result = await reconstruct(registry, api, spec, treatment, order);
      result.match.rejected_reorders = rejected.length;
      result.match.reconstruction_fallback = rejected.length ? 'earliest_content_preserving_legal_order' : null;
      result.match.rejection_examples = rejected.slice(0, 3);
      return result;
    } catch (error) {
      if (!(error instanceof assert.AssertionError)) throw error;
      rejected.push({ exchange_order: order.map(index => exchanges[index].id), reason: error.message.split('\n')[0] });
    }
  }
  throw new Error(`${spec.case_id}: no content-preserving reconstruction, including original order: ${JSON.stringify(rejected)}`);
}

async function reconstruct(registry, api, spec, treatment, order) {
  const exchanges = treatment.before.exchanges;
  const positions = new Map(order.map((oldIndex, newIndex) => [oldIndex + 1, newIndex + 1]));
  const mapping = new Map();
  let before = await api.createSession(registry, { dialogueVersion: 2, sessionId: spec.session_id });
  for (const oldIndex of order) {
    const old = exchanges[oldIndex];
    const source = spec.prefix[oldIndex];
    const question = source.address_policy ? renderPolicy(source.address_policy,
      positions.get(source.address_policy.exchange_index)).question : old.question;
    if (old.outcome === 'provider_error') before = await api.recordProviderFailure(registry, before, question, old.error_code);
    else if (old.error_code === 'incomplete_model_output') before = await api.recordIncompleteModelOutput(registry, before,
      question, old.raw_output, old.generation.token_ids, old.generation.stop_reason);
    else {
      const raw = old.outcome === 'committed' ? api.encodeCalls(remapCalls(old.model_calls, mapping)) : old.raw_output;
      before = await api.applyModelOutput(registry, before, question, raw);
    }
    const current = before.exchanges.at(-1);
    assert.equal(current.outcome, old.outcome, `${spec.case_id}: reordered outcome`);
    assert.equal(current.error_code, old.error_code, `${spec.case_id}: reordered failure type`);
    assert.equal(current.assistant, old.assistant, `${spec.case_id}: reordered displayed content`);
    mapping.set(old.id, current.id);
    if (old.semantic_turn_id) {
      const oldTurn = treatment.before.semantic.turns.find(turn => turn.id === old.semantic_turn_id);
      const newTurn = before.semantic.turns.find(turn => turn.id === current.semantic_turn_id);
      mapping.set(old.semantic_turn_id, current.semantic_turn_id);
      assert.equal(oldTurn.answers.length, newTurn.answers.length);
      for (let part = 0; part < oldTurn.answers.length; part++) {
        mapping.set(oldTurn.answers[part].id, newTurn.answers[part].id);
        assert.equal(stable(oldTurn.answers[part].call), stable(newTurn.answers[part].call), 'Scientific arguments changed during reconstruction');
      }
      for (const pending of before.semantic.pending.filter(pending => pending.id.startsWith(`${current.semantic_turn_id}.`))) {
        mapping.set(`${old.semantic_turn_id}${pending.id.slice(current.semantic_turn_id.length)}`, pending.id);
      }
    }
  }
  const policy = spec.continuation.exchange_index ? renderPolicy(spec.continuation,
    positions.get(spec.continuation.exchange_index)) : spec.continuation;
  const gold = continuation(before, policy);
  const mappedPending = treatment.before.semantic.pending.map(item => ({ ...item, id: mapping.get(item.id),
    ...(item.revision_target ? { revision_target: mapping.get(item.revision_target) } : {}) }));
  assert.equal(stable(mappedPending), stable(before.semantic.pending), 'Pending lifecycle changed during reconstruction');
  assert.equal(stable(gold.calls), stable(remapCalls(treatment.calls, mapping)), `${spec.case_id}: matched continuation differs`);
  const after = await api.applyModelOutput(registry, before, gold.question, api.encodeCalls(gold.calls));
  assert.equal(after.exchanges.at(-1).outcome, 'committed');
  assert.equal(after.exchanges.at(-1).assistant, treatment.after.exchanges.at(-1).assistant,
    `${spec.case_id}: matched continuation text differs`);
  return { case_id: `${treatment.case_id}-control`, paired_case_id: treatment.case_id,
    family_id: treatment.family_id, group: treatment.group, split: 'train', question: gold.question,
    before, actual_before: api.sessionHistory(before), calls: gold.calls, target: api.encodeCalls(gold.calls), after,
    history_origin: 'reconstructed_earliest_legal_order_from_paired_history', branch: gold.branch,
    reference_targets: gold.reference_targets,
    match: { ...treatment.match, identity_order_pair: order.every((oldIndex, newIndex) => oldIndex === newIndex),
      exchange_order: order.map(index => exchanges[index].id), reference_mapping: Object.fromEntries(mapping),
      exact_scientific_arguments_preserved: true, displayed_content_preserved: true,
      current_action_arity_preserved: true, pending_lifecycle_order_preserved: true },
    provenance: { ...treatment.provenance, actual_prefix_generated: false, reconstructed_prefix: true,
      parent_history_origin: treatment.history_origin, authoring: 'Prospective matched-control order reconstruction',
      estimator: 'position/order and corresponding reference wording; not pure provenance or literal old-data replay' } };
}
