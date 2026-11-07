import { canonical, ContractError } from './schema.mjs';
import { applyModelOutput } from './session.mjs';
import { encodeCalls } from './functiongemma.mjs';

export async function textHash(text) {
  const bytes = await globalThis.crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
  return [...new Uint8Array(bytes)].map(byte => byte.toString(16).padStart(2, '0')).join('');
}

const same = (a, b) => canonical(a) === canonical(b);
const invalidGold = message => { throw new ContractError('unapproved_gold', message); };

/** Compare a committed candidate trace with independently reviewed admissible
 * programs on its OWN prior history. Expected sessions are never returned to
 * the caller and cannot become the next inference input through this API. */
export async function gradeTurn(registry, before, after, gold, observation) {
  if (gold.review?.status !== 'approved' || !gold.author || !gold.review.reviewer ||
      gold.author === gold.review.reviewer) invalidGold('Independent approved interpretation is required.');
  if (gold.review.question_sha256 !== await textHash(gold.question) ||
      gold.review.registry_sha256 !== await registry.fingerprint()) invalidGold('Question or source registry changed after review.');
  if (!Array.isArray(gold.allowed_model_call_batches) || !gold.allowed_model_call_batches.length) {
    invalidGold('At least one independently admitted complete program is required.');
  }
  const origin = observation?.origin ?? 'session_log';
  const failure = (reasons, critical = 'review_required') => ({ case_id: gold.case_id,
    success: false, reasons, critical_status: critical, observation_origin: origin });
  if (before.registry_id !== after.registry_id || before.session_id !== after.session_id ||
      before.dialogue_version !== after.dialogue_version ||
      after.exchanges.length !== before.exchanges.length + 1 ||
      !same(after.exchanges.slice(0, -1), before.exchanges)) {
    return failure(['actual_history_rewritten_or_exchange_missing'], 'detected');
  }
  const actual = after.exchanges.at(-1);
  if (actual.question !== gold.question) return failure(['different_question'], 'detected');
  const displayed = observation?.visible_text ?? actual.assistant;
  let bestReasons;
  for (const calls of gold.allowed_model_call_batches) {
    const expected = await applyModelOutput(registry, before, gold.question, encodeCalls(calls));
    if (expected.exchanges.at(-1).outcome !== 'committed') {
      invalidGold('An admitted program cannot execute on the actual previous history; adjudicate or mark blocked.');
    }
    const required = expected.exchanges.at(-1);
    const reasons = [];
    if (actual.outcome !== required.outcome) reasons.push('missing_complete_outcome');
    if (displayed !== required.assistant) reasons.push('visible_answer_differs');
    if (actual.assistant !== required.assistant) reasons.push('stored_answer_differs');
    if (!same(after.semantic, expected.semantic)) reasons.push('complete_semantic_state_differs');
    if (!reasons.length) return { case_id: gold.case_id, success: true, reasons: [],
      critical_status: 'none_observed', observation_origin: origin };
    if (!bestReasons || reasons.length < bestReasons.length) bestReasons = reasons;
  }
  const unchangedOnError = actual.outcome !== 'committed' && same(after.semantic, before.semantic);
  return failure(bestReasons, unchangedOnError ? 'none_observed' : 'review_required');
}

/** Planned cases, including missing/error/review cases, define denominators. */
export function summarizeEvaluation(planned, results) {
  const ids = new Set(planned.map(row => row.case_id));
  if (ids.size !== planned.length) throw new Error('Duplicate planned case.');
  if (planned.some(row => !['supported', 'outside_scope'].includes(row.cohort))) throw new Error('Unknown planned cohort.');
  const observed = new Map();
  for (const row of results) {
    if (!ids.has(row.case_id) || observed.has(row.case_id)) throw new Error('Unknown or duplicate result.');
    if (!['none_observed', 'review_required', 'detected'].includes(row.critical_status)) throw new Error('Unclassified critical status.');
    observed.set(row.case_id, row);
  }
  const passedCase = id => observed.get(id)?.success === true && observed.get(id)?.critical_status === 'none_observed';
  const count = cohort => {
    const rows = planned.filter(row => row.cohort === cohort);
    const passed = rows.filter(row => passedCase(row.case_id)).length;
    return { planned: rows.length, passed, accuracy: rows.length ? passed / rows.length : null };
  };
  const dialogues = new Map();
  for (const row of planned.filter(row => row.dialogue_id)) {
    if (!dialogues.has(row.dialogue_id)) dialogues.set(row.dialogue_id, []);
    dialogues.get(row.dialogue_id).push(row.case_id);
  }
  const complete = [...dialogues.values()].filter(cases => cases.every(passedCase)).length;
  return { supported: count('supported'), outside_scope: count('outside_scope'),
    whole_dialogues: { planned: dialogues.size, passed: complete, accuracy: dialogues.size ? complete / dialogues.size : null },
    missing: planned.length - observed.size,
    critical_detected: results.filter(row => row.critical_status === 'detected').length,
    critical_review_required: results.filter(row => row.critical_status === 'review_required').length,
    release_ready: false };
}
