import { canonical, clone, freeze, ContractError } from './schema.mjs';
import { createSession, sessionHistory, SessionController } from './session.mjs';
import { gradeTurn, summarizeEvaluation, textHash } from './evaluation.mjs';
import { IncompleteModelOutput } from './model-result.mjs';

const fail = message => { throw new ContractError('invalid_evaluation_plan', message); };
const same = (left, right) => canonical(left) === canonical(right);

/** What the model/user actually saw, including errors, pending and focus.
 * Private native serialization and the host-assigned UUID are not language. */
export async function visibleHistoryHash(session) {
  const { session_id, ...history } = sessionHistory(session);
  return textHash(canonical(history));
}

async function approved(registryId, gold) {
  if (gold.review?.status !== 'approved' || !gold.author || !gold.review.reviewer ||
      gold.author === gold.review.reviewer || gold.review.registry_sha256 !== registryId ||
      gold.review.question_sha256 !== await textHash(gold.question) ||
      !Array.isArray(gold.allowed_model_call_batches) || !gold.allowed_model_call_batches.length) {
    fail('Every planned question requires current independent approved interpretation.');
  }
}

const identity = row => ({ case_id: row.case_id, split: row.split, cohort: row.cohort,
  dialogue_id: row.dialogue_id ?? null, turn_index: row.turn_index, question: row.question, author: row.author });

/** Sequential development evaluation. The first provider argument is the
 * normal production request; the second is a technical case identity for
 * receipts and must never be added to the model prompt or used for routing.
 * Gold/adjudication never supplies the provider's next session.
 * This core neither launches models nor authenticates external job files;
 * the frozen CLI/native-worker adapter owns those boundaries and persistence. */
export async function evaluateDevelopment(registry, planned, {
  split, provider, adjudicate, onRecord = async () => {}, stopOnProviderFailure = true, sessionIdFor,
} = {}) {
  if (!['train', 'dev'].includes(split) || typeof provider !== 'function' ||
      !Array.isArray(planned) || !planned.length || typeof onRecord !== 'function' ||
      (adjudicate !== undefined && typeof adjudicate !== 'function') || typeof stopOnProviderFailure !== 'boolean') {
    fail('Explicit TRAIN-fit or DEV plan and provider required; acceptance is separate.');
  }
  // Copy/freeze before any model output. A callback cannot change denominators
  // or future questions after seeing a response.
  planned = freeze(clone(planned));
  const registryId = await registry.fingerprint(), ids = new Set(), episodeCounts = new Map();
  for (const row of planned) {
    if (row.split !== split || !row.case_id || ids.has(row.case_id) ||
        !['supported', 'outside_scope'].includes(row.cohort) || typeof row.question !== 'string' || !row.question.trim()) {
      fail('Unknown split/cohort, missing question or duplicate planned case.');
    }
    ids.add(row.case_id); await approved(registryId, row);
    if (row.dialogue_id) {
      const previous = episodeCounts.get(row.dialogue_id) ?? 0;
      if (row.turn_index !== previous + 1 || !['context_invariant', 'exact_visible_prefix_or_independent_adjudication'].includes(row.history_policy)) {
        fail('Complete ordered episodes and explicitly reviewed history policy required.');
      }
      if (!/^[a-f0-9]{64}$/.test(row.review.before_visible_history_sha256 ?? '')) {
        fail('Dialogue gold must bind its complete independently reviewed visible prefix.');
      }
      episodeCounts.set(row.dialogue_id, row.turn_index);
    } else if (row.turn_index !== 0) fail('Standalone questions start in fresh sessions.');
  }
  if ([...episodeCounts.values()].some(count => count !== 5)) fail('The declared development episodes have exactly five turns.');
  const sessions = new Map(), blocked = new Map(), results = [];
  let aborted = null, generated = 0;
  for (const row of planned) {
    const before = sessions.get(row.dialogue_id) ?? await createSession(registry, sessionIdFor?.(row.dialogue_id ?? row.case_id));
    const historyHash = await visibleHistoryHash(before);
    let gold = row, interpretation = 'reviewed_original', blockedReason = blocked.get(row.dialogue_id), adjudication = null;
    if (!blockedReason && row.dialogue_id && row.history_policy !== 'context_invariant' &&
        historyHash !== row.review.before_visible_history_sha256) {
      gold = null; interpretation = 'actual_history_review_required';
      if (adjudicate) {
        // No nominal calls and no current candidate output are exposed here.
        // The interpretation is made BEFORE the provider sees this question.
        adjudication = await adjudicate(freeze({ case_id: row.case_id, question: row.question,
          cohort: row.cohort, dialogue_id: row.dialogue_id, turn_index: row.turn_index,
          actual_before: before, before_visible_history_sha256: historyHash, registry_sha256: registryId }));
        if (!adjudication || !['grade', 'blocked', 'review'].includes(adjudication.decision)) fail('Unknown actual-history adjudication.');
        if (adjudication.decision === 'grade') {
          const actualGold = freeze(clone(adjudication.gold));
          await approved(registryId, actualGold);
          if (!same(identity(row), identity(actualGold)) || actualGold.history_policy !== 'reviewed_actual_candidate' ||
              actualGold.review.before_visible_history_sha256 !== historyHash) {
            fail('Adjudication must preserve the exact planned question and bind its actual visible history.');
          }
          gold = actualGold; interpretation = 'independently_reviewed_actual_history';
        } else if (adjudication.decision === 'blocked') {
          if (!adjudication.reviewer || adjudication.reviewer === row.author || typeof adjudication.reason !== 'string' || !adjudication.reason ||
              adjudication.question_sha256 !== row.review.question_sha256 ||
              adjudication.registry_sha256 !== registryId || adjudication.before_visible_history_sha256 !== historyHash) {
            fail('Blocking a continuation requires an independent reason on its exact actual prefix.');
          }
          blockedReason = adjudication.reason; blocked.set(row.dialogue_id, blockedReason);
        }
      }
    }
    if (blockedReason) {
      const result = { case_id: row.case_id, success: false, reasons: ['blocked_continuation'],
        critical_status: 'none_observed', observation_origin: 'planned_but_not_generated' };
      results.push(result);
      await onRecord(freeze({ case_id: row.case_id, status: 'blocked', reason: blockedReason,
        before, after: before, question_shown: false, before_visible_history_sha256: historyHash, result }));
      continue;
    }
    let providerFailure = null;
    const controller = new SessionController(registry, before);
    const after = await controller.send(row.question, async request => {
      try {
        const raw = await provider(request, freeze({ case_id: row.case_id, split,
          dialogue_id: row.dialogue_id ?? null, turn_index: row.turn_index }));
        if (typeof raw !== 'string') throw new ContractError('invalid_provider_result', 'Provider must return its exact raw decoded string.');
        return raw;
      }
      catch (error) {
        if (!(error instanceof IncompleteModelOutput)) providerFailure = typeof error.code === 'string' ? error.code : 'inference_failed';
        throw error;
      }
    });
    if (after.exchanges.at(-1).outcome === 'provider_error') providerFailure ??= after.exchanges.at(-1).error_code;
    generated++;
    if (row.dialogue_id) sessions.set(row.dialogue_id, after);
    let result;
    if (gold) {
      try { result = await gradeTurn(registry, before, after, gold); }
      catch (error) {
        if (!(error instanceof ContractError) || error.code !== 'unapproved_gold') throw error;
        // The actual candidate trace survives. An invalid control on this
        // prefix is not a model success or permission to inject a gold state.
        result = { case_id: row.case_id, success: false, reasons: ['actual_history_gold_requires_review'],
          critical_status: 'review_required', observation_origin: 'session_log' };
        interpretation = 'actual_history_review_required';
      }
    } else result = { case_id: row.case_id, success: false, reasons: ['actual_history_requires_independent_review'],
      critical_status: 'review_required', observation_origin: 'session_log' };
    results.push(result);
    await onRecord(freeze({ case_id: row.case_id, status: 'observed', interpretation, before, after,
      question_shown: true, before_visible_history_sha256: historyHash,
      adjudicated_review: gold !== row && gold ? gold.review : null, provider_failure: providerFailure, result }));
    if (providerFailure && stopOnProviderFailure) {
      aborted = { case_id: row.case_id, reason: providerFailure }; break;
    }
  }
  return freeze({ schema_version: 1, split, planned_cases: planned.length, generated_cases: generated,
    observed_records: results.length, aborted, stop_on_provider_failure: stopOnProviderFailure,
    results, summary: summarizeEvaluation(planned, results), model_quality_measured_by_this_core_alone: false,
    limitation: 'Provider identity/resource/decoding evidence and independent critical adjudication remain external requirements.' });
}
