import { createHash } from 'node:crypto';

export const hash = value => createHash('sha256').update(typeof value === 'string' ? value : JSON.stringify(value)).digest('hex');
const clone = value => structuredClone(value);
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const call = (name, args = {}) => ({ name, arguments: args });
const refs = calls => calls.flatMap(c => ['target', '_replace', '_resolve', '_supersede'].filter(k => typeof c.arguments?.[k] === 'string').map(k => c.arguments[k]));

export function createActualHistoryHarness(api) {
  const { applyModelOutput, createSession, encodeCalls, canonical, sessionHistory } = api;
  const equal = (a, b) => canonical(a) === canonical(b);
  const visibleHash = before => {
    const { session_id, ...history } = sessionHistory(before);
    return hash(canonical(history));
  };
  const exchange = (before, n) => before.exchanges[n - 1];
  const answer = (before, n, part = 1) => {
    const ex = exchange(before, n);
    return before.semantic.turns.find(t => t.id === ex?.semantic_turn_id)?.answers[part - 1];
  };
  const pending = (before, policy) => {
    const ex = exchange(before, policy.target_exchange);
    const ids = before.semantic.turns.find(t => t.id === ex?.semantic_turn_id)?.response_parts?.filter(p => p.kind === 'clarification').map(p => p.id) ?? [];
    return before.semantic.pending.find(p => ids.includes(p.id) && (!policy.function || p.function === policy.function));
  };
  function resolve(row, before) {
    const p = row.policy;
    const ex = exchange(before, p.target_exchange);
    const object = answer(before, p.target_exchange, p.part ?? 1);
    let branch = 'default', batches, referenceTargets = [];
    if (p.kind === 'fresh') batches = p.allowed_batches ?? [p.calls];
    else if (p.kind === 'quote') {
      branch = ex ? 'present' : 'absent';
      batches = [[ex ? call('repeat_exchange', { target: ex.id }) : call('clarify_reference', { kind: 'exchange' })]];
      if (ex) referenceTargets = [ex.id];
    } else if (p.kind === 'repeat_answer') {
      branch = object ? 'present' : 'absent';
      batches = [[object ? call('repeat_answer', { target: object.id }) : call('clarify_reference', { kind: 'answer' })]];
      if (object) referenceTargets = [object.id];
    } else if (p.kind === 'retry') {
      branch = ex && ['error', 'provider_error'].includes(ex.outcome) ? 'failed' : 'answered';
      batches = [[...(branch === 'failed' ? [call('retry_exchange', { target: ex.id })] : []), ...clone(p.calls)]];
      if (branch === 'failed') referenceTargets = [ex.id];
    } else if (p.kind === 'replace') {
      branch = object ? 'present' : 'absent';
      const fresh = clone(p.calls);
      if (object && p.calls.length === 1 && equal(object.call, p.calls[0])) {
        branch = 'already_correct'; batches = [[call('repeat_answer', { target: object.id })]]; referenceTargets = [object.id];
      } else if (object) {
        fresh[0].arguments._replace = object.id;
        referenceTargets = [object.id];
      }
      batches ??= [fresh];
      if (branch !== 'already_correct' && object && p.calls.length === 1 && object.call.name === p.calls[0].name) {
        const changes = Object.fromEntries(Object.entries(p.calls[0].arguments).filter(([key, value]) => !Object.hasOwn(object.call.arguments, key) || !equal(object.call.arguments[key], value)));
        const remove = Object.keys(object.call.arguments).filter(key => !Object.hasOwn(p.calls[0].arguments, key));
        batches.push([call(`change_${object.call.name}`, { target: object.id, changes, ...(remove.length ? { remove } : {}) })]);
      }
    } else if (p.kind === 'pending_resolve' || p.kind === 'pending_cancel' || p.kind === 'pending_supersede') {
      const item = pending(before, p);
      branch = item ? 'present' : 'absent';
      if (item) {
        referenceTargets = [item.id];
        if (p.kind === 'pending_cancel') batches = [[call('cancel_pending', { target: item.id })]];
        else if (p.kind === 'pending_resolve') batches = [[call(item.function, { ...clone(p.values), _resolve: item.id })]];
        else batches = [[{ ...clone(p.calls[0]), arguments: { ...clone(p.calls[0].arguments), _supersede: item.id } }]];
      } else batches = p.kind === 'pending_cancel' ? [[call('clarify_reference', { kind: 'pending' })]] : [clone(p.fallback_calls ?? p.calls)];
    } else if (p.kind === 'ambiguous') {
      batches = [[call('clarify_reference', { kind: p.object_kind })]];
    } else throw new Error(`Unknown policy ${p.kind}`);
    const addressedTurn = before.semantic.turns.find(turn => turn.id === ex?.semantic_turn_id);
    const answerOnlyDisplay = ex?.outcome === 'committed' && addressedTurn?.answers.length > 0 &&
      addressedTurn.response_parts.length === addressedTurn.answers.length &&
      addressedTurn.response_parts.every((part, index) => part.kind === 'answer' && part.id === addressedTurn.answers[index].id) &&
      addressedTurn.response_parts.map(part => part.text).join('\n\n') === ex.assistant;
    if (p.kind === 'quote' && answerOnlyDisplay) {
      batches.push(addressedTurn.answers.map(item => call('repeat_answer', { target: item.id })));
    }
    if ((p.kind === 'repeat_answer' || (p.kind === 'replace' && branch === 'already_correct')) && object &&
        answerOnlyDisplay && addressedTurn.answers.length === 1 && addressedTurn.answers[0].id === object.id) {
      batches.push([call('repeat_exchange', { target: ex.id })]);
    }
    const question = row.questions?.[branch] ?? row.question;
    if (!question || !batches?.length || batches.some(b => !b?.length)) throw new Error(`Incomplete oracle ${row.case_id}/${branch}`);
    return { question, branch, allowed_batches: batches, reference_targets: referenceTargets,
      reference_target_alternatives: batches.map(refs),
      reference_eligible: row.reference_eligible && referenceTargets.length > 0,
      authored_reference_eligible: row.reference_eligible,
      before_visible_history_sha256: visibleHash(before) };
  }
  async function initial(registry, row) {
    let state = await createSession(registry, { dialogueVersion: 2, sessionId: `dev-${row.dialogue_id ?? row.case_id}` });
    for (const entry of row.initial_prefix ?? []) {
      if (entry.policy) {
        const resolved = resolve(entry, state);
        state = await applyModelOutput(registry, state, resolved.question, encodeCalls(resolved.allowed_batches[0]));
      } else state = await applyModelOutput(registry, state, entry.question, entry.raw_output ?? encodeCalls(entry.calls));
      if (!entry.expected_outcome || state.exchanges.at(-1).outcome !== entry.expected_outcome) throw new Error(`Prefix outcome mismatch ${row.case_id}`);
    }
    return state;
  }
  async function grade(registry, row, before, after, resolved) {
    const ex = after.exchanges.at(-1);
    const expected = await Promise.all(resolved.allowed_batches.map(batch => applyModelOutput(registry, before, resolved.question, encodeCalls(batch))));
    if (expected.some(state => state.exchanges.at(-1).outcome !== 'committed')) throw new Error(`Oracle execution failed ${row.case_id}/${resolved.branch}`);
    if (['quote', 'repeat_answer'].includes(row.policy.kind) || resolved.branch === 'already_correct') {
      for (const state of expected) {
        if (!equal(state.semantic.pending, before.semantic.pending) || !equal(state.semantic.turns.slice(0, -1), before.semantic.turns) ||
            state.exchanges.at(-1).assistant !== expected[0].exchanges.at(-1).assistant) throw new Error(`Incomplete quote alternative ${row.case_id}`);
      }
    }
    const appendOnly = after.exchanges.length === before.exchanges.length + 1 && before.exchanges.every((e, i) => equal(e, after.exchanges[i]));
    const actualRefs = refs(ex.model_calls ?? []);
    const allowedRefs = resolved.reference_target_alternatives;
    const referencesCorrect = allowedRefs.some(targets => equal(actualRefs, targets));
    const referenceSuccess = resolved.authored_reference_eligible && resolved.reference_targets.length > 0 && referencesCorrect;
    const pass = appendOnly && ex.question === resolved.question && expected.some(state =>
      ex.outcome === 'committed' && ex.assistant === state.exchanges.at(-1).assistant && equal(after.semantic, state.semantic));
    let critical_status = pass ? 'none_observed' : 'review_required', critical_category = null, failure_category = pass ? null : 'semantic_unresolved';
    if (!appendOnly || ex.question !== resolved.question) {
      critical_status = 'detected'; critical_category = 'history_integrity';
    } else if (ex.outcome !== 'committed') {
      critical_status = 'none_observed'; failure_category = 'execution_error';
    } else if (!pass && resolved.reference_targets.length && actualRefs.length && !referencesCorrect) {
      critical_status = 'detected'; critical_category = 'wrong_reference'; failure_category = 'wrong_reference';
    } else if (!pass && (ex.model_calls ?? []).some(c => c.name === 'clarify_reference' || c.arguments?._clarify) &&
      !resolved.allowed_batches.some(b => b.some(c => c.name === 'clarify_reference' || c.arguments?._clarify))) {
      critical_status = 'none_observed'; failure_category = 'unnecessary_clarification';
    } else if (!pass && resolved.reference_targets.length && !actualRefs.length && (ex.model_calls ?? []).some(c => !c.arguments?._clarify)) {
      critical_status = 'detected'; critical_category = 'omitted_reference'; failure_category = 'omitted_reference';
    }
    return { case_id: row.case_id, family_id: row.family_id, bootstrap_family_id: row.bootstrap_family_id ?? row.family_id, panel: row.panel, cohort: row.cohort,
      dialogue_id: row.dialogue_id ?? null, turn_index: row.turn_index, standalone: row.cohort === 'supported' && !row.dialogue_id,
      success: pass, critical_status, critical_category, failure_category,
      reference_eligible: row.reference_eligible, resolved_reference_available: resolved.reference_targets.length > 0,
      reference_success: row.reference_eligible ? referenceSuccess && ex.outcome === 'committed' : null,
      expected_reference_targets: resolved.reference_targets, expected_reference_alternatives: allowedRefs, selected_reference_targets: actualRefs };
  }
  function summarize(rows, results) {
    const byId = new Map(results.map(r => [r.case_id, r]));
    const successes = rows.filter(row => byId.get(row.case_id)?.success === true);
    const metric = select => {
      const planned = rows.filter(select), passed = successes.filter(select).length;
      return { planned: planned.length, passed, accuracy: planned.length ? passed / planned.length : null };
    };
    const dialogues = [...new Set(rows.filter(r => r.dialogue_id).map(r => r.dialogue_id))];
    const whole = dialogues.filter(id => rows.filter(r => r.dialogue_id === id).every(r => byId.get(r.case_id)?.success === true)).length;
    const referenceRows = rows.filter(r => r.reference_eligible);
    const referencePassed = referenceRows.filter(r => byId.get(r.case_id)?.reference_success === true).length;
    const criticalTypes = {};
    for (const r of results.filter(r => r.critical_status === 'detected')) criticalTypes[r.critical_category ?? 'other'] = (criticalTypes[r.critical_category ?? 'other'] ?? 0) + 1;
    return {
      supported: metric(r => r.cohort === 'supported'), outside_scope: metric(r => r.cohort === 'outside_scope'),
      whole_dialogues: { planned: dialogues.length, passed: whole, accuracy: dialogues.length ? whole / dialogues.length : null },
      standalone: metric(r => r.cohort === 'supported' && !r.dialogue_id),
      exact_reference: { planned: referenceRows.length, passed: referencePassed, accuracy: referenceRows.length ? referencePassed / referenceRows.length : null },
      missing: rows.filter(r => !byId.has(r.case_id)).length,
      critical_detected: results.filter(r => r.critical_status === 'detected').length,
      critical_review_required: results.filter(r => r.critical_status === 'review_required').length,
      history_critical: results.filter(r => r.critical_status === 'detected' && ['wrong_reference', 'omitted_reference', 'history_integrity'].includes(r.critical_category)).length,
      critical_types: criticalTypes, critical_by_kind: { history: results.filter(r => r.critical_status === 'detected' && ['wrong_reference', 'omitted_reference', 'history_integrity'].includes(r.critical_category)).length, other: results.filter(r => r.critical_status === 'detected' && !['wrong_reference', 'omitted_reference', 'history_integrity'].includes(r.critical_category)).length }, release_ready: false,
    };
  }
  return { initial, resolve, grade, summarize, visibleHash, refs };
}
