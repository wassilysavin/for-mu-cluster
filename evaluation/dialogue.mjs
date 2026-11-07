import { canonical, clone, ContractError, freeze, validate } from './schema.mjs';

const string = { type: 'string', minLength: 1 };
const object = (properties, required = Object.keys(properties)) => ({
  type: 'object', properties, required, additionalProperties: false,
});

export const declineRequestMessage = "I can help with GEM-pRF parameter questions, but that part of the request is outside this assistant's scope.";
export const declineRequestTool = freeze({
  name: 'decline_request',
  description: 'Decline a request outside GEM-pRF parameter assistance. For a mixed request, also issue the supported calls in their requested order. Does not cancel pending requests. Do not use for missing source evidence, incomplete arguments or implementation errors.',
  parameters: object({}),
});

// Domain tools select semantic subjects themselves. These tools address only
// actual application-owned completed calls or pending questions by their IDs,
// or explicitly decline an outside-scope request without a scientific answer.
export const dialogueTools = freeze([
  declineRequestTool,
  { name: 'repeat_answer', description: 'Repeat the full answer for an actual completed call.',
    parameters: object({ target: string }) },
  { name: 'revise_call', description: 'Change named arguments of an actual completed call, retaining all other arguments. Re-execute atomically.',
    parameters: object({ target: string, changes: { type: 'object' }, remove: { type: 'array', items: string, uniqueItems: true } }, ['target', 'changes']) },
  { name: 'replace_call', description: 'Replace the selected completed operation with a fully specified new call.',
    parameters: object({ target: string, replacement: { type: 'object' } }) },
  { name: 'clarify_call', description: 'Retain a partially specified domain call and ask for missing arguments.',
    parameters: object({ function: string, known: { type: 'object' }, missing: { type: 'array', items: string, minItems: 1, uniqueItems: true },
      each: string, revision_target: string, pending_target: string }, ['function', 'known', 'missing']) },
  { name: 'complete_call', description: 'Fill missing arguments of an actual pending call, retaining already known arguments.',
    parameters: object({ target: string, values: { type: 'object' } }) },
  { name: 'replace_pending', description: 'Resolve an actual pending request with an explicitly changed complete domain call.',
    parameters: object({ target: string, replacement: { type: 'object' } }) },
  { name: 'cancel_pending', description: 'Cancel a specifically addressed pending request.',
    parameters: object({ target: string }) },
  { name: 'clarify_subjects', description: 'Ask which parameter(s) an identified function applies to. Mode each preserves an unresolved group and explicitly requests one call per member.',
    parameters: object({ function: string, mode: { type: 'string', enum: ['one', 'each'] } }) },
  { name: 'resolve_subjects', description: 'Supply the exact members for one actual pending subject question. Retains other pending questions.',
    parameters: object({ target: string, subjects: { type: 'array', items: string, minItems: 1, uniqueItems: true } }) },
]);

const dialogueByName = new Map(dialogueTools.map(t => [t.name, t]));
const callShape = object({ name: string, arguments: { type: 'object' } });

const referenceQuestions = {
  answer: 'Which earlier answer do you mean? If none matches, please restate your question.',
  exchange: 'Which earlier request do you mean? If none matches, please restate it.',
  pending: 'Which pending request do you mean? If none matches, please restate what you want to resolve.',
};

export function recoveryTools(exchanges) {
  if (exchanges === undefined) return [];
  const targets = exchanges.map(exchange => exchange.id);
  const failures = exchanges.filter(exchange => ['error', 'provider_error'].includes(exchange.outcome))
    .map(exchange => exchange.id);
  return [
    ...(targets.length ? [{ name: 'repeat_exchange',
      description: 'Repeat the entire displayed reply to this actual exchange verbatim, including an error or clarification. Do not try to answer the old request.',
      parameters: object({ target: { type: 'string', enum: targets } }) }] : []),
    ...(failures.length ? [{ name: 'retry_exchange',
      description: 'Start a batch that answers this actual failed exchange again. Follow with the fresh typed domain calls for its original request and the current corrections. Do not modify previous answers or pending items.',
      parameters: object({ target: { type: 'string', enum: failures } }) }] : []),
    { name: 'clarify_reference',
      description: 'Ask the user to identify or restate the intended prior answer, exchange or pending request when it cannot be safely addressed.',
      parameters: object({ kind: { type: 'string', enum: Object.keys(referenceQuestions) } }) },
  ];
}

export function createState() {
  return freeze({ version: 1, revision: 0, turns: [], pending: [], focus: [] });
}

function lookupAnswer(state, id) {
  const found = state.turns.flatMap(turn => turn.answers).find(answer => answer.id === id);
  if (!found) throw new ContractError('unknown_reference', `No actual completed call has ID ${id}.`);
  return found;
}

function makeAnswer(registry, call, id, extra = {}) {
  const { record, focus, delivery } = registry.resolve(call);
  const sources = [...new Map([...record.answer.sources, ...(delivery?.additional_sources ?? [])]
    .map(source => [canonical(source), source])).values()];
  return {
    id, call: clone(call), knowledge_id: record.id,
    meaning_id: record.meaning_id, status: delivery?.evidence_qualification?.status ?? record.answer.status,
    ...(delivery?.evidence_qualification ? { source_status: record.answer.status,
      evidence_qualification: clone(delivery.evidence_qualification) } : {}),
    knowledge_ids: clone(record.source_record_ids), meaning_ids: clone(record.meaning_ids),
    text: record.answer.text, ...(delivery ? { display_text: delivery.text } : {}), sources: clone(sources),
    limitations: [...record.answer.limitations,
      ...(delivery?.evidence_qualification ? [delivery.evidence_qualification.reason] : [])], focus: clone(focus), ...extra,
  };
}

export function renderAnswer(answer) {
  let text = answer.display_text ?? answer.text;
  const extra = answer.limitations.filter(item => !text.includes(item));
  if (extra.length) text += `\n\nLimitations: ${extra.join(' ')}`;
  return text;
}

function validateRetry(registry, calls, callIndex) {
  const fresh = call => registry.tool(call.name) ||
    (call.name === 'clarify_call' && !call.arguments.revision_target && !call.arguments.pending_target);
  const followers = calls.slice(1);
  followers.forEach(call => validate(callShape, call));
  if (callIndex !== 0 || !followers.some(fresh) ||
      followers.some(call => call.name !== 'decline_request' && !fresh(call))) {
    throw new ContractError('invalid_retry', 'A retry must start a batch of fresh domain calls, with at least one supported operation.');
  }
}

/**
 * Execute model-selected calls, never interpret the raw question. Compute the
 * entire turn against the previous committed state, then commit once. Rejected
 * batches return the identical original state; the UI may separately display
 * an error, but must not claim that this is a completed scientific answer.
 */
export function executeTurn(registry, state, question, inputCalls, exchanges) {
  try {
    if (state.version !== 1 || state.revision !== state.turns.length) throw new ContractError('invalid_state', 'Invalid state revision.');
    if (typeof question !== 'string' || !question.trim()) throw new ContractError('invalid_question', 'A nonempty question is required.');
    canonical(inputCalls);
    if (!Array.isArray(inputCalls) || inputCalls.length === 0) throw new ContractError('invalid_calls', 'At least one call is required.');
    const calls = clone(inputCalls);
    const turnId = `t${state.revision + 1}`;
    const answers = [];
    const pending = clone(state.pending);
    const responseParts = [];
    const effects = [];
    const touchedTargets = new Set();
    const groupProgress = new Map();
    const pendingRevisionOwners = new Map();
    const recovery = new Map(recoveryTools(exchanges).map(tool => [tool.name, tool]));

    function appendAnswer(call, extra) {
      const answer = makeAnswer(registry, call, `${turnId}.c${answers.length + 1}`, extra);
      answers.push(answer);
      responseParts.push({ kind: 'answer', id: answer.id, text: renderAnswer(answer) });
    }
    function touch(target) {
      if (touchedTargets.has(target)) throw new ContractError('conflicting_actions', `Target ${target} is changed more than once in this turn.`);
      touchedTargets.add(target);
    }
    function partialSchema(name) {
      const tool = registry.tool(name);
      if (!tool) throw new ContractError('unknown_function', `Function is not implemented: ${name}`);
      return { ...tool.parameters, required: [] };
    }
    function getPending(id) {
      const item = state.pending.find(p => p.id === id);
      if (!item) throw new ContractError('unknown_reference', `No actual pending question has ID ${id}.`);
      return item;
    }
    function claimPendingRevision(item) {
      if (!item.revision_target) return;
      if (item.mode === 'each' && pendingRevisionOwners.get(item.revision_target) === item.id) return;
      touch(item.revision_target);
      pendingRevisionOwners.set(item.revision_target, item.id);
    }
    function removePending(id) {
      const index = pending.findIndex(p => p.id === id);
      if (index < 0) throw new ContractError('conflicting_actions', `Pending request ${id} was already changed.`);
      pending.splice(index, 1);
    }
    function addPending(args, responseMarker) {
      const schema = partialSchema(args.function);
      validate(schema, args.known);
      const members = args.members ?? [];
      for (const member of members) {
        validate(callShape, member);
        if (member.name !== args.function) throw new ContractError('invalid_arguments', 'A pending group member has a different operation.');
        validate(schema, member.arguments);
      }
      if (args.missing.some(key => !Object.hasOwn(schema.properties, key) || Object.hasOwn(args.known, key))) {
        throw new ContractError('invalid_arguments', 'Missing arguments must be declared and genuinely absent.');
      }
      const required = registry.tool(args.function).parameters.required ?? [];
      if (required.some(key => !Object.hasOwn(args.known, key) && !args.missing.includes(key) &&
          (!members.length || members.some(member => !Object.hasOwn(member.arguments, key))))) {
        throw new ContractError('invalid_arguments', 'An omitted required argument must remain pending.');
      }
      if (args.each && (!Object.hasOwn(schema.properties, args.each) ||
          ['object', 'array'].includes(schema.properties[args.each].type) || Object.hasOwn(args.known, args.each) ||
          (members.length ? members.some(member => !Object.hasOwn(member.arguments, args.each)) : !args.missing.includes(args.each)))) {
        throw new ContractError('invalid_arguments', 'Distribution requires one explicitly selected scalar slot, supplied separately for each member.');
      }
      if (args.revision_target) lookupAnswer(state, args.revision_target);
      const id = `${turnId}.p${pending.filter(p => p.id.startsWith(`${turnId}.`)).length + 1}`;
      const item = { id, function: args.function, known: clone(args.known), missing: clone(args.missing),
        mode: args.each ? 'each' : 'one', ...(args.each ? { slot: args.each } : {}),
        ...(members.length ? { members: clone(members) } : {}),
        ...(args.revision_target ? { revision_target: args.revision_target } : {}) };
      pending.push(item);
      const labels = { subject: 'parameter', left: 'first item', right: 'second item', condition_subject: 'condition parameter',
        condition_value: 'whether the condition is enabled or disabled', direction: 'increase or decrease', value: 'value', component: 'component' };
      const requested = args.missing.map(key => labels[key] ?? key.replaceAll('_', ' '));
      const groupMissing = args.each && args.missing.includes(args.each);
      const otherRequested = args.missing.filter(key => key !== args.each).map(key => labels[key] ?? key.replaceAll('_', ' '));
      const text = groupMissing ?
        `${args.each === 'subject' ? 'Which parameters do you mean?' : `Which values do you mean for ${labels[args.each] ?? args.each.replaceAll('_', ' ')}?`}${otherRequested.length ? ` Please also specify ${otherRequested.join(' and ')}.` : ''}` :
        `Please specify ${requested.join(' and ')}.`;
      const response = { kind: 'clarification', id, text };
      if (responseMarker) responseParts.splice(responseParts.indexOf(responseMarker), 1, response);
      else responseParts.push(response);
      effects.push({ kind: 'pending_created', target: id,
        ...(args.revision_target ? { revises: args.revision_target } : {}) });
      return item;
    }

    // Group completions address an actual previous pending ID throughout this
    // batch. Only unfinished members become a new pending item at commit time.
    function completeGroup(item, values) {
      let progress = groupProgress.get(item.id);
      if (!progress) {
        touch(item.id);
        claimPendingRevision(item);
        removePending(item.id);
        const members = clone(item.members ?? []);
        const answered = [];
        const completedSlots = new Set();
        progress = { item, known: clone(item.known ?? {}), members, answered, completedSlots,
          required: [...new Set([...(registry.tool(item.function).parameters.required ?? []),
            ...(item.missing ?? [item.slot]), ...Object.keys(item.known ?? {}),
            ...members.flatMap(member => Object.keys(member.arguments)), item.slot])] };
        groupProgress.set(item.id, progress);
      }
      const keys = Object.keys(values);
      if (!keys.length || keys.some(key => !progress.required.includes(key) || Object.hasOwn(progress.known, key))) {
        throw new ContractError('invalid_arguments', 'Group completion must fill missing arguments without changing known common values.');
      }
      validate(partialSchema(item.function), values);
      if (Object.hasOwn(values, item.slot)) {
        const signature = canonical(values[item.slot]);
        if (progress.completedSlots.has(signature)) throw new ContractError('invalid_arguments', 'A completed group member is repeated.');
        const member = progress.members.find(member => canonical(member.arguments[item.slot]) === signature);
        if (member) {
          const changed = keys.filter(key => key !== item.slot);
          if (!changed.length || changed.some(key => Object.hasOwn(member.arguments, key))) {
            throw new ContractError('invalid_arguments', 'An existing group member may receive only its missing arguments.');
          }
          Object.assign(member.arguments, values);
        } else {
          progress.members.push({ name: item.function, arguments: { ...progress.known, ...values } });
        }
      } else {
        if (progress.members.length) {
          if (keys.some(key => progress.members.every(member => Object.hasOwn(member.arguments, key)))) {
            throw new ContractError('invalid_arguments', 'A common completion must fill a missing argument of at least one member.');
          }
          // Member-specific values are authoritative. Shared values fill only
          // holes and are retained for subsequently selected members.
          for (const member of progress.members) {
            for (const key of keys) if (!Object.hasOwn(member.arguments, key)) member.arguments[key] = values[key];
          }
        } else if (progress.answered.length) {
          throw new ContractError('invalid_arguments', 'All selected members are already complete; there is no common missing argument.');
        }
        Object.assign(progress.known, values);
      }
      const unfinished = [];
      for (const member of progress.members) {
        if (progress.required.some(key => !Object.hasOwn(member.arguments, key))) {
          unfinished.push(member);
          continue;
        }
        appendAnswer(member, { resolves: item.id,
          ...(item.revision_target ? { revises: item.revision_target } : {}) });
        const answerId = answers.at(-1).id;
        progress.answered.push(answerId);
        progress.completedSlots.add(canonical(member.arguments[item.slot]));
        if (item.revision_target) effects.push({ kind: 'revision', target: item.revision_target, replacement: answerId });
      }
      progress.members = unfinished;
      if ((unfinished.length || !progress.answered.length) && !progress.responseMarker) {
        progress.responseMarker = {};
        responseParts.push(progress.responseMarker);
      }
    }

    for (const [callIndex, call] of calls.entries()) {
      validate(callShape, call);
      const recoveryTool = recovery.get(call.name);
      if (recoveryTool) {
        validate(recoveryTool.parameters, call.arguments);
        const { kind, target } = call.arguments;
        if (call.name === 'retry_exchange') {
          validateRetry(registry, calls, callIndex);
          effects.push({ kind: 'exchange_retried', target });
        } else if (call.name === 'repeat_exchange') {
          const previous = exchanges.find(exchange => exchange.id === target);
          responseParts.push({ kind: 'replayed_exchange', id: target, text: previous.assistant });
          effects.push({ kind: 'exchange_repeated', target });
        } else {
          responseParts.push({ kind: 'reference_clarification', text: referenceQuestions[kind] });
          effects.push({ kind: 'reference_clarification', reference: kind });
        }
        continue;
      }
      const tool = dialogueByName.get(call.name);
      if (!tool) { appendAnswer(call); continue; }
      validate(tool.parameters, call.arguments);
      const args = call.arguments;
      if (call.name === 'decline_request') {
        // This branch is reachable only through the explicit model-selected
        // control. Failed domain calls never become a declined request.
        const id = `${turnId}.d${effects.filter(effect => effect.kind === 'declined_request').length + 1}`;
        responseParts.push({ kind: 'declined_request', id, text: declineRequestMessage });
        effects.push({ kind: 'declined_request', id, call_index: callIndex, text: declineRequestMessage });
      } else if (call.name === 'repeat_answer') {
        const previous = lookupAnswer(state, args.target);
        const { call: priorCall, knowledge_id, knowledge_ids, meaning_id, meaning_ids, status, source_status, evidence_qualification, text, display_text, sources, limitations, focus } = previous;
        const repeated = { id: `${turnId}.c${answers.length + 1}`, call: clone(priorCall), knowledge_id,
          knowledge_ids: clone(knowledge_ids), meaning_id, meaning_ids: clone(meaning_ids), status, text,
          ...(display_text !== undefined ? { display_text } : {}),
          ...(evidence_qualification ? { source_status, evidence_qualification: clone(evidence_qualification) } : {}),
          sources: clone(sources), limitations: clone(limitations), focus: clone(focus), repeats: previous.id };
        answers.push(repeated);
        responseParts.push({ kind: 'answer', id: repeated.id, text: renderAnswer(repeated) });
        effects.push({ kind: 'repeat', target: previous.id, answer: repeated.id });
      } else if (call.name === 'revise_call') {
        touch(args.target);
        const previous = lookupAnswer(state, args.target);
        if (!Object.keys(args.changes).length && !args.remove?.length) throw new ContractError('invalid_arguments', 'Correction has no changes.');
        validate(partialSchema(previous.call.name), args.changes);
        const revised = { name: previous.call.name, arguments: { ...previous.call.arguments, ...args.changes } };
        for (const key of args.remove ?? []) {
          if (!Object.hasOwn(previous.call.arguments, key) || Object.hasOwn(args.changes, key)) {
            throw new ContractError('invalid_arguments', 'Removed arguments must exist and cannot also be changed.');
          }
          delete revised.arguments[key];
        }
        if (canonical(revised) === canonical(previous.call)) throw new ContractError('invalid_arguments', 'Correction does not change the call.');
        appendAnswer(revised, { revises: previous.id });
        effects.push({ kind: 'revision', target: previous.id, replacement: answers.at(-1).id });
      } else if (call.name === 'replace_call') {
        touch(args.target);
        const previous = lookupAnswer(state, args.target);
        if (canonical(args.replacement) === canonical(previous.call)) throw new ContractError('invalid_arguments', 'Replacement does not change the call.');
        appendAnswer(args.replacement, { revises: previous.id });
        effects.push({ kind: 'revision', target: previous.id, replacement: answers.at(-1).id });
      } else if (call.name === 'clarify_call') {
        const clarified = clone(args);
        if (args.pending_target) {
          touch(args.pending_target);
          const previous = getPending(args.pending_target);
          if (previous.revision_target) {
            if (clarified.revision_target && clarified.revision_target !== previous.revision_target) {
              throw new ContractError('conflicting_actions', 'A pending correction cannot silently change its original target.');
            }
            clarified.revision_target = previous.revision_target;
          }
          removePending(args.pending_target);
        }
        if (clarified.revision_target) touch(clarified.revision_target);
        const added = addPending(clarified);
        if (args.pending_target) effects.push({ kind: 'pending_revision', target: args.pending_target, replacement: added.id });
      } else if (call.name === 'complete_call') {
        const item = getPending(args.target);
        if (item.mode === 'each') {
          completeGroup(item, args.values);
          continue;
        }
        const known = item.known ?? {};
        const missing = item.missing ?? [item.slot];
        if (!Object.keys(args.values).length || Object.keys(args.values).some(key => !missing.includes(key))) {
          throw new ContractError('invalid_arguments', 'Completion may supply only missing arguments; changing known arguments requires an explicit replacement.');
        }
        validate(partialSchema(item.function), args.values);
        const combined = { ...known, ...args.values };
        const remaining = missing.filter(key => !Object.hasOwn(combined, key));
        touch(item.id); removePending(item.id);
        claimPendingRevision(item);
        if (remaining.length) {
          const updated = addPending({ function: item.function, known: combined, missing: remaining,
            ...(item.revision_target ? { revision_target: item.revision_target } : {}) });
          effects.push({ kind: 'pending_progress', target: item.id, replacement: updated.id });
          continue;
        }
        appendAnswer({ name: item.function, arguments: combined }, { resolves: item.id,
          ...(item.revision_target ? { revises: item.revision_target } : {}) });
        effects.push({ kind: 'pending_resolved', target: item.id });
        if (item.revision_target) effects.push({ kind: 'revision', target: item.revision_target, replacement: answers.at(-1).id });
      } else if (call.name === 'replace_pending') {
        touch(args.target);
        const item = getPending(args.target);
        removePending(item.id);
        claimPendingRevision(item);
        appendAnswer(args.replacement, { resolves: item.id,
          ...(item.revision_target ? { revises: item.revision_target } : {}) });
        effects.push({ kind: 'pending_replaced', target: item.id, answer: answers.at(-1).id });
        if (item.revision_target) effects.push({ kind: 'revision', target: item.revision_target, replacement: answers.at(-1).id });
      } else if (call.name === 'cancel_pending') {
        touch(args.target);
        getPending(args.target);
        removePending(args.target);
        responseParts.push({ kind: 'cancellation', id: args.target, text: 'That pending request has been cancelled.' });
        effects.push({ kind: 'pending_cancelled', target: args.target });
      } else if (call.name === 'clarify_subjects') {
        const domain = registry.tool(args.function);
        // Initial slice has single-subject functions only. Additional argument
        // slots require explicit future contracts, not guessed placeholders.
        if (!domain || canonical(domain.parameters.required) !== canonical(['subject']) ||
            Object.keys(domain.parameters.properties).length !== 1) {
          throw new ContractError('clarification_not_implemented', 'This function needs a different declared clarification contract.');
        }
        const id = `${turnId}.p${pending.filter(p => p.id.startsWith(`${turnId}.`)).length + 1}`;
        const item = { id, function: args.function, mode: args.mode, slot: 'subject' };
        pending.push(item);
        responseParts.push({ kind: 'clarification', id,
          text: args.mode === 'each' ? 'Which parameters do you mean?' : 'Which parameter do you mean?' });
        effects.push({ kind: 'pending_created', target: id });
      } else if (call.name === 'resolve_subjects') {
        touch(args.target);
        // References must come from actual previous state, not a fabricated or
        // not-yet-displayed pending item in this same model output.
        const item = state.pending.find(p => p.id === args.target);
        if (!item) throw new ContractError('unknown_reference', `No actual pending question has ID ${args.target}.`);
        if ((item.missing && canonical(item.missing) !== canonical(['subject'])) || (item.slot && item.slot !== 'subject')) {
          throw new ContractError('invalid_arguments', 'This pending request is not a subject-only clarification.');
        }
        if (item.mode === 'one' && args.subjects.length !== 1) throw new ContractError('invalid_arguments', 'This pending question requires one parameter.');
        claimPendingRevision(item);
        for (const subject of args.subjects) {
          appendAnswer({ name: item.function, arguments: { ...(item.known ?? {}), subject } }, { resolves: item.id,
            ...(item.revision_target ? { revises: item.revision_target } : {}) });
          if (item.revision_target) effects.push({ kind: 'revision', target: item.revision_target, replacement: answers.at(-1).id });
        }
        pending.splice(pending.findIndex(p => p.id === item.id), 1);
        effects.push({ kind: 'pending_resolved', target: item.id });
      }
    }
    for (const progress of groupProgress.values()) {
      const { item, members, answered, responseMarker } = progress;
      if (members.length || !answered.length) {
        const missing = progress.required.filter(key => members.length ?
          members.some(member => !Object.hasOwn(member.arguments, key)) : !Object.hasOwn(progress.known, key));
        const updated = addPending({ function: item.function, known: progress.known, missing,
          each: item.slot, members,
          ...(item.revision_target ? { revision_target: item.revision_target } : {}) }, responseMarker);
        effects.push({ kind: 'pending_progress', target: item.id, replacement: updated.id });
      } else {
        if (responseMarker) responseParts.splice(responseParts.indexOf(responseMarker), 1);
        effects.push({ kind: 'pending_resolved', target: item.id });
      }
    }
    const text = responseParts.map(p => p.text).join('\n\n');
    const turn = { id: turnId, question, calls, answers, response_parts: responseParts, text, effects };
    const focus = answers.length ? [...new Set(answers.flatMap(a => a.focus))] : clone(state.focus);
    const next = freeze({ version: 1, revision: state.revision + 1, turns: [...state.turns, turn], pending, focus });
    return freeze({ ok: true, state: next, turn: next.turns.at(-1) });
  } catch (error) {
    if (!(error instanceof ContractError)) throw error;
    return freeze({ ok: false, state, error: { code: error.code, message: error.message, details: error.details } });
  }
}

export function modelHistory(state) {
  // Full actually displayed messages remain authoritative. IDs and prior calls
  // are metadata for explicit references; they are never substitute answers.
  return clone({ turns: state.turns.map(turn => ({ id: turn.id, user: turn.question, assistant: turn.text,
    calls: turn.answers.map(answer => ({ id: answer.id, ...answer.call,
      ...(answer.revises ? { revises: answer.revises } : {}),
      ...(answer.resolves ? { resolves: answer.resolves } : {}),
      ...(answer.repeats ? { repeats: answer.repeats } : {}),
      ...(answer.evidence_qualification ? { evidence_status: answer.status, source_evidence_status: answer.source_status } : {}),
    })), actions: turn.effects,
    clarifications: turn.response_parts.filter(p => p.kind === 'clarification') })),
    pending: state.pending, focus: state.focus });
}
