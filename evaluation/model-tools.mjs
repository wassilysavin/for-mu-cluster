import { clone, ContractError, freeze, validate, assertSchema } from './schema.mjs';
import { declineRequestTool, recoveryTools } from './dialogue.mjs';

const string = { type: 'string', minLength: 1 };
const object = (properties, required = []) => ({ type: 'object', properties, required, additionalProperties: false });
const metaKeys = ['_clarify', '_missing', '_each', '_replace', '_resolve', '_supersede'];

/** Concrete schemas for the current real history. No untyped changes object,
 * universal program string, opaque source ID or implicit source selection. */
export function modelTools(registry, state, exchanges) {
  const answers = state.turns.flatMap(turn => turn.answers);
  const answerIds = answers.map(answer => answer.id);
  const pendingIds = state.pending.map(item => item.id);
  const result = [clone(declineRequestTool)];
  for (const domain of registry.tools) {
    const properties = clone(domain.parameters.properties);
    if (Object.keys(properties).some(key => metaKeys.includes(key))) throw new Error('Domain argument collides with dialogue metadata.');
    const keys = Object.keys(properties);
    properties._clarify = { type: 'boolean', const: true, description: 'Ask for missing arguments instead of answering.' };
    properties._missing = { type: 'array', items: { type: 'string', enum: keys }, minItems: 1, uniqueItems: true,
      description: 'Missing optional arguments, if needed; required missing arguments are retained automatically.' };
    const scalarKeys = keys.filter(key => !['object', 'array'].includes(domain.parameters.properties[key].type));
    if (scalarKeys.length) properties._each = { type: 'string', enum: scalarKeys,
      description: 'Clarify an unknown group, then perform one call per member for this argument.' };
    if (answerIds.length) properties._replace = { type: 'string', enum: answerIds, description: 'Address an actual answer to replace with this operation.' };
    const matchingPending = state.pending.filter(item => item.function === domain.name).map(item => item.id);
    if (matchingPending.length) properties._resolve = { type: 'string', enum: matchingPending, description: 'Supply missing arguments of this actual pending request.' };
    if (pendingIds.length) properties._supersede = { type: 'string', enum: pendingIds, description: 'Explicitly change this pending request to this operation.' };
    result.push({ name: domain.name,
      description: `${domain.description} Complete answers require: ${(domain.parameters.required ?? []).join(', ') || 'no mandatory arguments'}. Omission is permitted only for explicit clarification or pending completion.`,
      parameters: { ...clone(domain.parameters), properties, required: [] } });

    const targets = answers.filter(answer => answer.call.name === domain.name).map(answer => answer.id);
    if (targets.length) result.push({
      name: `change_${domain.name}`,
      description: `Change only named arguments of an actual ${domain.name} answer. Retain all other arguments and conditions.`,
      parameters: object({ target: { type: 'string', enum: targets },
        changes: { ...clone(domain.parameters), required: [] },
        remove: { type: 'array', items: { type: 'string', enum: keys }, uniqueItems: true },
      }, ['target', 'changes']),
    });
  }
  if (answerIds.length) result.push({ name: 'repeat_answer', description: 'Repeat exactly one actual answer.',
    parameters: object({ target: { type: 'string', enum: answerIds } }, ['target']) });
  if (pendingIds.length) result.push({ name: 'cancel_pending', description: 'Cancel only this actual pending request.',
    parameters: object({ target: { type: 'string', enum: pendingIds } }, ['target']) });
  result.push(...recoveryTools(exchanges));
  const names = result.map(tool => tool.name);
  if (new Set(names).size !== names.length) throw new Error('Domain and dialogue function names collide.');
  result.forEach(tool => assertSchema(tool.parameters));
  return freeze(result);
}

export function compileModelCalls(registry, state, input, exchanges) {
  if (!Array.isArray(input) || !input.length) throw new ContractError('invalid_calls', 'At least one model call is required.');
  const exposed = new Map(modelTools(registry, state, exchanges).map(tool => [tool.name, tool]));
  const controlNames = new Set(['decline_request', 'repeat_answer', 'cancel_pending',
    ...recoveryTools(exchanges).map(tool => tool.name)]);
  const calls = clone(input);
  return calls.map(call => {
    validate(object({ name: string, arguments: { type: 'object' } }, ['name', 'arguments']), call);
    const tool = exposed.get(call.name);
    if (!tool) throw new ContractError('unknown_function', 'The model selected a function not declared for this conversation.');
    validate(tool.parameters, call.arguments);
    if (controlNames.has(call.name)) return call;
    if (!registry.tool(call.name)) {
      if (!call.name.startsWith('change_')) throw new ContractError('unknown_function', 'Unknown dialogue function.');
      return { name: 'revise_call', arguments: call.arguments };
    }
    const { _clarify, _missing, _each, _replace, _resolve, _supersede, ...args } = call.arguments;
    if ([_replace, _resolve, _supersede].filter(Boolean).length > 1) throw new ContractError('conflicting_actions', 'A call cannot address multiple different actions.');
    if ((_missing || _each) && !_clarify) throw new ContractError('invalid_arguments', 'Missing-argument metadata requires explicit clarification.');
    if (_clarify && _resolve) throw new ContractError('invalid_arguments', 'Pending completion already retains missing arguments; it cannot also start a separate clarification.');
    if (_clarify) {
      const missing = [...new Set([...(registry.tool(call.name).parameters.required ?? []).filter(key => !Object.hasOwn(args, key)), ...(_missing ?? [])])];
      if (!missing.length) throw new ContractError('invalid_arguments', 'There is no missing argument to clarify.');
      return { name: 'clarify_call', arguments: { function: call.name, known: args, missing,
        ...(_each ? { each: _each } : {}), ...(_replace ? { revision_target: _replace } : {}),
        ...(_supersede ? { pending_target: _supersede } : {}),
      } };
    }
    if (_resolve) return { name: 'complete_call', arguments: { target: _resolve, values: args } };
    const replacement = { name: call.name, arguments: args };
    registry.validateCall(replacement);
    if (_replace) return { name: 'replace_call', arguments: { target: _replace, replacement } };
    if (_supersede) return { name: 'replace_pending', arguments: { target: _supersede, replacement } };
    return replacement;
  });
}
