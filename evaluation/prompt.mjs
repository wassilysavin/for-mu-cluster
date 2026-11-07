import { assertSchema, canonical, ContractError, freeze } from './schema.mjs';

const FORMAT = 'gemprf-shared-tool-contract';
const VERSION = 1;
const PREFIX = ' Complete answers require: ';
const SUFFIX = '. Omission is permitted only for explicit clarification or pending completion.';
const ADDRESS = ['_replace', '_resolve', '_supersede'];
const META = ['_clarify', '_missing', '_each', ...ADDRESS];
const object = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const equal = (a, b) => canonical(a) === canonical(b);
const keysAre = (value, keys) => object(value) && equal(Object.keys(value).sort(), [...keys].sort());
const fail = message => { throw new ContractError('invalid_tool_contract', message); };

// Validate JSON without sorting member order: argument-name enums have an order
// even though the member order of a JSON object does not affect its semantics.
function copy(value) {
  canonical(value);
  return JSON.parse(JSON.stringify(value));
}

function walkSchema(schema, enumMapper, path = []) {
  const result = copy(schema);
  if (Object.hasOwn(schema, 'enum')) result.enum = enumMapper(schema.enum, path);
  if (schema.properties) {
    for (const [name, child] of Object.entries(schema.properties)) {
      result.properties[name] = walkSchema(child, enumMapper, [...path, name]);
    }
  }
  if (schema.items) result.items = walkSchema(schema.items, enumMapper, [...path, 'items']);
  for (const key of ['oneOf', 'anyOf']) {
    if (schema[key]) result[key] = schema[key].map(child => walkSchema(child, enumMapper, path));
  }
  // Do not recurse through const/enum literals, descriptions, or annotations.
  return result;
}

function supportedTool(tool) {
  if (!keysAre(tool, ['name', 'description', 'parameters']) ||
      typeof tool.name !== 'string' || typeof tool.description !== 'string') return false;
  try {
    assertSchema(tool.parameters);
    walkSchema(tool.parameters, values => {
      if (!Array.isArray(values)) throw new Error('An enum must be an array.');
      return values;
    });
    return true;
  } catch { return false; }
}

function domainCandidate(tool) {
  const parameters = tool.parameters;
  if (!keysAre(parameters, ['type', 'properties', 'required', 'additionalProperties']) ||
      parameters.type !== 'object' || parameters.additionalProperties !== false ||
      !equal(parameters.required, []) || !object(parameters.properties)) return null;
  const properties = parameters.properties;
  if (!properties._clarify || !properties._missing?.items ||
      !tool.description.endsWith(SUFFIX)) return null;
  const split = tool.description.lastIndexOf(PREFIX);
  if (split < 0) return null;
  const completeText = tool.description.slice(split + PREFIX.length, -SUFFIX.length);
  const complete = completeText === 'no mandatory arguments' ? [] : completeText.split(', ');
  const args = Object.fromEntries(Object.entries(properties).filter(([name]) => !META.includes(name)));
  const names = Object.keys(args);
  if (complete.some(name => !names.includes(name))) return null;
  const groupable = names.filter(name => !['object', 'array'].includes(args[name].type));
  if (!Array.isArray(properties._missing.items.enum) || !equal(properties._missing.items.enum, names) ||
      (groupable.length ? !Array.isArray(properties._each?.enum) || !equal(properties._each.enum, groupable) : Object.hasOwn(properties, '_each'))) return null;
  for (const name of ADDRESS) {
    if (properties[name] && !Array.isArray(properties[name].enum)) return null;
  }
  return { name: tool.name, description: tool.description.slice(0, split), Complete: complete, arguments: args };
}

function commonFor(candidates, tools) {
  const metadata = {};
  for (const name of META) {
    const index = candidates.findIndex((candidate, i) => candidate && tools[i].parameters.properties[name]);
    if (index < 0) continue;
    metadata[name] = copy(tools[index].parameters.properties[name]);
    if (name === '_missing') metadata[name].items.enum = 'ArgumentNames';
    if (name === '_each') metadata[name].enum = 'GroupableArgumentNames';
    if (ADDRESS.includes(name)) metadata[name].enum = 'Targets';
  }
  return {
    parameter_object: { type: 'object', required: [], additionalProperties: false },
    description_suffix: { prefix: PREFIX, join_complete_arguments_with: ', ', if_none: 'no mandatory arguments', suffix: SUFFIX },
    dialogue_metadata: metadata,
    inheritance_rules: [
      'Only entries without Tool or Raw inherit parameter_object and dialogue_metadata.',
      'ArgumentNames is the listed arguments in their listed order. GroupableArgumentNames excludes object and array arguments.',
      'Use ArgumentNames for _missing.items.enum; use GroupableArgumentNames for _each.enum, omitting _each if empty.',
      'An Addressing entry includes that metadata argument with its listed Targets; all other addressing metadata arguments are absent.',
      'Append description_suffix using Complete; Complete lists the arguments required for a complete answer.',
      'An enum domain reference expands to its entire literal array, preserving order; reference names are never emitted as argument values.',
    ],
  };
}

function restoreInherited(row, common, expand) {
  const args = {};
  for (const [name, schema] of Object.entries(row.arguments)) args[name] = walkSchema(schema, expand);
  const names = Object.keys(args);
  const groupable = names.filter(name => !['object', 'array'].includes(args[name].type));
  const metadata = common.dialogue_metadata;
  for (const name of ['_clarify', '_missing']) {
    if (!metadata[name]) fail(`Missing common ${name} definition.`);
    args[name] = copy(metadata[name]);
  }
  args._missing.items.enum = names;
  if (groupable.length) {
    if (!metadata._each) fail('Missing common _each definition.');
    args._each = copy(metadata._each);
    args._each.enum = groupable;
  }
  for (const [name, targets] of Object.entries(row.Addressing ?? {})) {
    if (!ADDRESS.includes(name) || !metadata[name]) fail(`Unknown addressing metadata ${name}.`);
    args[name] = copy(metadata[name]);
    args[name].enum = expand(targets);
  }
  const suffix = common.description_suffix;
  const complete = row.Complete.join(suffix.join_complete_arguments_with) || suffix.if_none;
  return { name: row.name, description: row.description + suffix.prefix + complete + suffix.suffix,
    parameters: { ...copy(common.parameter_object), properties: args } };
}

function domainName(toolName, path, values) {
  const argument = path[0] ?? 'argument';
  if (values.length === 60 && argument === 'subject') return 'Parameters';
  if (argument === '_replace') return 'AnswerTargets';
  if (argument === '_supersede') return 'PendingTargets';
  const words = argument === '_resolve' ? `${toolName}_pending_targets`
    : `${toolName.replace(/^(explain_|get_|describe_|interpret_|compare_)/, '')}_${argument}_values`;
  return words.split(/[^a-zA-Z0-9]+/).filter(Boolean).map(word => word[0].toUpperCase() + word.slice(1).toLowerCase()).join('') || 'Values';
}

function mapRow(row, mapper) {
  if (row.Raw !== undefined) return copy(row);
  if (row.Tool !== undefined) return { Tool: { ...copy(row.Tool), parameters: walkSchema(row.Tool.parameters, mapper) } };
  const result = copy(row);
  for (const [name, schema] of Object.entries(row.arguments)) result.arguments[name] = walkSchema(schema, mapper, [name]);
  for (const [name, values] of Object.entries(row.Addressing ?? {})) result.Addressing[name] = mapper(values, [name]);
  return result;
}

/** Lossless presentation only. Native function-call output is unchanged. */
export function packTools(input) {
  if (!Array.isArray(input)) fail('Tools must be an array.');
  const tools = copy(input);
  const supported = tools.map(supportedTool);
  const candidates = tools.map((tool, i) => supported[i] ? domainCandidate(tool) : null);
  const common = commonFor(candidates, tools);
  const rows = tools.map((tool, i) => {
    if (!supported[i]) return { Raw: tool };
    const row = candidates[i];
    if (row) {
      const addressing = Object.fromEntries(ADDRESS.filter(name => tool.parameters.properties[name])
        .map(name => [name, tool.parameters.properties[name].enum]));
      if (Object.keys(addressing).length) row.Addressing = addressing;
      // A differing annotation, root constraint or metadata detail prevents
      // inheritance; retain the entire tool instead of guessing equivalence.
      if (equal(restoreInherited(row, common, copy), tool)) return row;
    }
    return { Tool: tool };
  });
  const occurrences = new Map();
  const proposals = [];
  rows.forEach(row => mapRow(row, (values, path) => {
    const key = canonical(values);
    occurrences.set(key, (occurrences.get(key) ?? 0) + 1);
    proposals.push({ key, values, name: domainName(row.name ?? row.Tool?.name ?? '', path, values) });
    return values;
  }));
  const domains = {};
  const references = new Map();
  for (const { key, values, name: proposed } of proposals) {
    if (references.has(key) || occurrences.get(key) < 2) continue;
    let name = proposed;
    for (let n = 2; Object.hasOwn(domains, name); n += 1) name = `${proposed}${n}`;
    if (JSON.stringify(values).length <= name.length + 12) continue;
    domains[name] = values;
    references.set(key, name);
  }
  const packed = {
    format: FORMAT, version: VERSION,
    Reader: 'Read these shared JSON function contracts by expanding Common and Domains. Tool and Raw entries are complete tools without inheritance. Emit ordinary native function calls, never presentation keys or domain-reference names.',
    Common: common, Domains: domains,
    Functions: rows.map(row => mapRow(row, values => references.has(canonical(values)) ? { domain: references.get(canonical(values)) } : values)),
  };
  if (!equal(unpackTools(packed), tools)) fail('Tool packing did not round-trip exactly.');
  return freeze(packed);
}

export function unpackTools(input) {
  const packed = copy(input);
  if (!object(packed) || packed.format !== FORMAT || packed.version !== VERSION || !Array.isArray(packed.Functions) ||
      !object(packed.Domains) || !object(packed.Common)) fail('Unknown or malformed tool contract format.');
  const expand = values => {
    if (Array.isArray(values)) return copy(values);
    if (!keysAre(values, ['domain']) || typeof values.domain !== 'string' ||
        !Object.hasOwn(packed.Domains, values.domain) || !Array.isArray(packed.Domains[values.domain])) fail('Invalid enum domain reference.');
    return copy(packed.Domains[values.domain]);
  };
  return packed.Functions.map(row => {
    if (keysAre(row, ['Raw'])) return copy(row.Raw);
    if (keysAre(row, ['Tool'])) return { ...copy(row.Tool), parameters: walkSchema(row.Tool.parameters, expand) };
    const keys = ['name', 'description', 'Complete', 'arguments', ...(Object.hasOwn(row, 'Addressing') ? ['Addressing'] : [])];
    if (!keysAre(row, keys) || !object(row.arguments) || !Array.isArray(row.Complete) ||
        typeof row.name !== 'string' || typeof row.description !== 'string') fail('Unknown or malformed function entry.');
    return restoreInherited(row, packed.Common, expand);
  });
}

export function renderToolContract(tools) {
  return JSON.stringify(packTools(tools));
}
