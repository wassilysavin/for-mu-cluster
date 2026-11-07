// Small, explicit subset for our declarative domain contracts. Unsupported schema
// constructs fail closed instead of silently weakening a function's requirements.
const keywords = new Set(['type', 'properties', 'required', 'additionalProperties',
  'items', 'minItems', 'maxItems', 'uniqueItems', 'minLength', 'maxLength',
  'enum', 'const', 'oneOf', 'anyOf', 'description', 'title']);
const dangerous = new Set(['__proto__', 'prototype', 'constructor']);

export class ContractError extends Error {
  constructor(code, message, details = {}) {
    super(message);
    this.name = 'ContractError';
    this.code = code;
    this.details = details;
  }
}

export function canonical(value) {
  if (value === null || typeof value === 'boolean' || typeof value === 'string') return JSON.stringify(value);
  if (typeof value === 'number' && Number.isFinite(value)) return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value && Object.getPrototypeOf(value) === Object.prototype) {
    const keys = Object.keys(value).sort();
    if (keys.some(k => dangerous.has(k))) throw new ContractError('invalid_json', 'Unsafe object property.');
    return `{${keys.map(k => `${JSON.stringify(k)}:${canonical(value[k])}`).join(',')}}`;
  }
  throw new ContractError('invalid_json', 'Only finite JSON values are allowed.');
}

export function clone(value) {
  return JSON.parse(canonical(value));
}

export function freeze(value) {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    Object.values(value).forEach(freeze);
    Object.freeze(value);
  }
  return value;
}

export function assertSchema(schema) {
  if (!schema || typeof schema !== 'object' || Array.isArray(schema)) throw new Error('Invalid schema.');
  for (const key of Object.keys(schema)) {
    if (!keywords.has(key)) throw new Error(`Unsupported schema keyword: ${key}`);
  }
  if (schema.type && !['object', 'array', 'string', 'number', 'integer', 'boolean', 'null'].includes(schema.type)) {
    throw new Error(`Unsupported type: ${schema.type}`);
  }
  if (schema.type === 'object' && schema.additionalProperties !== false) {
    throw new Error('Every contract object must explicitly reject extra properties.');
  }
  Object.values(schema.properties ?? {}).forEach(assertSchema);
  if (schema.items) assertSchema(schema.items);
  for (const key of ['oneOf', 'anyOf']) (schema[key] ?? []).forEach(assertSchema);
}

export function validate(schema, value, path = '$') {
  const fail = message => { throw new ContractError('invalid_arguments', `${path}: ${message}`); };
  if (schema.const !== undefined && canonical(schema.const) !== canonical(value)) fail('unexpected constant');
  if (schema.enum && !schema.enum.some(x => canonical(x) === canonical(value))) fail('value is not in the declared domain');
  for (const key of ['oneOf', 'anyOf']) {
    if (!schema[key]) continue;
    const matches = schema[key].filter(s => {
      try { validate(s, value, path); return true; } catch (e) { if (e instanceof ContractError) return false; throw e; }
    }).length;
    if (key === 'oneOf' ? matches !== 1 : matches === 0) {
      throw new ContractError('invalid_arguments', `${path}: does not satisfy ${key}`,
        typeof schema.description === 'string' ? { clarification: schema.description } : {});
    }
  }
  const kind = value === null ? 'null' : Array.isArray(value) ? 'array' : typeof value;
  if (schema.type && !(schema.type === 'integer' ? Number.isInteger(value) : schema.type === kind)) fail(`expected ${schema.type}`);
  if (kind === 'object') {
    for (const key of schema.required ?? []) if (!Object.hasOwn(value, key)) fail(`missing ${key}`);
    for (const [key, item] of Object.entries(value)) {
      if (dangerous.has(key)) fail('unsafe property');
      if (Object.hasOwn(schema.properties ?? {}, key)) validate(schema.properties[key], item, `${path}.${key}`);
      else if (schema.additionalProperties === false) fail(`unexpected ${key}`);
    }
  }
  if (kind === 'array') {
    if (schema.minItems !== undefined && value.length < schema.minItems) fail('too few items');
    if (schema.maxItems !== undefined && value.length > schema.maxItems) fail('too many items');
    if (schema.uniqueItems && new Set(value.map(canonical)).size !== value.length) fail('duplicate items');
    if (schema.items) value.forEach((v, i) => validate(schema.items, v, `${path}[${i}]`));
  }
  if (kind === 'string') {
    if (schema.minLength !== undefined && [...value].length < schema.minLength) fail('empty or too short');
    if (schema.maxLength !== undefined && [...value].length > schema.maxLength) fail('too long');
  }
  if (kind === 'number' && !Number.isFinite(value)) fail('non-finite number');
}
