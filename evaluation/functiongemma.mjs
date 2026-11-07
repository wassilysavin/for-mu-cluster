import { canonical, ContractError } from './schema.mjs';

const START = '<start_function_call>call:';
const END = '<end_function_call>';
const DONE = '<start_function_response>';
const ESC = '<escape>';
const identifier = /^[A-Za-z_][A-Za-z0-9_]*/;

// This parses only the vendor's function-call transport. It never parses an
// English question, selects a source, repairs a subject or drops a failed call.
export function decodeCalls(output) {
  if (typeof output !== 'string' || output.length > 131072) throw new ContractError('invalid_model_output', 'Invalid function output length.');
  let at = 0;
  const fail = (message, code = 'invalid_model_output') => { throw new ContractError(code, message, { offset: at }); };
  const whitespace = () => { while (/\s/.test(output[at] ?? '') && at < output.length) at++; };
  function consume(token) {
    if (!output.startsWith(token, at)) fail(`Expected ${token}.`);
    at += token.length;
  }
  function name() {
    const match = identifier.exec(output.slice(at));
    if (!match) fail('Expected an argument or function name.');
    at += match[0].length;
    if (['__proto__', 'constructor', 'prototype'].includes(match[0])) fail('Unsafe argument name.');
    return match[0];
  }
  function value(depth = 0) {
    if (depth > 24) fail('Function arguments are nested too deeply.');
    whitespace();
    if (output.startsWith(ESC, at)) {
      at += ESC.length;
      const end = output.indexOf(ESC, at);
      if (end < 0) fail('Unterminated string.', 'incomplete_model_output');
      const result = output.slice(at, end);
      at = end + ESC.length;
      return result;
    }
    if (output[at] === '{') {
      at++;
      const result = {};
      whitespace();
      if (output[at] === '}') { at++; return result; }
      while (true) {
        whitespace();
        const key = name();
        if (Object.hasOwn(result, key)) fail('Duplicate argument key.');
        whitespace(); consume(':');
        result[key] = value(depth + 1);
        whitespace();
        if (output[at] === '}') { at++; return result; }
        consume(',');
      }
    }
    if (output[at] === '[') {
      at++;
      const result = [];
      whitespace();
      if (output[at] === ']') { at++; return result; }
      while (true) {
        result.push(value(depth + 1));
        whitespace();
        if (output[at] === ']') { at++; return result; }
        consume(',');
      }
    }
    for (const [token, result] of [['true', true], ['false', false], ['null', null]]) {
      if (output.startsWith(token, at) && !/[A-Za-z0-9_]/.test(output[at + token.length] ?? '')) {
        at += token.length; return result;
      }
    }
    const match = /^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?/.exec(output.slice(at));
    if (match) {
      at += match[0].length;
      const result = Number(match[0]);
      if (!Number.isFinite(result)) fail('Non-finite number.');
      return result;
    }
    fail('Expected a typed function argument.');
  }
  const calls = [];
  whitespace();
  while (output.startsWith(START, at)) {
    consume(START);
    const functionName = name();
    const args = value();
    if (args === null || Array.isArray(args) || typeof args !== 'object') fail('Arguments must be an object.');
    consume(END);
    calls.push({ name: functionName, arguments: args });
    whitespace();
  }
  if (!calls.length) fail('The model did not return a function call.');
  // An end-of-call marker is insufficient: more obligations may follow. The
  // model must explicitly yield the complete batch to the function executor.
  if (!output.startsWith(DONE, at)) fail('Missing complete-batch marker.', 'incomplete_model_output');
  consume(DONE);
  whitespace();
  if (at !== output.length) fail('Unexpected content after the function batch.');
  canonical(calls);
  return calls;
}

export function encodeCalls(calls) {
  canonical(calls);
  if (!Array.isArray(calls) || calls.length === 0) throw new ContractError('invalid_calls', 'At least one call is required.');
  const safeName = key => {
    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(key) || ['__proto__', 'constructor', 'prototype'].includes(key)) {
      throw new ContractError('invalid_transport_key', 'A native function argument name is invalid.');
    }
    return key;
  };
  function format(value) {
    if (typeof value === 'string') {
      if (value.includes(ESC)) throw new ContractError('invalid_transport_literal', 'The vendor string delimiter cannot occur inside a literal.');
      return `${ESC}${value}${ESC}`;
    }
    if (Array.isArray(value)) return `[${value.map(format).join(',')}]`;
    if (value && typeof value === 'object') return `{${Object.keys(value).sort().map(key => `${safeName(key)}:${format(value[key])}`).join(',')}}`;
    return JSON.stringify(value);
  }
  return calls.map(call => {
    if (Object.keys(call).sort().join(',') !== 'arguments,name' || !call.arguments || Array.isArray(call.arguments) || typeof call.arguments !== 'object') {
      throw new ContractError('invalid_calls', 'Expected a named call with an argument object.');
    }
    return `${START}${safeName(call.name)}${format(call.arguments)}${END}`;
  }).join('') + DONE;
}

export const functionBatchStopToken = DONE;
