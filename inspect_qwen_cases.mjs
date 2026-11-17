import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { createInterface } from 'node:readline';
import { pathToFileURL } from 'node:url';
import { createTrainDiagnostics } from './diagnose-data.mjs';
import { createRegistry } from './evaluation/registry.mjs';
import { canonical, clone, ContractError, validate } from './evaluation/schema.mjs';
import { compileModelCalls, modelTools } from './evaluation/model-tools.mjs';
import { executeTurn } from './evaluation/dialogue.mjs';
import { decodeCalls, encodeCalls } from './evaluation/functiongemma.mjs';
import { applyModelOutput } from './evaluation/session.mjs';

const equal = (a, b) => canonical(a) === canonical(b);
const hash = value => createHash('sha256').update(typeof value === 'string' ? value : canonical(value)).digest('hex');
const errorData = error => ({ code: error.code, message: error.message, details: clone(error.details ?? {}) });
const refs = calls => calls.flatMap(call => ['target', '_replace', '_resolve', '_supersede']
  .filter(key => typeof call.arguments?.[key] === 'string').map(key => call.arguments[key]));

function attempt(operation) {
  try { return { value: operation() }; }
  catch (error) {
    if (!(error instanceof ContractError)) throw error;
    return { error: errorData(error) };
  }
}

function firstFailure(registry, before, question, calls, internal, stage, error) {
  const exchanges = before.dialogue_version === 2 ? before.exchanges : undefined;
  if (stage === 'execute' && internal.some(call => call.name === 'complete_call' &&
      before.semantic.pending.some(item => item.id === call.arguments.target && item.mode === 'each'))) {
    return { call_index: null, attribution: 'unassigned_group_finalization_depends_on_full_batch' };
  }
  for (let count = 1; count <= calls.length; count++) {
    const outcome = stage === 'compile' ?
      attempt(() => compileModelCalls(registry, before.semantic, calls.slice(0, count), exchanges)) :
      executeTurn(registry, before.semantic, question, internal.slice(0, count), exchanges);
    const found = outcome.error;
    if (!found) continue;
    if (equal(errorData(found), error)) {
      return { call_index: count - 1, attribution: stage === 'compile' ?
        'first_failing_compile_prefix_same_frozen_state' : 'first_failing_execution_prefix_no_group_finalization' };
    }
    // Retry validates its followers; a shortened prefix may not contain one yet.
    if (stage === 'execute' && found.code === 'invalid_retry' && internal[0].name === 'retry_exchange') continue;
    return { call_index: null, attribution: 'unassigned_prefix_differs_from_full_batch' };
  }
  return { call_index: null, attribution: 'unassigned' };
}

function traceBatch(registry, item, full) {
  const before = item.before;
  const exchanges = before.dialogue_version === 2 ? before.exchanges : undefined;
  const decoded = attempt(() => decodeCalls(full.generation.raw_text));
  if (decoded.error) return { model_calls: null, trace: { stage: 'decode', error: decoded.error, call_index: null } };
  const calls = decoded.value;
  const compiled = attempt(() => compileModelCalls(registry, before.semantic, calls, exchanges));
  let stage = 'compile', failure = compiled.error, executed;
  if (!failure) {
    stage = 'execute';
    executed = executeTurn(registry, before.semantic, item.question, compiled.value, exchanges);
    if (!executed.ok) failure = errorData(executed.error);
  }
  const saved = full.after.exchanges.at(-1);
  if (failure) {
    assert.equal(saved.outcome, 'error', 'Full-batch replay differs from saved outcome');
    assert.equal(saved.error_code, failure.code, 'Full-batch replay differs from saved error');
    return { model_calls: calls, internal_calls: compiled.value ?? null,
      trace: { stage, error: failure,
        ...firstFailure(registry, before, item.question, calls, compiled.value, stage, failure) } };
  }
  assert.equal(saved.outcome, 'committed', 'Full-batch replay differs from saved outcome');
  assert.equal(saved.assistant, executed.turn.text, 'Full-batch delivered text differs');
  assert.ok(equal(full.after.semantic, executed.state), 'Full-batch semantic state differs');
  return { model_calls: calls, internal_calls: compiled.value,
    trace: { stage: 'committed', error: null, call_index: null, attribution: 'no_execution_failure' } };
}

function schemaFragment(schema, message) {
  const path = message?.match(/^(\$(?:\.[^.:[\]\s]+|\[\d+\])*):/)?.[1] ?? '$';
  const keys = path.match(/[^.\[\]$]+|\[\d+\]/g) ?? [];
  let selected = schema;
  for (const key of keys) selected = key.startsWith('[') ? selected?.items : selected?.properties?.[key];
  const rootRule = message?.match(/: (?:missing|unexpected) (\w+)$/)?.[1];
  if (rootRule && selected?.type === 'object') {
    return { path, fragment: { type: selected.type, required: selected.required ?? [],
      additionalProperties: selected.additionalProperties ?? null,
      declared_properties: Object.keys(selected.properties ?? {}),
      field: rootRule, field_schema: selected.properties?.[rootRule] ?? null } };
  }
  return { path, path_resolved: selected !== undefined, fragment: clone(selected ?? schema) };
}

function schemaEvidence(name, source, schema, args, originalMessage) {
  const checked = attempt(() => validate(schema, args));
  return { name, source, valid: !checked.error,
    error: checked.error ?? null, parameters_sha256: hash(schema),
    ...schemaFragment(schema, checked.error?.message ?? originalMessage) };
}

function collectSchemas(registry, item, traced) {
  const exchanges = item.before.dialogue_version === 2 ? item.before.exchanges : undefined;
  const exposed = new Map(modelTools(registry, item.before.semantic, exchanges).map(tool => [tool.name, tool]));
  const index = traced.trace.call_index;
  const chosen = index === null ? (traced.model_calls ?? []).map((call, callIndex) => [callIndex, call]) :
    [[index, traced.model_calls[index]]];
  const schemas = chosen.flatMap(([callIndex, call]) => {
    const tool = exposed.get(call.name);
    return tool ? [{ call_index: callIndex, attributed: index !== null,
      ...schemaEvidence(call.name, 'dynamic_model_tool', tool.parameters,
      call.arguments, traced.trace.error?.message) }] : [];
  });
  const modelCall = index === null ? null : traced.model_calls[index];
  const domainCall = traced.trace.error?.details?.call ??
    (modelCall && registry.tool(modelCall.name) && !modelCall.arguments._clarify && !modelCall.arguments._resolve ? {
      name: modelCall.name,
      arguments: Object.fromEntries(Object.entries(modelCall.arguments)
        .filter(([key]) => !['_clarify', '_missing', '_each', '_replace', '_resolve', '_supersede'].includes(key))),
    } : null);
  if (domainCall && registry.tool(domainCall.name)) schemas.push({ call_index: index,
    ...schemaEvidence(domainCall.name, 'domain_contract', registry.tool(domainCall.name).parameters,
      domainCall.arguments, traced.trace.error?.message) });
  return schemas;
}

function bindingEvidence(registry, contracts, domainCall) {
  assert.ok(domainCall && registry.tool(domainCall.name), 'Missing exact domain call for contract failure');
  const validation = attempt(() => registry.validateCall(domainCall));
  assert.ok(!validation.error, 'Unbound contract call unexpectedly fails schema validation');
  const sameFunction = contracts.bindings.filter(binding => binding.call.name === domainCall.name);
  const subject = domainCall.arguments.subject;
  const sameSubject = sameFunction.filter(binding => Object.hasOwn(domainCall.arguments, 'subject') &&
    Object.hasOwn(binding.call.arguments, 'subject') && equal(binding.call.arguments.subject, subject));
  assert.ok(!sameFunction.some(binding => equal(binding.call, domainCall)), 'Exact missing binding unexpectedly exists');
  const sample = bindings => bindings.slice(0, 3).map(binding => ({ call: clone(binding.call),
    knowledge_ids: clone(binding.knowledge_ids ?? [binding.knowledge_id]) }));
  return { domain_call: clone(domainCall), schema_valid: true, exact_binding_present: false,
    same_function_count: sameFunction.length, same_function_subject_count: sameSubject.length,
    same_function_examples: sample(sameFunction), same_function_subject_examples: sample(sameSubject),
    interpretation: 'Schema-valid call has no exact frozen binding; model error versus missing implementation is not adjudicated.' };
}

function summarize(value) {
  if (typeof value === 'string') return value.length <= 240 ? value : `${value.slice(0, 240)}…`;
  if (Array.isArray(value)) return { kind: 'array', length: value.length, sha256: hash(value) };
  if (value && typeof value === 'object') return { kind: 'object', keys: Object.keys(value), sha256: hash(value) };
  return value;
}

function differences(actual, expected, limit = 8) {
  const entries = [];
  let truncated = false;
  function walk(a, b, path) {
    if (equal(a, b)) return;
    if (entries.length >= limit) { truncated = true; return; }
    if (a && b && typeof a === 'object' && typeof b === 'object' && Array.isArray(a) === Array.isArray(b)) {
      for (const key of new Set([...Object.keys(a), ...Object.keys(b)])) {
        if (!Object.hasOwn(a, key) || !Object.hasOwn(b, key)) {
          if (entries.length >= limit) { truncated = true; break; }
          entries.push({ path: `${path}.${key}`, actual_present: Object.hasOwn(a, key), expected_present: Object.hasOwn(b, key),
            actual: Object.hasOwn(a, key) ? summarize(a[key]) : null,
            expected: Object.hasOwn(b, key) ? summarize(b[key]) : null });
        } else walk(a[key], b[key], `${path}.${key}`);
        if (truncated) break;
      }
    } else entries.push({ path, actual: summarize(a), expected: summarize(b) });
  }
  walk(actual, expected, '$');
  return { entries, truncated, limit };
}

function textComparison(actual, expected) {
  let first = 0;
  while (first < actual.length && first < expected.length && actual[first] === expected[first]) first++;
  const start = Math.max(0, first - 80), end = first + 320;
  return { equal: actual === expected, actual_sha256: hash(actual), expected_sha256: hash(expected),
    actual_chars: actual.length, expected_chars: expected.length,
    first_difference: actual === expected ? null : first,
    excerpt_start: start, actual_excerpt_end: Math.min(actual.length, end),
    expected_excerpt_end: Math.min(expected.length, end),
    actual_excerpt: actual.slice(start, end), expected_excerpt: expected.slice(start, end) };
}

async function reviewEvidence(registry, item, full) {
  const actual = full.after;
  const actualExchange = actual.exchanges.at(-1);
  const alternatives = [];
  for (const [index, batch] of item.allowed_batches.entries()) {
    const expected = await applyModelOutput(registry, item.before, item.question, encodeCalls(batch));
    const expectedExchange = expected.exchanges.at(-1);
    assert.equal(expectedExchange.outcome, 'committed', 'Frozen allowed program fails execution');
    alternatives.push({ index, calls: clone(batch), reference_targets: refs(batch),
      expected: { delivered: expectedExchange.assistant, semantic: clone(expected.semantic) },
      delivered: textComparison(actualExchange.assistant, expectedExchange.assistant),
      semantic: { equal: equal(actual.semantic, expected.semantic), actual_sha256: hash(actual.semantic),
        expected_sha256: hash(expected.semantic), ...differences(actual.semantic, expected.semantic) } });
  }
  return { adjudicated: false, status: full.critical_status,
    actual: { delivered: actualExchange.assistant, semantic: clone(actual.semantic) },
    failure_category: full.semantic_result?.failure_category ?? null,
    selected_reference_targets: full.semantic_result?.selected_reference_targets ?? [], alternatives };
}

export function createCaseInspector(registry, contracts) {
  const diagnostic = createTrainDiagnostics(registry);
  return async (item, observation) => {
    const full = await diagnostic.expand(item, observation);
    assert.ok(full.semantic_eligible && full.after, 'Inspection needs a reproducible before/after state');
    assert.ok(['contract_not_implemented', 'invalid_arguments'].includes(full.error_kind) ||
      full.critical_status === 'review_required', 'Case is outside the requested inspection groups');
    const traced = traceBatch(registry, item, full);
    const result = { case_id: item.case_id, case_sha256: item.case_sha256,
      error_kind: full.error_kind, critical_status: full.critical_status,
      receipt_verified: true, scores_changed: false, ...traced,
      schemas: collectSchemas(registry, item, traced) };
    if (full.error_kind === 'contract_not_implemented') result.contract = bindingEvidence(registry, contracts,
      traced.trace.error?.details?.call);
    if (full.critical_status === 'review_required') result.review = await reviewEvidence(registry, item, full);
    return result;
  };
}

export async function inspectCase(registry, contracts, item, observation) {
  return createCaseInspector(registry, contracts)(item, observation);
}

async function main() {
  assert.ok(process.argv[2], 'Expected registry JSON path');
  const bundle = JSON.parse(await readFile(process.argv[2], 'utf8'));
  const registry = createRegistry(bundle.knowledge, bundle.contracts);
  assert.equal(await registry.fingerprint(), bundle.fingerprint);
  const inspect = createCaseInspector(registry, bundle.contracts);
  for await (const line of createInterface({ input: process.stdin, crlfDelay: Infinity })) {
    const request = JSON.parse(line);
    assert.equal(request.action, 'inspect', 'Unknown inspection action');
    const output = { kind: 'inspection', inspection: await inspect(request.case, request.observation) };
    await new Promise((yes, no) => process.stdout.write(`${JSON.stringify(output)}\n`, error => error ? no(error) : yes()));
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) await main();
