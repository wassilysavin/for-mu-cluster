import assert from 'node:assert/strict';
import { createReadStream } from 'node:fs';
import { resolve } from 'node:path';
import { createInterface } from 'node:readline';
import { pathToFileURL } from 'node:url';
import { isDeepStrictEqual as equal } from 'node:util';
import { loadPanel, summarize } from './diagnose-dev.mjs';
import { createActualHistoryHarness, hash } from './evaluation/actual-history.mjs';
import { decodeCalls, encodeCalls } from './evaluation/functiongemma.mjs';
import { renderPrompt } from './evaluation/history-train.mjs';
import { buildModelInput, pinnedTokenizerSha256 } from './evaluation/model-input.mjs';
import { canonical, ContractError } from './evaluation/schema.mjs';
import { applyModelOutput, createSession, recordIncompleteModelOutput, recordProviderFailure,
  sessionHistory } from './evaluation/session.mjs';

const api = { applyModelOutput, createSession, encodeCalls, canonical, sessionHistory,
  buildModelInput, pinnedTokenizerSha256 };
const harness = createActualHistoryHarness(api);
const referenceKeys = new Set(['target', '_replace', '_resolve', '_supersede']);
const referenceId = /^(?:e\d+|t\d+\.[cp]\d+)$/;
const sorted = set => Array.from(set).sort();

function operationCounts(calls) {
  const counts = new Map();
  for (const { name } of calls) counts.set(name, (counts.get(name) ?? 0) + 1);
  return counts;
}

function programDifferences(actual, expected) {
  const signals = new Set(), fields = new Set();
  if (actual.length === expected.length &&
      equal(actual.map(canonical).sort(), expected.map(canonical).sort())) {
    return { signals: ['call_order'], argument_fields: [] };
  }
  if (actual.length < expected.length) signals.add('fewer_calls');
  if (actual.length > expected.length) signals.add('extra_calls');
  const a = operationCounts(actual), b = operationCounts(expected);
  const unmatched = (left, right) => [...left].some(([k, n]) => n > (right.get(k) ?? 0));
  if (unmatched(a, b) && unmatched(b, a)) signals.add('operation_substitution');
  const sameSequence = equal(actual.map(c => c.name), expected.map(c => c.name));
  if (!sameSequence && equal(a, b)) signals.add('call_order');
  if (sameSequence) {
    for (let i = 0; i < actual.length; i++) {
      const left = actual[i].arguments, right = expected[i].arguments;
      for (const key of new Set([...Object.keys(left), ...Object.keys(right)])) {
        if (Object.hasOwn(left, key) && Object.hasOwn(right, key) && equal(left[key], right[key])) continue;
        if (!Object.hasOwn(left, key)) signals.add('missing_argument');
        else if (!Object.hasOwn(right, key)) signals.add('extra_argument');
        else signals.add('argument_value');
        if (referenceKeys.has(key) && [left[key], right[key]].some(
          value => typeof value === 'string' && referenceId.test(value))) signals.add('reference_argument');
        fields.add(key);
      }
    }
  }
  return { signals: sorted(signals), argument_fields: sorted(fields) };
}

export function comparePrograms(actual, allowedBatches) {
  assert.ok(Array.isArray(allowedBatches) && allowedBatches.length, 'Missing frozen allowed programs');
  if (allowedBatches.some(batch => equal(batch, actual))) {
    return { signals: [], argument_fields: [], comparison: 'exact_allowed' };
  }
  const alternatives = allowedBatches.map(batch => programDifferences(actual, batch));
  const common = field => alternatives[0][field].filter(item => alternatives.every(alt => alt[field].includes(item)));
  return { signals: common('signals'), argument_fields: common('argument_fields'),
    comparison: alternatives.every(alt => equal(alt, alternatives[0])) ? 'different' : 'alternative_dependent' };
}

// Match the recorded FunctionGemma protocol in evaluate-actual.mjs, including its token limit.
async function replayGeneration(registry, before, question, gen) {
  if (gen.error) {
    assert.ok(['context_overflow', 'inference_failed'].includes(gen.error), 'Unknown provider failure');
    return recordProviderFailure(registry, before, question, gen.error);
  }
  const ids = gen.raw_ids;
  assert.ok(typeof gen.raw_text === 'string' && Array.isArray(ids) && ids.length > 0 && ids.length <= 512 &&
    ids.every(id => Number.isInteger(id) && id >= 0 && id < 262144), 'Invalid saved native generation');
  if (ids.at(-1) !== 50) {
    assert.ok(ids.length === 512 && !ids.includes(50), 'Invalid termination of saved native generation');
    return recordIncompleteModelOutput(registry, before, question, gen.raw_text, ids, 'token_limit');
  }
  assert.ok(!ids.slice(0, -1).includes(50), 'Stop token before end of saved generation');
  return applyModelOutput(registry, before, question, gen.raw_text);
}

function decodedBatch(exchange) {
  if (exchange.outcome === 'committed') return exchange.model_calls;
  if (exchange.outcome === 'provider_error' || exchange.generation?.complete_native_batch === false) return null;
  try { return decodeCalls(exchange.raw_output); }
  catch (error) {
    if (!(error instanceof ContractError)) throw error;
    return null;
  }
}

function exampleNames(calls) {
  const names = calls.slice(0, 5).map(c => c.name.slice(0, 80));
  if (calls.length > names.length) names.push('...');
  return names;
}

export function caseRecord(row, observed, actual) {
  const exchange = observed.after.exchanges.at(-1);
  const result = { case_id: row.case_id, family_id: row.family_id, panel: row.panel, cohort: row.cohort,
    dialogue_id: row.dialogue_id ?? null, turn_index: row.turn_index, group: row.group,
    success: observed.success, critical_status: observed.critical_status,
    critical_category: observed.critical_category, failure_category: observed.failure_category,
    reference_success: observed.reference_success, outcome: exchange.outcome, error_code: exchange.error_code ?? null,
    comparison: observed.success ? 'passed' : 'not_decodable', signals: [], argument_fields: [] };
  if (observed.success) return result;
  const batches = observed.expected_model_call_batches;
  const displayTargets = ids => ids.slice(0, 5).map(id => referenceId.test(id) ? id.slice(0, 32) : '(invalid ID)');
  result.selected_targets = displayTargets(observed.selected_reference_targets ?? []);
  result.first_allowed_targets = displayTargets(observed.expected_reference_alternatives?.[0] ?? []);
  const examples = batches.map(exampleNames);
  result.expected_names = examples.filter((value, i) => examples.findIndex(other => equal(other, value)) === i).slice(0, 3);
  result.actual_names = actual === null ? null : exampleNames(actual);
  if (actual !== null) Object.assign(result, comparePrograms(actual, batches));
  return result;
}

export async function inspectArm(context, observationsPath) {
  const { plan, data, registry, bundleHash } = context;
  const lastIndex = new Map(), sessions = new Map(), rows = [], infrastructure = [];
  const keyOf = row => JSON.stringify([row.panel, row.dialogue_id]);
  plan.forEach((row, i) => { if (row.dialogue_id) lastIndex.set(keyOf(row), i); });
  const stream = createReadStream(observationsPath, { encoding: 'utf8' });
  const lines = createInterface({ input: stream, crlfDelay: Infinity });
  try {
    for await (const line of lines) {
      const index = rows.length, row = plan[index];
      assert.ok(row, 'Extra saved DEV observation');
      let observed;
      try { observed = JSON.parse(line); } catch { throw new Error('Invalid DEV JSON at row ' + (index + 1)); }
      assert.ok(observed.case_id === row.case_id && observed.bundle_sha256 === bundleHash,
        'Different DEV case, order or bundle at row ' + (index + 1));
      const key = keyOf(row), before = sessions.get(key) ?? await harness.initial(registry, row);
      const resolved = harness.resolve(row, before);
      const prompt = await renderPrompt(registry, api, before, resolved.question);
      const gen = observed.generation;
      const after = await replayGeneration(registry, before, resolved.question, gen);
      const grade = await harness.grade(registry, row, before, after, resolved);
      const replay = { ...grade, bundle_sha256: bundleHash, prompt_sha256: hash(prompt),
        before_visible_history_sha256: resolved.before_visible_history_sha256, question: resolved.question,
        policy_branch: resolved.branch, expected_model_call_batches: resolved.allowed_batches,
        generation: gen, before, after };
      assert.ok(canonical(observed) === canonical(replay), 'Saved state, prompt or grade differs from replay for ' + row.case_id);
      const exchange = after.exchanges.at(-1);
      if (exchange.outcome === 'provider_error' && exchange.error_code !== 'context_overflow') infrastructure.push(row.case_id);
      rows.push(caseRecord(row, replay, grade.success ? null : decodedBatch(exchange)));
      if (row.dialogue_id && lastIndex.get(key) !== index) sessions.set(key, after);
      else sessions.delete(key);
    }
  } finally { lines.close(); stream.destroy(); }
  assert.equal(rows.length, plan.length, 'Incomplete saved DEV observations');
  const summary = summarize(data, rows);
  const score = { ...summary.summary, ...summary, bundle_sha256: bundleHash, diagnostic_reference_panel: 'fixed_prefix',
    infrastructure_failures: infrastructure.length, infrastructure_failure_case_ids: infrastructure,
    quality_valid: infrastructure.length === 0 };
  return { rows, score, bundle_sha256: bundleHash };
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  try {
    assert.equal(process.argv.length, 5, 'Expected registry, approved DEV bundle and observations path');
    const context = await loadPanel(process.argv[2], process.argv[3]);
    process.stdout.write(JSON.stringify(await inspectArm(context, process.argv[4])) + '\n');
  } catch (error) {
    process.stderr.write('Saved student DEV inspection failed: ' + error.message + '\n');
    process.exitCode = 1;
  }
}
