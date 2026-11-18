/** Grade a frozen TRAIN continuation against the session actually shown to the model. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { createInterface } from 'node:readline';
import { pathToFileURL } from 'node:url';
import { createActualHistoryHarness } from './evaluation/actual-history.mjs';
import * as native from './evaluation/functiongemma.mjs';
import { continuation, materializeScenario, renderPrompt } from './evaluation/history-train.mjs';
import * as modelInput from './evaluation/model-input.mjs';
import { createRegistry } from './evaluation/registry.mjs';
import { canonical } from './evaluation/schema.mjs';
import * as session from './evaluation/session.mjs';
import { createTrainDiagnostics, materializeGeneration } from './diagnose-data.mjs';
import { gradeTransfer, transferRegistry, upgradePacked } from './transfer_contract.mjs';
import { observedSession } from './transfer_train.mjs';

const api = { ...session, ...native, ...modelInput, canonical };
const harness = createActualHistoryHarness(api);
const hash = text => createHash('sha256').update(text).digest('hex');
const userTurn = '<end_of_turn>\n<start_of_turn>user\n';
const modelTurn = '<end_of_turn>\n<start_of_turn>model\n';
const contract = '\n\n{"format":"gemprf-shared-tool-contract",';

async function* records(path) {
  const stream = createReadStream(path);
  try {
    for await (const line of createInterface({ input: stream, crlfDelay: Infinity })) yield JSON.parse(line);
  } finally { stream.destroy(); }
}

function insert(map, key, value) {
  assert.ok(typeof key === 'string' && !map.has(key), 'Duplicate or missing TRAIN identifier: ' + key);
  map.set(key, value);
}

function sharedPrompt(original, bundle, cache) {
  const [prefix, rest, ...extra] = original.split(contract);
  assert.ok(rest && !extra.length, 'Expected exactly one packed shared tool contract');
  const [tools, payload, ...tail] = rest.split(userTurn);
  assert.ok(payload && !tail.length && payload.endsWith(modelTurn));
  if (!cache.has(tools)) {
    const packed = JSON.parse(contract.slice(2) + tools);
    cache.set(tools, upgradePacked(packed, bundle).replaceAll('<', '\\u003c'));
  }
  return prefix + '\n\n' + cache.get(tools) + userTurn + payload;
}

function payload(prompt) {
  const parts = prompt.split(userTurn);
  assert.ok(parts.length === 2 && parts[1].endsWith(modelTurn), 'Unexpected frozen TRAIN prompt boundaries');
  return JSON.parse(parts[1].slice(0, -modelTurn.length));
}

export async function createScorer(bundle) {
  const original = createRegistry(bundle.knowledge, bundle.contracts);
  const originalId = await original.fingerprint();
  assert.equal(originalId, bundle.fingerprint);
  const registry = await transferRegistry(bundle);
  const revisedId = await registry.fingerprint();
  const diagnostics = new Map([original, registry].map(r => [r, createTrainDiagnostics(r)]));
  const cases = new Map();
  const contracts = new Map();

  async function prepare({ case_id, prompt, target }, row, before, { section, migrated = false, branch }) {
    const historicalRegistry = migrated ? original : registry;
    const historicalBefore = migrated ? { ...before, registry_id: originalId } : before;
    const revisedBefore = before.registry_id === revisedId ? before : { ...before, registry_id: revisedId };
    assert.equal(target, native.encodeCalls(row.calls), 'Frozen TRAIN target differs: ' + case_id);
    const historicalPrompt = await renderPrompt(historicalRegistry, api, historicalBefore, row.question);
    const expected = section === 'shared' ? sharedPrompt(historicalPrompt, bundle, contracts)
      : migrated ? await renderPrompt(registry, api, revisedBefore, row.question) : historicalPrompt;
    assert.ok(prompt === expected, 'Frozen TRAIN prompt differs: ' + case_id);
    const item = await diagnostics.get(historicalRegistry).prepare({ source: section === 'shared' ? 'common' : 'history',
      row, before: historicalBefore, prompt: historicalPrompt });
    assert.ok(item.semantic_eligible && item.allowed_batches?.length, 'Missing TRAIN oracle: ' + case_id);
    if (migrated) {
      const expected = await session.applyModelOutput(registry, revisedBefore, row.question, target);
      assert.equal(expected.exchanges.at(-1).outcome, 'committed');
      for (const batch of item.allowed_batches.slice(1)) {
        const alternative = await session.applyModelOutput(registry, revisedBefore, row.question, native.encodeCalls(batch));
        assert.equal(alternative.exchanges.at(-1).outcome, 'committed');
        assert.equal(alternative.exchanges.at(-1).assistant, expected.exchanges.at(-1).assistant);
        assert.deepEqual(alternative.semantic.pending, expected.semantic.pending);
        assert.deepEqual(alternative.semantic.turns.slice(0, -1), expected.semantic.turns.slice(0, -1));
      }
    }
    const meta = { case_id, section, group: row.group, cohort: item.cohort, prompt_sha256: hash(prompt),
      target_sha256: hash(target) };
    if (cases.has(case_id)) assert.deepEqual(cases.get(case_id).meta, meta, 'Inconsistent repeated TRAIN row');
    const targets = harness.refs(row.calls);
    cases.set(case_id, { meta, before: revisedBefore, target, calls: row.calls,
      resolved: { question: row.question, branch: branch === 'requested_answer_missing' ? 'absent' : branch,
        allowed_batches: item.allowed_batches, reference_targets: targets,
        reference_target_alternatives: item.allowed_batches.map(harness.refs) } });
    return { kind: 'prepared', ...meta };
  }

  async function score(caseId, generation) {
    const item = cases.get(caseId);
    assert.ok(item, 'Case was never prepared: ' + caseId);
    assert.equal(generation.provider_kind, 'functiongemma');
    assert.equal(generation.input_sha256, item.meta.prompt_sha256, 'Generated against a different TRAIN input');
    assert.ok(!generation.error, 'TRAIN inference infrastructure failure: ' + generation.error);
    assert.equal(typeof generation.raw_text, 'string');
    assert.ok(Array.isArray(generation.raw_ids) && generation.raw_ids.length <= 512);
    assert.equal(generation.complete, generation.raw_ids.at(-1) === 50);
    assert.equal(generation.finish_reason, generation.complete ? 'complete_batch' : 'length');
    const { resolved, before, meta } = item;
    const after = await materializeGeneration(registry, before, resolved.question, generation);
    const result = await harness.grade(registry, { ...meta, family_id: caseId, panel: 'train_fixed',
      policy: { kind: 'fresh' }, turn_index: null, reference_eligible: resolved.reference_targets.length > 0 }, before, after, resolved);
    const grade = gradeTransfer(result, resolved, before, after);
    let calls;
    if (generation.complete) {
      try { calls = native.decodeCalls(generation.raw_text); } catch { /* Preserved as invalid transport. */ }
    }
    return { kind: 'scored', observation: { ...meta, success: grade.success, critical_status: grade.critical_status,
      critical_category: grade.critical_category, failure_category: grade.failure_category,
      native_literal_exact: generation.complete && generation.raw_text === item.target,
      canonical_call_exact: Boolean(calls && canonical(calls) === canonical(item.calls)),
      transport_valid: Boolean(calls) } };
  }
  return { original, registry, prepare, score };
}

async function loadSources(registryPath, init) {
  const dir = dirname(registryPath);
  const bundle = JSON.parse(await readFile(registryPath, 'utf8'));
  const scorer = await createScorer(bundle);
  const shared = new Map(), contexts = new Map();
  for await (const row of records(join(dir, 'train-full-v2.jsonl'))) {
    insert(shared, 'shared:' + row.case_id, { case_id: 'shared:' + row.case_id, group: 'shared_replay',
      question: row.question, calls: row.calls });
  }
  for await (const row of records(init.contexts ?? join(dir, 'diagnostic-train-contexts.jsonl'))) {
    insert(contexts, 'shared:' + row.case_id, row.before);
  }
  const roots = new Map(), cases = new Map(), observed = new Map(), specs = new Map();
  for await (const root of records(join(dir, 'transfer-train-roots.jsonl'))) insert(roots, root.root_id, root);
  const observedIds = new Set([...roots.values()].map(root => root.observed_prefix_id).filter(Boolean));
  const specPath = join(dir, 'history-train-specs.jsonl');
  for await (const spec of records(specPath)) if (observedIds.has(spec.prefix_id)) {
    const prior = specs.get(spec.prefix_id);
    if (prior) {
      assert.deepEqual(prior.prefix, spec.prefix);
      assert.equal(prior.session_id, spec.session_id);
    } else specs.set(spec.prefix_id, spec);
  }
  const specsDigest = hash(await readFile(specPath));
  for await (const row of records(init.receipts)) if (observedIds.has(row.prefix_id)) insert(observed, row.prefix_id, row);
  assert.equal(observed.size, observedIds.size, 'Real 7205 TRAIN prefix receipts are incomplete');
  for (const arm of ['ordinary', 'contrast']) {
    for await (const row of records(join(dir, 'transfer-train-' + arm + '.jsonl'))) insert(cases, row.case_id, row);
  }
  const states = new Map();
  async function prepare(request) {
    const id = request.case_id;
    if (shared.has(id)) {
      const data = payload(request.prompt);
      const before = contexts.get(id) ?? await session.createSession(scorer.original, {
        sessionId: data.history.session_id, dialogueVersion: data.format.endsWith('v2') ? 2 : 1 });
      return scorer.prepare(request, shared.get(id), before, { section: 'shared', migrated: true, branch: 'frozen_train' });
    }
    const row = cases.get(id);
    assert.ok(row, 'Unknown TRAIN row: ' + id);
    const root = roots.get(row.root_id);
    if (!states.has(root.root_id)) {
      const old = root.observed_prefix_id;
      const state = old ? await observedSession(scorer.registry, scorer.original, observed.get(old), root, specs.get(old), specsDigest)
        : (await materializeScenario(scorer.registry, api, { ...root, case_id: id, session_id: root.root_id,
          continuation: row.continuation })).before;
      states.set(root.root_id, state);
    }
    const before = states.get(root.root_id);
    const target = continuation(before, row.continuation);
    return scorer.prepare(request, { case_id: id, group: root.group, ...target }, before,
      { section: 'variable', migrated: Boolean(root.observed_prefix_id), branch: target.branch });
  }
  return { prepare, score: scorer.score };
}

async function main(registryPath) {
  let server;
  for await (const line of createInterface({ input: process.stdin, crlfDelay: Infinity })) {
    const req = JSON.parse(line);
    let result;
    if (req.action === 'init' && !server) {
      server = await loadSources(registryPath, req);
      result = { kind: 'initialized' };
    } else if (req.action === 'prepare') result = await server.prepare(req);
    else if (req.action === 'score') result = await server.score(req.case_id, req.generation);
    else throw new Error('Unexpected TRAIN-fit RPC action: ' + req.action);
    await new Promise((yes, no) => process.stdout.write(JSON.stringify(result) + '\n', err => err ? no(err) : yes()));
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) await main(process.argv[2]);
