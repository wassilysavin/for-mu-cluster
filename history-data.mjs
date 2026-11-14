import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { mkdir, open, readFile, stat, writeFile } from 'node:fs/promises';
import { createInterface } from 'node:readline';
import { dirname, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { applyObservation, materializeScenario, renderPrompt } from './evaluation/history-train.mjs';
import { matchedControl } from './evaluation/history-control.mjs';

const sha = value => createHash('sha256').update(value).digest('hex');
const line = row => `${JSON.stringify(row)}\n`;
const send = row => new Promise((yes, no) => process.stdout.write(line(row), error => error ? no(error) : yes()));
const args = Object.fromEntries(process.argv.slice(3).reduce((result, arg, index, all) => {
  if (arg.startsWith('--')) result.push([arg.slice(2), all[index + 1]]);
  return result;
}, []));
const command = process.argv[2];
assert.ok(['check', 'materialize', 'collect'].includes(command), 'Use check, materialize or collect');
for (const key of ['specs', 'registry', 'runtime']) assert.ok(args[key], `--${key} is required`);
const load = name => import(pathToFileURL(resolve(args.runtime, `${name}.mjs`)).href);
const api = Object.assign({}, ...await Promise.all(['session', 'functiongemma', 'model-input'].map(load)));
const { createRegistry } = await load('registry');
const bundle = JSON.parse(await readFile(args.registry, 'utf8'));
const registry = createRegistry(bundle.knowledge, bundle.contracts);
const specBytes = await readFile(args.specs);
const specs = specBytes.toString('utf8').trimEnd().split('\n').map(JSON.parse);
const prefixSpecs = [...new Map(specs.map(spec => [spec.prefix_id, spec])).values()];
const immutable = { specs_sha256: sha(specBytes), registry_sha256: await registry.fingerprint() };
assert.equal(new Set(specs.map(spec => spec.case_id)).size, specs.length);

function validateObservation(observed) {
  assert.ok(['native', 'incomplete'].includes(observed.kind));
  assert.equal(typeof observed.raw_output, 'string');
  assert.ok(Array.isArray(observed.raw_ids) && observed.raw_ids.length > 0 && observed.raw_ids.length <= 512);
  assert.ok(observed.raw_ids.every(id => Number.isInteger(id) && id >= 0 && id < 262144));
  assert.ok(!observed.raw_ids.slice(0, -1).includes(50));
  if (observed.kind === 'native') assert.equal(observed.raw_ids.at(-1), 50);
  else { assert.equal(observed.raw_ids.length, 512); assert.ok(!observed.raw_ids.includes(50)); }
}

async function existing(path) {
  try { return await stat(path); } catch (error) { if (error.code === 'ENOENT') return null; throw error; }
}
async function readReceipts(path) {
  const receipts = new Map();
  if (!await existing(path)) return receipts;
  const bytes = await readFile(path);
  const lastComplete = bytes.lastIndexOf(10) + 1;
  if (lastComplete !== bytes.length) {
    const tail = bytes.subarray(lastComplete);
    const preserved = `${path}.interrupted-tail-${sha(tail)}.bin`;
    if (!await existing(preserved)) {
      const evidence = await open(preserved, 'wx');
      try { await evidence.writeFile(tail); await evidence.sync(); } finally { await evidence.close(); }
    }
    const repair = await open(path, 'r+');
    try { await repair.truncate(lastComplete); await repair.sync(); } finally { await repair.close(); }
    await send({ kind: 'receipt_tail_recovered', truncated_bytes: tail.length, preserved });
  }
  for await (const text of createInterface({ input: createReadStream(path), crlfDelay: Infinity })) {
    const record = JSON.parse(text);
    assert.equal(record.specs_sha256, immutable.specs_sha256, 'Receipt belongs to different frozen specs');
    assert.equal(record.registry_sha256, immutable.registry_sha256);
    assert.ok(!receipts.has(record.prefix_id), 'Duplicate complete prefix receipt');
    const spec = prefixSpecs.find(spec => spec.prefix_id === record.prefix_id);
    assert.ok(spec, 'Unknown frozen prefix receipt');
    assert.equal(record.observations.length, spec.prefix.length, 'Partial prefix stored as complete');
    let before = await api.createSession(registry, { dialogueVersion: 2, sessionId: spec.session_id });
    for (const [index, observed] of record.observations.entries()) {
      validateObservation(observed);
      const question = spec.prefix[index].question;
      assert.equal(observed.question, question);
      const prompt = await renderPrompt(registry, api, before, question);
      assert.equal(observed.model_input_sha256, sha(prompt), 'Stored generation input changed');
      before = await applyObservation(registry, api, before, question, observed);
    }
    assert.deepEqual(before, record.actual_session, 'Stored actual session differs from replayed raw observations');
    receipts.set(record.prefix_id, record);
  }
  return receipts;
}
async function outputRows(path, receipts = null) {
  const file = await open(path, 'wx');
  const controlFile = args['control-output'] ? await open(args['control-output'], 'wx') : null;
  const hash = createHash('sha256');
  const controlHash = createHash('sha256');
  let rows = 0;
  let identityPairs = 0;
  const branches = {}, targetIds = {}, outcomes = {};
  try {
    for (const spec of specs) {
      const receipt = receipts?.get(spec.prefix_id);
      if (receipts) assert.ok(receipt, `Missing actual prefix: ${spec.prefix_id}`);
      const row = await materializeScenario(registry, api, spec, receipt?.observations ?? null);
      row.model_input = await renderPrompt(registry, api, row.before, row.question);
      row.provenance = { ...row.provenance, ...immutable,
        ...(receipt ? { model_fingerprint: receipt.model_fingerprint, observed_receipt_sha256: sha(line(receipt)),
          reconstructed_prefix: false } : {}) };
      const bytes = line(row); await file.writeFile(bytes); hash.update(bytes); rows++;
      if (controlFile) {
        const control = await matchedControl(registry, api, spec, row);
        control.model_input = await renderPrompt(registry, api, control.before, control.question);
        const controlBytes = line(control); await controlFile.writeFile(controlBytes); controlHash.update(controlBytes);
        if (control.match.identity_order_pair) identityPairs++;
      }
      branches[row.branch] = (branches[row.branch] ?? 0) + 1;
      for (const id of row.reference_targets) targetIds[id] = (targetIds[id] ?? 0) + 1;
      for (const exchange of row.before.exchanges) outcomes[exchange.outcome] = (outcomes[exchange.outcome] ?? 0) + 1;
    }
    await file.sync(); if (controlFile) await controlFile.sync();
  } finally { await file.close(); if (controlFile) await controlFile.close(); }
  return { ...immutable, rows, sha256: hash.digest('hex'), branches, target_ids: targetIds,
    ...(controlFile ? { control_sha256: controlHash.digest('hex'), control_rows: rows,
      identity_order_pairs: identityPairs, changed_order_pairs: rows - identityPairs } : {}),
    input_exchange_occurrences: outcomes, actual_observed_rows: receipts ? rows : 0,
    reconstructed_rows: receipts ? 0 : rows, model_output_used_as_gold: false, tokenizer_invoked: false };
}

if (command === 'check') {
  for (const spec of specs) await materializeScenario(registry, api, spec);
  await send({ kind: 'check', ...immutable, cases: specs.length, prefixes: prefixSpecs.length,
    requests: prefixSpecs.reduce((sum, spec) => sum + spec.prefix.length, 0), tokenizer_invoked: false });
} else if (command === 'materialize') {
  assert.ok(args.output, '--output is required');
  await mkdir(dirname(resolve(args.output)), { recursive: true });
  await send({ kind: 'result', ...await outputRows(args.output) });
} else {
  for (const key of ['output', 'receipts', 'model-fingerprint']) assert.ok(args[key], `--${key} is required`);
  const fingerprint = JSON.parse(await readFile(args['model-fingerprint'], 'utf8'));
  assert.ok(fingerprint.checkpoint_sha256 && fingerprint.checkpoint_step === 7205,
    'Collection requires a frozen checkpoint7205 fingerprint');
  const receipts = await readReceipts(args.receipts);
  for (const receipt of receipts.values()) assert.deepEqual(receipt.model_fingerprint, fingerprint, 'Changed collection model');
  await mkdir(dirname(resolve(args.receipts)), { recursive: true });
  const reader = createInterface({ input: process.stdin, crlfDelay: Infinity });
  const replies = reader[Symbol.asyncIterator]();
  try {
    for (const spec of prefixSpecs) {
      if (receipts.has(spec.prefix_id)) continue;
      let before = await api.createSession(registry, { dialogueVersion: 2, sessionId: spec.session_id });
      const observations = [];
      for (const [index, step] of spec.prefix.entries()) {
        const prompt = await renderPrompt(registry, api, before, step.question);
        const caseId = `${spec.prefix_id}-e${index + 1}`;
        await send({ kind: 'generate', case_id: caseId, scenario_id: spec.case_id,
          prefix_id: spec.prefix_id, prefix_index: index, prompt });
        const response = await replies.next();
        assert.ok(!response.done, 'Model provider disconnected');
        const reply = JSON.parse(response.value);
        if (reply.case_id) assert.equal(reply.case_id, caseId);
        assert.equal(typeof reply.raw_text, 'string');
        assert.ok(Array.isArray(reply.raw_ids) && reply.raw_ids.length > 0 && reply.raw_ids.length <= 512);
        assert.ok(reply.raw_ids.every(id => Number.isInteger(id) && id >= 0 && id < 262144));
        const complete = reply.raw_ids.at(-1) === 50;
        assert.ok(!reply.raw_ids.slice(0, -1).includes(50));
        if (!complete) { assert.equal(reply.raw_ids.length, 512); assert.ok(!reply.raw_ids.includes(50)); }
        const observed = { kind: complete ? 'native' : 'incomplete', question: step.question,
          raw_output: reply.raw_text, raw_ids: reply.raw_ids, model_input_sha256: sha(prompt) };
        validateObservation(observed);
        before = await applyObservation(registry, api, before, step.question, observed);
        observations.push(observed);
      }
      const receipt = { ...immutable, prefix_id: spec.prefix_id, model_fingerprint: fingerprint,
        observations, actual_session: before, completed_at: new Date().toISOString() };
      const receiptFile = await open(args.receipts, 'a');
      try { await receiptFile.writeFile(line(receipt)); await receiptFile.sync(); } finally { await receiptFile.close(); }
      receipts.set(spec.prefix_id, receipt);
      await send({ kind: 'prefix_complete', prefix_id: spec.prefix_id, completed: receipts.size, planned: prefixSpecs.length });
    }
    await mkdir(dirname(resolve(args.output)), { recursive: true });
    const result = await outputRows(args.output, receipts);
    await writeFile(`${args.output}.manifest.json`, JSON.stringify({ ...result, model_fingerprint: fingerprint }, null, 2) + '\n', { flag: 'wx' });
    await send({ kind: 'result', ...result });
  } finally { reader.close(); }
}
