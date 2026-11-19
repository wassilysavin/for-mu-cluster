/** Replay a TRAIN target on the original and revised registries with its recorded history. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { createInterface } from 'node:readline';
import { pathToFileURL } from 'node:url';
import { createScorer } from './new_train_fit.mjs';
import { canonical } from './evaluation/schema.mjs';
import { applyModelOutput, createSession } from './evaluation/session.mjs';

const assets = new URL('./assets/', import.meta.url);
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
const registryHash = 'b006fbacf4fd0f7f423b9178c7f365e98a6cc91bc99d8c9a16277bfd5954d4d0';
const contextManifestHash = 'b181085faba8cdb73bcf602e83c0eb0842f52496b4d85b29f9e584804add8aa3';
const trainHash = '024d21fba1d2fe73dc9f8d795552303acaeb099675d9d78df179c082c7e7afcb';
const userTurn = '<end_of_turn>\n<start_of_turn>user\n';
const modelTurn = '<end_of_turn>\n<start_of_turn>model\n';

async function readJson(name, expected) {
  const bytes = await readFile(new URL(name, assets));
  assert.equal(sha(bytes), expected, 'Frozen TRAIN asset changed: ' + name);
  return JSON.parse(bytes);
}

async function readIndex(name, expected, project) {
  const map = new Map(), hash = createHash('sha256');
  const stream = createReadStream(new URL(name, assets));
  stream.on('data', bytes => hash.update(bytes));
  try {
    for await (const line of createInterface({ input: stream, crlfDelay: Infinity })) {
      const row = JSON.parse(line);
      assert.ok(!map.has(row.case_id), 'Repeated frozen TRAIN ID: ' + row.case_id);
      map.set(row.case_id, project(row));
    }
  } finally { stream.destroy(); }
  assert.equal(hash.digest('hex'), expected, 'Frozen TRAIN asset changed: ' + name);
  return map;
}

export async function verifier() {
  const contextsManifest = await readJson('diagnostic-train-contexts.json', contextManifestHash);
  assert.equal(contextsManifest.split, 'train');
  assert.equal(contextsManifest.source_dataset.sha256, trainHash);
  const [registry, contexts, source] = await Promise.all([
    readJson('registry.json', registryHash),
    readIndex('diagnostic-train-contexts.jsonl', contextsManifest.contexts.sha256, r => r),
    readIndex('train-full-v2.jsonl', trainHash, ({ question, calls }) => ({ question, calls })),
  ]);
  const scorer = await createScorer(registry);
  const registryId = await scorer.registry.fingerprint();
  return async function verify({ case_id, source_case_id, model_input, target, calls, question }) {
    assert.equal(case_id, 'week-replay:' + source_case_id);
    assert.deepEqual(source.get(source_case_id), { question, calls }, 'TRAIN source question/target mismatch: ' + case_id);
    const user = model_input.split(userTurn);
    assert.ok(user.length === 2 && user[1].endsWith(modelTurn), 'Incomplete TRAIN prompt: ' + case_id);
    const data = JSON.parse(user[1].slice(0, -modelTurn.length));
    assert.equal(data.question, question);
    const saved = contexts.get(source_case_id);
    assert.equal(Boolean(saved), Boolean(data.history.exchanges.length));
    const before = saved?.before ?? await createSession(scorer.original, {
      sessionId: data.history.session_id, dialogueVersion: data.format.endsWith('v2') ? 2 : 1,
    });
    const proof = await scorer.prepare({ case_id, prompt: model_input, target },
      { case_id, question, calls, group: 'shared_replay' }, before, { section: 'shared', migrated: true, branch: 'frozen_train' });
    const old = await applyModelOutput(scorer.original, before, question, target);
    const now = await applyModelOutput(scorer.registry, { ...before, registry_id: registryId }, question, target);
    assert.equal(now.exchanges.at(-1).outcome, 'committed', 'Revised gold failed: ' + case_id);
    assert.deepEqual(now.exchanges, old.exchanges, 'Displayed gold or recorded exchanges changed: ' + case_id);
    assert.deepEqual(now.semantic, old.semantic, 'Gold dialogue state changed: ' + case_id);
    if (saved) {
      assert.equal(sha(canonical(before)), saved.before_sha256);
      assert.equal(sha(canonical(now.semantic)), saved.expected_after_semantic_sha256);
      assert.equal(sha(canonical(now.exchanges)), saved.expected_after_exchanges_sha256);
    }
    const { session_id, ...visible } = data.history;
    return { kind: 'verified', case_id, source_case_id, prompt_sha256: proof.prompt_sha256,
      target_sha256: proof.target_sha256, visible_sha256: sha(canonical({ ...data, history: visible })), group: proof.group };
  };
}

async function main() {
  const verify = await verifier();
  const seen = new Set();
  for await (const line of createInterface({ input: process.stdin, crlfDelay: Infinity })) {
    const request = JSON.parse(line);
    assert.ok(!seen.has(request.case_id), 'Duplicate TRAIN replay request: ' + request.case_id);
    const checked = await verify(request);
    seen.add(request.case_id);
    await new Promise((done, fail) => process.stdout.write(JSON.stringify(checked) + '\n', e => e ? fail(e) : done()));
  }
  console.log(JSON.stringify({ kind: 'complete', cases: seen.size }));
}

if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) await main();
