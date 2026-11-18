/** Materialize TRAIN decisions against the displayed history and versioned runtime. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { createInterface } from 'node:readline';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';
import * as session from './evaluation/session.mjs';
import * as native from './evaluation/functiongemma.mjs';
import * as modelInput from './evaluation/model-input.mjs';
import { createRegistry } from './evaluation/registry.mjs';
import { continuation, applyObservation, renderPrompt } from './evaluation/history-train.mjs';
import { transferRegistry } from './transfer_contract.mjs';

const api = { ...session, ...native, ...modelInput };
const assets = fileURLToPath(new URL('./assets/', import.meta.url));
const sha = value => createHash('sha256').update(value).digest('hex');
const read = async name => JSON.parse(await readFile(resolve(assets, name), 'utf8'));
const lines = async name => (await readFile(resolve(assets, name), 'utf8')).trimEnd().split('\n').map(JSON.parse);

async function* readLines(path) {
  for await (const line of createInterface({input: createReadStream(path), crlfDelay: Infinity})) {
    yield JSON.parse(line);
  }
}

export async function observedSession(registry, original, record, root, spec, specsDigest) {
  assert.equal(record.specs_sha256, specsDigest);
  assert.equal(record.registry_sha256, await original.fingerprint());
  assert.equal(record.model_fingerprint.checkpoint_step, 7205);
  assert.equal(record.observations.length, root.prefix.length);
  assert.deepEqual(root.prefix.map(s => s.question), spec.prefix.map(s => s.question));
  let before = await api.createSession(original, {dialogueVersion: 2, sessionId: spec.session_id});
  for (const [i, item] of record.observations.entries()) {
    const question = root.prefix[i].question;
    assert.equal(item.question, question);
    const prompt = await renderPrompt(original, api, before, question);
    assert.equal(item.model_input_sha256, sha(prompt));
    before = await applyObservation(original, api, before, question, item);
  }
  assert.deepEqual(before, record.actual_session, 'Saved TRAIN dialogue differs from original native replay');
  // The user saw these original replies. Preserve them verbatim across the schema revision.
  return { ...before, registry_id: await registry.fingerprint() };
}

async function authoredSession(registry, root) {
  let before = await api.createSession(registry, {dialogueVersion: 2, sessionId: root.root_id});
  for (const step of root.prefix) {
    if (step.kind === 'provider_error') {
      before = await api.recordProviderFailure(registry, before, step.question, step.error_code);
    } else {
      const raw = step.kind === 'malformed' ? step.raw_output
        : api.encodeCalls(step.policy ? continuation(before, step.policy).calls : step.calls);
      before = await api.applyModelOutput(registry, before, step.question, raw);
      assert.equal(before.exchanges.at(-1).outcome, step.kind === 'malformed' ? 'error' : 'committed');
    }
  }
  return before;
}

function checkMeaning(policy, before, calls) {
  const e = before.exchanges[policy.exchange_index - 1];
  if (policy.kind === 'fresh') assert.deepEqual(calls, policy.calls);
  if (policy.kind === 'clarify_reference') {
    assert.ok(before.exchanges.filter(item => item.outcome === 'committed').length >= 2);
    assert.deepEqual(calls, [{name: 'clarify_reference', arguments: {kind: 'answer'}}]);
  }
  if (policy.kind === 'answer_again') {
    const failed = ['error', 'provider_error'].includes(e.outcome);
    const expected = failed ? [{name: 'retry_exchange', arguments: {target: e.id}}, ...policy.calls] : policy.calls;
    assert.deepEqual(calls, expected);
  }
  if (policy.kind === 'revise_answer') {
    const answer = before.semantic.turns.find(turn => turn.id === e.semantic_turn_id)?.answers[policy.part - 1];
    assert.ok(answer, 'Named answer to revise must exist in this authored scenario');
    assert.deepEqual(answer.call, policy.expected_original);
    assert.deepEqual(calls, [{name: `change_${answer.call.name}`, arguments: {target: answer.id, changes: policy.changes}}]);
    assert.deepEqual({...answer.call.arguments, ...policy.changes}, policy.replacement.arguments);
  }
  if (policy.kind === 'quote_exchange') assert.deepEqual(calls, [{name: 'repeat_exchange', arguments: {target: e.id}}]);
  if (policy.kind === 'quote_answer') {
    const answer = before.semantic.turns.find(t => t.id === e.semantic_turn_id)?.answers[policy.part - 1];
    assert.deepEqual(calls, [answer ? {name: 'repeat_answer', arguments: {target: answer.id}}
      : {name: 'clarify_reference', arguments: {kind: 'answer'}}]);
  }
}

export async function materialize({receipts, fixture = false, emit = async () => {}}) {
  const source = await read('registry.json');
  const registry = await transferRegistry(source);
  const original = createRegistry(source.knowledge, source.contracts);
  const roots = await lines('transfer-train-roots.jsonl');
  const observedIds = new Set(roots.map(r => r.observed_prefix_id).filter(Boolean));
  const specs = await lines('history-train-specs.jsonl');
  const byPrefix = new Map(specs.map(row => [row.prefix_id, row]));
  const digest = sha(await readFile(resolve(assets, 'history-train-specs.jsonl')));
  const selected = new Map();
  if (receipts) {
    for await (const record of readLines(receipts)) {
      if (observedIds.has(record.prefix_id)) {
        assert.ok(!selected.has(record.prefix_id));
        selected.set(record.prefix_id, record);
      }
    }
    assert.equal(selected.size, observedIds.size);
  } else assert.ok(fixture, 'The real TRAIN requires the original 7205 prefix receipts');
  const prefixes = new Map();
  for (const r of roots) {
    const spec = byPrefix.get(r.observed_prefix_id);
    if (r.observed_prefix_id) assert.ok(spec);
    const before = spec && receipts
      ? await observedSession(registry, original, selected.get(r.observed_prefix_id), r, spec, digest)
      : await authoredSession(registry, spec ? {...r, prefix: spec.prefix} : r);
    prefixes.set(r.root_id, {before, group: r.group, observed: Boolean(spec && receipts)});
  }
  for (const arm of ['ordinary', 'contrast']) {
    for (const row of await lines(`transfer-train-${arm}.jsonl`)) {
      const {before, group, observed} = prefixes.get(row.root_id);
      const {question, calls, branch} = continuation(before, row.continuation);
      checkMeaning(row.continuation, before, calls);
      const target = api.encodeCalls(calls);
      const after = await api.applyModelOutput(registry, before, question, target);
      assert.equal(after.exchanges.at(-1).outcome, 'committed', `${row.case_id}: ${after.exchanges.at(-1).error_code}`);
      assert.deepEqual(after.exchanges.slice(0, -1), before.exchanges);
      await emit({arm, case_id: row.case_id, root_id: row.root_id, group, question, calls, branch, observed,
                  model_input: await renderPrompt(registry, api, before, question), target});
    }
  }
}

if (fileURLToPath(import.meta.url) === resolve(process.argv[1] || '.')) {
  const fixture = process.argv[2] === '--fixture';
  assert.ok(fixture || process.argv.length === 3);
  await materialize({fixture, receipts: fixture ? null : process.argv[2], emit: row =>
    new Promise((yes, no) => process.stdout.write(JSON.stringify(row) + '\n', err => err ? no(err) : yes()))});
}
