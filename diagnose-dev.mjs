import assert from 'node:assert/strict';
import { readFile, open, statfs } from 'node:fs/promises';
import { dirname } from 'node:path';
import { pathToFileURL } from 'node:url';
import { createInterface } from 'node:readline';
import { createRegistry } from './evaluation/registry.mjs';
import { canonical } from './evaluation/schema.mjs';
import { createSession, applyModelOutput, sessionHistory } from './evaluation/session.mjs';
import { encodeCalls } from './evaluation/functiongemma.mjs';
import { buildModelInput, pinnedTokenizerSha256 } from './evaluation/model-input.mjs';
import { createActualHistoryHarness, hash } from './evaluation/actual-history.mjs';
import { materializeGeneration } from './diagnose-data.mjs';

const api = { createSession, applyModelOutput, sessionHistory, encodeCalls, canonical };
const harness = createActualHistoryHarness(api);
const panels = ['fixed_prefix', 'closed_loop', 'challenge'];
const emit = value => new Promise((resolve, reject) => process.stdout.write(
  `${JSON.stringify(value)}\n`, error => error ? reject(error) : resolve()));

export function shardOf(row, count = 4) {
  const unit = `${row.panel}:${row.dialogue_id ? `dialogue:${row.dialogue_id}` : `case:${row.case_id}`}`;
  return Number.parseInt(hash(unit).slice(0, 8), 16) % count;
}

export async function loadPanel(registryPath, bundlePath) {
  const source = JSON.parse(await readFile(registryPath, 'utf8'));
  const registry = createRegistry(source.knowledge, source.contracts);
  assert.equal(await registry.fingerprint(), source.fingerprint);
  const data = JSON.parse(await readFile(bundlePath, 'utf8'));
  assert.equal(data.schema, 'actual-history-v3-dev-1');
  assert.equal(data.registry_sha256, source.fingerprint);
  const bundleHash = hash(canonical({ ...data, review: null }));
  assert.equal(data.review.status, 'approved');
  assert.notEqual(data.review.reviewer, data.author);
  assert.equal(data.review.bundle_sha256, bundleHash);
  const plan = panels.flatMap(panel => data.panels[panel]);
  assert.equal(plan.length, 1250);
  assert.equal(new Set(plan.map(row => row.case_id)).size, plan.length);
  for (const panel of panels) {
    const turns = new Map();
    for (const row of data.panels[panel]) {
      assert.equal(row.panel, panel);
      assert.equal(row.split, 'dev');
      if (row.dialogue_id) {
        assert.equal(row.turn_index, (turns.get(row.dialogue_id) ?? 0) + 1);
        turns.set(row.dialogue_id, row.turn_index);
      }
    }
    for (const count of turns.values()) assert.equal(count, panel === 'challenge' ? 8 : 5);
  }
  return { registry, data, plan, bundleHash };
}

async function promptFor(registry, before, question) {
  const captured = new Error('prompt captured');
  let prompt;
  try {
    await buildModelInput(registry, before, question, { tokenizer: { sha256: pinnedTokenizerSha256,
      encode: async (text, options) => { assert.equal(options.truncation, false); prompt = text; throw captured; } },
    contextWindow: 32768, completionTokens: 512 });
  } catch (error) { if (error !== captured) throw error; }
  assert.equal(typeof prompt, 'string');
  return prompt;
}

async function readSaved(path) {
  let raw;
  try { raw = await readFile(path, 'utf8'); } catch (error) { if (error.code === 'ENOENT') return []; throw error; }
  if (raw && !raw.endsWith('\n')) {
    const boundary = raw.lastIndexOf('\n') + 1, last = raw.slice(boundary);
    let complete = false;
    try { JSON.parse(last); complete = true; } catch {}
    const handle = await open(path, 'r+');
    try {
      if (complete) { await handle.write('\n', Buffer.byteLength(raw), 'utf8'); raw += '\n'; }
      else { raw = raw.slice(0, boundary); await handle.truncate(Buffer.byteLength(raw)); }
      await handle.datasync();
    } finally { await handle.close(); }
  }
  return raw.trim() ? raw.trimEnd().split('\n').map(line => JSON.parse(line)) : [];
}

export function summarize(data, records) {
  const scores = Object.fromEntries(panels.map(panel => [panel,
    harness.summarize(data.panels[panel], records.filter(row => row.panel === panel))]));
  return { panels: scores, summary: scores.closed_loop, observed_records: records.length,
    review_status: 'automatic_conservative_unresolved_retained', release_ready: false };
}

async function evaluate(context, output, shard, count, reserve, binding) {
  const { registry, data, plan, bundleHash } = context;
  const selected = plan.filter(row => shardOf(row, count) === shard);
  const saved = await readSaved(output);
  assert.ok(saved.length <= selected.length, 'Extra saved DEV observations');
  const input = createInterface({ input: process.stdin, crlfDelay: Infinity });
  const replies = input[Symbol.asyncIterator](), handle = await open(output, 'a');
  const sessions = new Map(), records = [];
  try {
    for (let index = 0; index < selected.length; index++) {
      const row = selected[index];
      const before = sessions.get(row.dialogue_id) ?? await harness.initial(registry, row);
      const resolved = harness.resolve(row, before), prompt = await promptFor(registry, before, resolved.question);
      const previous = saved[index];
      let generation;
      if (previous) {
        assert.equal(previous.case_id, row.case_id);
        assert.equal(previous.bundle_sha256, bundleHash);
        assert.equal(previous.prompt_sha256, hash(prompt));
        generation = previous.generation;
      } else {
        await emit({ kind: 'generate', case_id: row.case_id, prompt });
        const next = await replies.next();
        assert.equal(next.done, false, 'Provider disconnected; committed prefix remains resumable');
        generation = JSON.parse(next.value);
      }
      assert.equal(generation.diagnostic_worker_sha256, binding, 'DEV generation belongs to a different provider or worker');
      const after = await materializeGeneration(registry, before, resolved.question, generation);
      const graded = await harness.grade(registry, row, before, after, resolved);
      const record = { ...graded, bundle_sha256: bundleHash, prompt_sha256: hash(prompt),
        before_visible_history_sha256: resolved.before_visible_history_sha256, question: resolved.question,
        policy_branch: resolved.branch, expected_model_call_batches: resolved.allowed_batches,
        generation, before, after };
      if (previous) assert.equal(canonical(previous), canonical(record), 'Saved DEV is not reproducible');
      else {
        const bytes = Buffer.from(`${JSON.stringify(record)}\n`);
        const space = await statfs(dirname(output), { bigint: true });
        assert.ok(space.bavail * space.bsize >= BigInt(reserve) + BigInt(bytes.length), 'Private run disk reserve exhausted');
        await handle.write(bytes); await handle.datasync();
      }
      if (row.dialogue_id) sessions.set(row.dialogue_id, after);
      records.push(record);
    }
    await emit({ kind: 'result', score: { observed_records: records.length, shard, shards: count, bundle_sha256: bundleHash } });
  } finally { input.close(); await handle.close(); }
}

async function main() {
  const [registryPath, bundlePath, output, shardText = '0', countText = '4', reserveText = '536870912', binding] = process.argv.slice(2);
  const context = await loadPanel(registryPath, bundlePath);
  if (output === '--plan') {
    await emit({ kind: 'plan', bundle_sha256: context.bundleHash,
      rows: context.plan.map(row => ({ case_id: row.case_id, panel: row.panel, dialogue_id: row.dialogue_id,
        family_id: row.family_id, cohort: row.cohort, shard: shardOf(row) })) });
    return;
  }
  if (output === '--summarize') {
    const records = await readSaved(shardText);
    assert.equal(records.length, context.plan.length);
    assert.deepEqual(records.map(row => row.case_id), context.plan.map(row => row.case_id));
    assert.ok(records.every(row => row.bundle_sha256 === context.bundleHash));
    await emit({ kind: 'result', score: summarize(context.data, records) });
    return;
  }
  const shard = Number(shardText), count = Number(countText), reserve = Number(reserveText);
  assert.ok(Number.isInteger(shard) && count === 4 && shard >= 0 && shard < count);
  assert.ok(Number.isSafeInteger(reserve) && reserve >= 536870912);
  assert.match(binding ?? '', /^[a-f0-9]{64}$/);
  await evaluate(context, output, shard, count, reserve, binding);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) await main();
