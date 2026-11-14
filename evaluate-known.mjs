import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { open, readFile, statfs } from 'node:fs/promises';
import { dirname } from 'node:path';
import { createInterface } from 'node:readline';

import { evaluateDevelopment, validateDevelopmentPlan } from './evaluation/development-evaluator.mjs';
import { buildModelInput, pinnedTokenizerSha256 } from './evaluation/model-input.mjs';
import { IncompleteModelOutput } from './evaluation/model-result.mjs';
import { applyKnownReview } from './evaluation/known-review.mjs';
import { summarizeEvaluation } from './evaluation/evaluation.mjs';
import { canonical, ContractError } from './evaluation/schema.mjs';
import { createRegistry } from './evaluation/registry.mjs';

const STOP_TOKEN = 50;
const COMPLETION_LIMIT = 512;
const sha = text => createHash('sha256').update(text).digest('hex');
const legacy = { cases: 700, supported: 600, outside_scope: 100, dialogues: 60, dialogue_version: 1 };

function sessionIdFor(id, spec) {
  const suite = spec.dialogue_version === 1 ? 'dev700/session-v1' : `${spec.id}/session-v2`;
  const hex = sha(`gemprf/cluster/${suite}/${id}`);
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-4${hex.slice(13, 16)}-8${hex.slice(17, 20)}-${hex.slice(20, 32)}`;
}

async function loadSuite(registryPath, devPath, spec) {
  assert.ok([1, 2].includes(spec.dialogue_version));
  assert.equal(spec.supported + spec.outside_scope, spec.cases);
  if (spec.dialogue_version === 2) assert.match(spec.id, /^dev-volume-[a-z0-9-]+$/);
  const source = JSON.parse(await readFile(registryPath, 'utf8'));
  const registry = createRegistry(source.knowledge, source.contracts);
  assert.equal(await registry.fingerprint(), source.fingerprint);
  const rows = (await readFile(devPath, 'utf8')).trimEnd().split('\n').map(line => JSON.parse(line));
  assert.equal(rows.length, spec.cases);
  assert.equal(new Set(rows.map(row => row.case_id)).size, rows.length);
  assert.equal(rows.filter(row => row.cohort === 'supported').length, spec.supported);
  assert.equal(rows.filter(row => row.cohort === 'outside_scope').length, spec.outside_scope);
  assert.equal(new Set(rows.flatMap(row => row.dialogue_id ? [row.dialogue_id] : [])).size, spec.dialogues);
  await validateDevelopmentPlan(registry, rows, { split: 'dev', dialogueVersion: spec.dialogue_version });
  const reviews = new Map();
  for (const verdict of spec.dialogue_version === 1 ? source.actual_history_reviews : []) {
    assert.equal(verdict.registry_sha256, source.fingerprint);
    assert.equal(verdict.independence.current_question_candidate_output_read, false);
    const key = `${verdict.case_id}:${verdict.before_visible_history_sha256}`;
    assert.ok(!reviews.has(key));
    reviews.set(key, verdict);
  }
  const policy = JSON.parse(await readFile(new URL('./assets/history-known-review.json', import.meta.url), 'utf8'));
  assert.equal(policy.registry_sha256, source.fingerprint);
  return { registry, rows, reviews, spec, policy };
}

async function renderPrompt(registry, session, question) {
  const captured = new Error('Prompt captured');
  let prompt;
  try {
    await buildModelInput(registry, session, question, {
      tokenizer: { sha256: pinnedTokenizerSha256, encode: async (text, options) => {
        assert.deepEqual(options, { add_special_tokens: false, truncation: false });
        prompt = text;
        throw captured;
      } },
      contextWindow: 32768,
      completionTokens: COMPLETION_LIMIT,
    });
  } catch (error) {
    if (error !== captured) throw error;
  }
  assert.ok(prompt);
  return prompt;
}

async function send(message) {
  await new Promise((resolve, reject) => {
    process.stdout.write(`${JSON.stringify(message)}\n`, error => error ? reject(error) : resolve());
  });
}

async function evaluate(suite, observationsPath, diskReserve) {
  const input = createInterface({ input: process.stdin, crlfDelay: Infinity });
  const replies = input[Symbol.asyncIterator]();
  let prior = [];
  try {
    const bytes = await readFile(observationsPath);
    const end = bytes.lastIndexOf(10) + 1;
    prior = bytes.subarray(0, end).toString().split('\n').filter(Boolean).map(JSON.parse);
    const handle = await open(observationsPath, 'r+');
    try { if (end !== bytes.length) { await handle.truncate(end); await handle.datasync(); } }
    finally { await handle.close(); }
  } catch (error) { if (error.code !== 'ENOENT') throw error; }
  assert.ok(prior.length <= suite.rows.length, 'Saved known DEV contains more records than its frozen plan');
  prior.forEach((record, index) => assert.equal(record.case_id, suite.rows[index].case_id, 'Saved known DEV order differs'));
  const observations = await open(observationsPath, 'a');
  let replayIndex = 0;
  const reviewedResults = [];
  const byCase = new Map(suite.rows.map(row => [row.case_id, row]));
  try {
    const score = await evaluateDevelopment(suite.registry, suite.rows, {
      split: 'dev',
      dialogueVersion: suite.spec.dialogue_version,
      sessionIdFor: id => sessionIdFor(id, suite.spec),
      adjudicate: async request => {
        const verdict = suite.reviews.get(`${request.case_id}:${request.before_visible_history_sha256}`);
        if (!verdict) return { decision: 'review' };
        assert.equal(verdict.question_sha256, sha(request.question));
        return verdict;
      },
      provider: async (request, metadata) => {
        const previous = prior[replayIndex];
        if (previous) {
          assert.equal(previous.case_id, metadata.case_id);
          assert.equal(canonical(previous.before), canonical(request.session));
          const exchange = previous.after.exchanges.at(-1);
          if (previous.provider_failure) {
            if (exchange.error_code === 'context_overflow') throw new ContractError('context_overflow', 'Retained context overflow');
            const failure = new Error('Retained infrastructure failure');
            failure.code = previous.provider_failure;
            throw failure;
          }
          if (exchange.generation?.complete_native_batch === false) {
            throw new IncompleteModelOutput(exchange.raw_output, exchange.generation.token_ids, 'token_limit');
          }
          return exchange.raw_output;
        }
        const prompt = await renderPrompt(suite.registry, request.session, request.question);
        await send({ kind: 'generate', case_id: metadata.case_id, prompt });
        const reply = await replies.next();
        if (reply.done) throw new Error('Model provider disconnected');
        const { raw_text, raw_ids } = JSON.parse(reply.value);
        assert.equal(typeof raw_text, 'string');
        assert.ok(Array.isArray(raw_ids) && raw_ids.length <= COMPLETION_LIMIT);
        assert.ok(raw_ids.every(id => Number.isInteger(id) && id >= 0 && id < 262144));
        if (raw_ids.at(-1) !== STOP_TOKEN) {
          assert.equal(raw_ids.length, COMPLETION_LIMIT);
          assert.ok(!raw_ids.includes(STOP_TOKEN));
          throw new IncompleteModelOutput(raw_text, raw_ids, 'token_limit');
        }
        assert.ok(!raw_ids.slice(0, -1).includes(STOP_TOKEN));
        return raw_text;
      },
      onRecord: async original => {
        const record = await applyKnownReview(suite.registry, byCase.get(original.case_id), original, suite.policy);
        reviewedResults.push(record.result);
        const previous = prior[replayIndex++];
        if (previous) { assert.equal(canonical(record), canonical(previous), 'Resumed known DEV prefix differs'); return; }
        const line = Buffer.from(`${JSON.stringify(record)}\n`);
        if (diskReserve > 0n) {
          const { bavail, bsize } = await statfs(dirname(observationsPath), { bigint: true });
          if (bavail * bsize < diskReserve + BigInt(line.length)) {
            throw new Error('Run filesystem lost its free-space reserve during DEV evaluation');
          }
        }
        await observations.appendFile(line);
        await observations.datasync();
        if (record.provider_failure) throw new Error('Provider failed; its failed attempt was saved. Resume continues from that actual state.');
      },
      stopOnProviderFailure: false,
    });
    const providerFailures = prior.filter(record => record.provider_failure && record.provider_failure !== 'context_overflow').length;
    await send({ kind: 'result', score: { ...score, results: reviewedResults, summary: summarizeEvaluation(suite.rows, reviewedResults),
      provider_failures: providerFailures, quality_valid: providerFailures === 0 } });
    if (score.aborted) process.exitCode = 1;
  } finally {
    input.close();
    await observations.close();
  }
}

const [registryPath, devPath, output, specification, diskReserve = '0'] = process.argv.slice(2);
if (!registryPath || !devPath || !output) throw new Error('Expected registry, development data, and observations path or --check');
const suite = await loadSuite(registryPath, devPath, specification ? JSON.parse(specification) : legacy);
if (output === '--check') await send({ kind: 'check', cases: suite.rows.length, history_reviews: suite.reviews.size });
else await evaluate(suite, output, BigInt(diskReserve));
