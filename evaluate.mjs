import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { open, readFile } from 'node:fs/promises';
import { createInterface } from 'node:readline';

import { evaluateDevelopment, validateDevelopmentPlan } from './evaluation/development-evaluator.mjs';
import { buildModelInput, pinnedTokenizerSha256 } from './evaluation/model-input.mjs';
import { IncompleteModelOutput } from './evaluation/model-result.mjs';
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
  return { registry, rows, reviews, spec };
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

async function evaluate(suite, observationsPath) {
  const input = createInterface({ input: process.stdin, crlfDelay: Infinity });
  const replies = input[Symbol.asyncIterator]();
  const observations = await open(observationsPath, 'wx');
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
      onRecord: async record => {
        await observations.appendFile(`${JSON.stringify(record)}\n`);
        await observations.datasync();
      },
      stopOnProviderFailure: true,
    });
    await send({ kind: 'result', score });
    if (score.aborted) process.exitCode = 1;
  } finally {
    input.close();
    await observations.close();
  }
}

const [registryPath, devPath, output, specification] = process.argv.slice(2);
if (!registryPath || !devPath || !output) throw new Error('Expected registry, development data, and observations path or --check');
const suite = await loadSuite(registryPath, devPath, specification ? JSON.parse(specification) : legacy);
if (output === '--check') await send({ kind: 'check', cases: suite.rows.length, history_reviews: suite.reviews.size });
else await evaluate(suite, output);
