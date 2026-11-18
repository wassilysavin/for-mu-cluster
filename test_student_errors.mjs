import assert from 'node:assert/strict';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { loadPanel } from './diagnose-dev.mjs';
import { createActualHistoryHarness, hash } from './evaluation/actual-history.mjs';
import { encodeCalls } from './evaluation/functiongemma.mjs';
import { renderPrompt } from './evaluation/history-train.mjs';
import { buildModelInput, pinnedTokenizerSha256 } from './evaluation/model-input.mjs';
import { canonical } from './evaluation/schema.mjs';
import { applyModelOutput, createSession, recordIncompleteModelOutput, sessionHistory } from './evaluation/session.mjs';
import { comparePrograms, inspectArm } from './student_errors.mjs';

const root = fileURLToPath(new URL('./', import.meta.url));
const api = { applyModelOutput, createSession, sessionHistory, encodeCalls, canonical, buildModelInput, pinnedTokenizerSha256 };
const harness = createActualHistoryHarness(api);
const call = (name, args = {}) => ({ name, arguments: args });

test('a valid alternative preserves exact matches and duplicate calls count separately', () => {
  const a = call('get_units', { subject: 'width' }), b = call('explain_purpose', { subject: 'width' });
  assert.equal(comparePrograms([a], [[a, b], [a]]).comparison, 'exact_allowed');
  assert.ok(comparePrograms([a], [[a, a]]).signals.includes('fewer_calls'));
  assert.ok(comparePrograms([a, a], [[a]]).signals.includes('extra_calls'));
  const result = comparePrograms([b, a], [[a, b]]);
  assert.ok(result.signals.includes('call_order'));
  assert.ok(!result.signals.includes('operation_substitution'));
  const first = call('repeat_answer', { target: 't1.c1' });
  const second = call('repeat_answer', { target: 't1.c2' });
  assert.deepEqual(comparePrograms([second, first], [[first, second]]).signals, ['call_order']);
});

test('alternative-dependent signals never become definitive omissions or wrong roles', () => {
  const a = call('get_units', { subject: 'width' });
  const b = call('get_default_value', { subject: 'width' });
  const c = call('get_units', { subject: 'height' });
  const result = comparePrograms([a], [[c], [b]]);
  assert.equal(result.comparison, 'alternative_dependent');
  assert.ok(!result.signals.includes('argument_value'));
  assert.ok(!result.signals.includes('operation_substitution'));
});

test('argument analysis retains true, 1, null and absent as distinct values', () => {
  const want = call('op', { condition: false, retained: null });
  const got = call('op', { condition: 0, invented: null });
  const result = comparePrograms([got], [[want]]);
  assert.ok(result.signals.includes('missing_argument'));
  assert.ok(result.signals.includes('extra_argument'));
  assert.ok(result.signals.some(key => key === 'argument_type' || key === 'argument_value'));
  assert.ok(!result.signals.includes('operation_substitution'));
});

test('read-only replay checks a full dialogue against its own imperfect prior answers', async () => {
  const source = await loadPanel(join(root, 'assets/registry.json'), join(root, 'assets/actual-history-v3-dev.json'));
  const plan = source.data.panels.closed_loop.filter(row => row.dialogue_id === 'dev-ah3-loop-quote_displayed-01');
  const ctx = { ...source, plan, data: { ...source.data, panels: { fixed_prefix: [], closed_loop: plan, challenge: [] } } };
  assert.equal(plan.length, 5);
  const full = [];
  let before = await harness.initial(ctx.registry, plan[0]);
  for (const [index, row] of plan.entries()) {
    const resolved = harness.resolve(row, before);
    const prompt = await renderPrompt(ctx.registry, api, before, resolved.question);
    const raw = index === 1 ? 'synthetic incomplete native call' : encodeCalls(resolved.allowed_batches[0]);
    const ids = index === 1 ? Array(512).fill(1) : [50];
    const generation = { raw_ids: ids, raw_text: raw };
    const after = index === 1
      ? await recordIncompleteModelOutput(ctx.registry, before, resolved.question, raw, ids, 'token_limit')
      : await applyModelOutput(ctx.registry, before, resolved.question, raw);
    const scored = await harness.grade(ctx.registry, row, before, after, resolved);
    full.push({ ...scored, bundle_sha256: ctx.bundleHash, prompt_sha256: hash(prompt),
      before_visible_history_sha256: resolved.before_visible_history_sha256, question: resolved.question,
      policy_branch: resolved.branch, expected_model_call_batches: resolved.allowed_batches, before, after, generation });
    before = after;
  }
  assert.equal(full[1].failure_category, 'execution_error');
  assert.equal(full[2].before.exchanges[1].error_code, 'incomplete_model_output');
  assert.deepEqual(full[2].expected_reference_targets, ['e2']);

  const temp = join(root, '.runtime/tmp');
  await mkdir(temp, { recursive: true });
  const dir = await mkdtemp(join(temp, 'student-inspector-'));
  try {
    const path = join(dir, 'observations.jsonl');
    const write = async rows => {
      const bytes = rows.map(row => JSON.stringify(row)).join('\n') + '\n';
      await writeFile(path, bytes);
      return bytes;
    };
    const initial = await write(full);
    const report = await inspectArm(ctx, path);
    assert.equal(report.score.panels.closed_loop.supported.passed, 4);
    assert.equal(report.score.panels.closed_loop.whole_dialogues.passed, 0);
    assert.equal(report.rows.length, plan.length);
    assert.equal(await readFile(path, 'utf8'), initial);
    for (const mutate of [
      rows => { rows[1].success = true; }, rows => { rows[2].expected_model_call_batches = [[]]; },
      rows => { rows[2].before = rows[1].before; }, rows => { rows[1].generation.raw_ids = [50]; },
      rows => { rows.pop(); }, rows => { rows[4] = rows[3]; },
    ]) {
      const broken = structuredClone(full);
      mutate(broken);
      const bytes = await write(broken);
      await assert.rejects(inspectArm(ctx, path));
      assert.equal(await readFile(path, 'utf8'), bytes);
    }
  } finally { await rm(dir, { force: true, recursive: true }); }
});
