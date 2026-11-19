import assert from 'node:assert/strict';
import { createReadStream } from 'node:fs';
import { createInterface } from 'node:readline';
import test from 'node:test';
import { decodeCalls } from './evaluation/functiongemma.mjs';
import { buildExtendedScope } from './extended_scope.mjs';

test('mixed-scope TRAIN preserves ancestry, user-visible history and distinct answers after failure', async () => {
  const source = new Set(), cases = new Set(), families = new Map();
  const asset = new URL('./assets/train-full-v2.jsonl', import.meta.url);
  for await (const line of createInterface({ input: createReadStream(asset), crlfDelay: Infinity })) {
    source.add(JSON.parse(line).case_id);
  }
  const user = '<end_of_turn>\n<start_of_turn>user\n', model = '<end_of_turn>\n<start_of_turn>model\n';
  const operations = new Set(), scientificSubjects = new Set(), outsideSources = new Set(), failures = new Set();
  const order = new Map();
  const report = await buildExtendedScope({ render: true, emit: row => {
    const [contract, payload, extra] = row.model_input.split(user);
    assert.ok(contract && payload && !extra && payload.endsWith(model));
    const visible = JSON.parse(payload.slice(0, -model.length));
    assert.equal(visible.question, row.question);
    assert.deepEqual(visible.history.exchanges.map(e => e.user), row.prefix.map(p => p.question));
    assert.equal(row.history_origin, row.prefix.length ? 'authored_executed_prefix' : 'fresh');
    assert.deepEqual(decodeCalls(row.target), row.calls);
    assert.ok(row.ancestor_case_ids.every(id => source.has(id)));
    assert.ok(row.prefix.flatMap(p => p.ancestor_case_ids).every(id => row.ancestor_case_ids.includes(id)));
    assert.ok(!cases.has(row.case_id));
    cases.add(row.case_id);
    const summary = { calls: row.calls, question: row.question, history: visible.history };
    if (!families.has(row.family_id)) families.set(row.family_id, new Map());
    const family = families.get(row.family_id);
    assert.ok(!family.has(row.group));
    family.set(row.group, summary);
    if (row.group === 'scope_mixed_parts') {
      const scientific = row.calls.find(c => c.name !== 'decline_request');
      operations.add(scientific.name);
      scientificSubjects.add(scientific.arguments.subject);
      outsideSources.add(row.ancestor_case_ids.find(id => id.startsWith('train-outside-')));
    }
    if (row.group === 'scope_retry_full') failures.add(visible.history.exchanges.at(-1).outcome);
    if (['scope_replace_science', 'scope_keep_outside'].includes(row.group)) {
      if (!order.has(row.group)) order.set(row.group, new Set());
      order.get(row.group).add(family.get('scope_mixed_parts').calls[0].name === 'decline_request');
    }
  } });
  assert.equal(report.cases, cases.size);
  assert.equal(report.families, families.size);
  assert.equal(outsideSources.size, 200);
  assert.ok(operations.size >= 40 && scientificSubjects.size >= 50);
  assert.deepEqual(failures, new Set(['provider_error', 'error']));
  for (const seen of order.values()) assert.deepEqual(seen, new Set([true, false]));
  for (const family of families.values()) {
    if (family.has('scope_mixed_parts')) {
      const original = family.get('scope_mixed_parts');
      for (const [normal, retry] of [['scope_fresh_full', 'scope_retry_full'], ['scope_retract_outside', 'scope_retry_subset']]) {
        const done = family.get(normal), failed = family.get(retry);
        assert.equal(done.question, failed.question);
        assert.equal(done.history.exchanges.at(-1).user, original.question);
        assert.equal(failed.history.exchanges.at(-1).user, original.question);
        assert.equal(done.history.exchanges.at(-1).outcome, 'committed');
        assert.deepEqual(failed.calls.slice(1), done.calls);
        assert.deepEqual(failed.calls[0], { name: 'retry_exchange', arguments: { target: failed.history.exchanges.at(-1).id } });
      }
    } else {
      const open = family.get('scope_keep_pending').history.pending;
      assert.equal(open.length, 1);
      const resolve = family.get('scope_resolve_only'), cancel = family.get('scope_cancel_then_decline');
      assert.deepEqual(resolve.history.pending, open);
      assert.deepEqual(cancel.history.pending, open);
      assert.equal(resolve.calls[0].arguments._resolve, open[0].id);
      assert.equal(cancel.calls[0].arguments.target, open[0].id);
    }
  }
});
