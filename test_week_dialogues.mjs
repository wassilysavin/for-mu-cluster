import assert from 'node:assert/strict';
import { before, test } from 'node:test';
import { buildWeekDialogues } from './week_dialogues.mjs';
import { decodeCalls } from './evaluation/functiongemma.mjs';

const records = [];
before(async () => {
  await buildWeekDialogues({ emit: row => records.push(row) });
});

test('all continuation targets are native programs with named TRAIN ancestors and diverse actual histories', () => {
  assert.equal(records.length, new Set(records.map(row => row.case_id)).size);
  assert.equal(new Set(records.map(row => row.family_id)).size, 384);
  assert.equal(records.length, 1472);
  assert.equal(new Set(records.map(row => row.group)).size, 14);
  const ancestors = new Set();
  for (const row of records) {
    assert.equal(row.split, 'train');
    assert.equal(row.history_origin, 'authored_executed_prefix');
    assert.ok(row.prefix.length >= 1 && row.prefix.length <= 6);
    assert.deepEqual(decodeCalls(row.target), row.calls);
    for (const id of row.ancestor_case_ids) {
      assert.match(id, /^train-intent-\d+$/);
      ancestors.add(id);
    }
  }
  assert.ok(ancestors.size >= 400);
});

test('identical requests after failure and success have different native programs', () => {
  const retryFamilies = new Set(records.filter(r => r.group === 'retry_failure').map(r => r.family_id));
  assert.equal(retryFamilies.size, 64);
  for (const family of retryFamilies) {
    const rows = records.filter(r => r.family_id === family);
    const get = group => rows.find(r => r.group === group);
    const retry = get('retry_failure'), fresh = get('recompute_success');
    assert.equal(retry.question, fresh.question);
    assert.equal(retry.calls[0].name, 'retry_exchange');
    assert.deepEqual(retry.calls.slice(1), fresh.calls);
    const index = Number(retry.calls[0].arguments.target.slice(1)) - 1;
    assert.ok(['provider_error', 'malformed'].includes(retry.prefix[index].kind));
    assert.equal(fresh.prefix[index].kind, 'gold');
    assert.equal(retry.prefix[index].question, fresh.prefix[index].question);
    assert.equal(get('missing_answer').question, get('single_part').question);
    assert.equal(get('missing_answer').calls[0].name, 'clarify_reference');
    assert.equal(get('single_part').calls[0].name, 'repeat_answer');
    assert.deepEqual(get('quote_failure').calls, get('whole_message').calls);
    assert.deepEqual(get('recovered_answer').prefix.at(-1).calls, retry.calls);
  }
});

test('second-stage references point to the recorded answer occurrence', () => {
  const split = new Map();
  for (const row of records) {
    if (!['original_or_corrected', 'same_text_answer_address', 'clarified_reference'].includes(row.group)) continue;
    const key = row.group + '/' + row.family_id;
    const ids = split.get(key) ?? [];
    ids.push(row.calls[0].arguments.target);
    split.set(key, ids);
    assert.equal(row.calls[0].name, 'repeat_answer');
    assert.match(ids.at(-1), /^t\d+\.c1$/);
    if (row.group === 'clarified_reference') assert.equal(row.prefix.at(-1).calls[0].name, 'clarify_reference');
    if (row.group === 'original_or_corrected') assert.ok(row.prefix.at(-1).calls[0].name.startsWith('change_'));
  }
  assert.equal(split.size, 192);
  for (const ids of split.values()) {
    assert.equal(ids.length, 2);
    assert.equal(new Set(ids).size, 2);
  }
});
