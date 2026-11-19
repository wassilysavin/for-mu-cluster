import assert from 'node:assert/strict';
import test from 'node:test';
import { createHash } from 'node:crypto';
import { buildFamilies, buildCompositions, usableAnchor } from './week_compositions.mjs';
import { canonical } from './evaluation/schema.mjs';
import { decodeCalls } from './evaluation/functiongemma.mjs';

test('only independent, labeled TRAIN subjects can be used as constituent questions', () => {
  function row(question, subject = 'TR', extra = {}) {
    return { history_origin: 'fresh', question, calls: [{ name: 'get_default_value', arguments: { subject } }], ...extra };
  }
  assert.equal(usableAnchor(row('What is the default TR value?')), true);
  assert.equal(usableAnchor(row('What is the default TR value?', 'TR', { history_origin: 'derived_history_v2' })), false);
  assert.equal(usableAnchor(row('What is its documented value?')), false);
  assert.equal(usableAnchor(row('What is the documented option for my STRay field?')), false);
  assert.equal(usableAnchor(row('Give only the documented TR default.')), false);
  assert.equal(usableAnchor(row('What is TR?', 'TR', { calls: [{ name: 'get_value_guidance', arguments: { subject: 'TR', _clarify: true } }] })), false);
  assert.equal(usableAnchor(row('How does the endpoint of default_hrf.t change?', 'default_hrf.t',
    { calls: [{ name: 'explain_change_effect', arguments: { subject: 'default_hrf.t', direction: 'increase' } }] })), false);
});

test('source coverage is bounded, relation labels are true and unordered programs unique', async () => {
  const families = await buildFamilies();
  const ids = new Set(), keys = new Set(), exposure = new Map();
  const totals = {};
  for (const family of families) {
    const { family_id, group, members } = family;
    const [a, b, c] = members;
    const key = canonical(members.map(m => canonical(m.call)).sort());
    assert.ok(!ids.has(family_id) && !keys.has(key));
    ids.add(family_id); keys.add(key);
    for (const m of members) exposure.set(m.id, (exposure.get(m.id) ?? 0) + 1);
    const equalsSubject = a.subject === b.subject;
    const equalsOp = a.call.name === b.call.name;
    if (group === 'science_same_subject') assert.ok(equalsSubject && !equalsOp && !c);
    if (group === 'science_same_operation') {
      assert.ok(!equalsSubject && equalsOp && !c);
      assert.deepEqual(a.options, b.options);
    }
    if (group === 'science_argument_contrast') {
      assert.ok(equalsSubject && equalsOp && !c);
      assert.deepEqual(Object.keys(a.options).sort(), Object.keys(b.options).sort());
      assert.equal(Object.keys(a.options).filter(k => canonical(a.options[k]) !== canonical(b.options[k])).length, 1);
    }
    if (group === 'science_three_obligations') {
      assert.ok(equalsSubject && !equalsOp && c && a.subject !== c.subject && a.call.name === c.call.name);
      assert.deepEqual(a.options, c.options);
    }
    totals[group] = (totals[group] ?? 0) + 1;
  }
  assert.deepEqual(totals, { science_argument_contrast: 50, science_same_subject: 112,
    science_same_operation: 112, science_three_obligations: 96 });
  assert.ok(Math.max(...exposure.values()) <= 5);
});

test('rendered TRAIN contains the original ordered clauses and gold only in its target', async () => {
  let pending;
  let rows = 0;
  const digest = createHash('sha256');
  for await (const row of buildCompositions(true)) {
    const q = row.question;
    assert.equal(row.calls.length, row.ancestor_case_ids.length);
    assert.deepEqual(decodeCalls(row.target), row.calls);
    assert.deepEqual(row.prefix, []);
    assert.equal(row.split, 'train');
    assert.ok(row.model_input.includes(JSON.stringify(q)) && row.model_input.endsWith('<start_of_turn>model\n'));
    if (!pending) pending = row;
    else {
      assert.equal(row.family_id, pending.family_id);
      assert.notEqual(row.case_id, pending.case_id);
      assert.deepEqual(row.calls, [...pending.calls].reverse());
      assert.deepEqual(row.ancestor_case_ids, [...pending.ancestor_case_ids].reverse());
      const clauses = text => text.split(/\n\(\d+\) /).slice(1);
      assert.deepEqual(clauses(q), clauses(pending.question).reverse());
      pending = null;
    }
    digest.update(q + '\n' + row.target + '\n');
    rows++;
  }
  assert.equal(digest.digest('hex'), 'c380876984d26142027930589de949371aba4009534ae5d9a4843cb3ec6baf2f');
  assert.equal(rows, 740);
  assert.equal(pending, null);
});
