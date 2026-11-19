import assert from 'node:assert/strict';
import { createReadStream } from 'node:fs';
import { createInterface } from 'node:readline';
import test from 'node:test';
import { decodeCalls } from './evaluation/functiongemma.mjs';
import { canonical } from './evaluation/schema.mjs';
import { directQuestions, expandedScience, relation, selectQuestion } from './extended_conditions.mjs';

const call = (name, args) => ({ name, arguments: args });

test('phrasing preserves condition presence, parsed booleans, case-sensitive literals, and bound direction', () => {
  const advice = args => directQuestions(call('get_value_guidance', args));
  assert.ok(advice({ subject:'num_sigmas' }).every(q => !q.includes('refinement')));
  for (const value of [true, false]) {
    const qs = advice({ subject:'num_sigmas', refinement_enabled:value });
    assert.equal(qs.length, 2);
    assert.ok(qs.every(q => q.includes('refinement is ' + (value ? 'enabled' : 'disabled'))));
  }
  assert.deepEqual(advice({ subject:'TR', invented:true }), []);
  const xml = directQuestions(call('explain_flag_literal_effect', {subject:'BIDS.enable', value:'False'}));
  assert.ok(xml.length && xml.every(q => q.includes('"False"') && q.includes('XML')));
  const lower = directQuestions(call('explain_flag_literal_effect', {subject:'BIDS.enable', value:'true'}));
  assert.ok(lower.every(q => q.includes('"true"') && !q.includes('the capital letter')));
  const ops = ['minimum','maximum'].map(bound => directQuestions(call('explain_item_count_limits', {subject:'concatenate_item',bound})));
  assert.ok(ops[0][0].includes('lower') && ops[1][0].includes('upper'));
});

test('the version default asks for the value required by this revision', () => {
  const version = directQuestions(call('get_default_value', {subject:'version'}));
  assert.equal(version.length, 2);
  for (const q of version) {
    assert.match(q, /default/);
    assert.match(q, /this (?:GEM-pRF )?revision/);
    assert.doesNotMatch(q, /fallback|example configuration|omitted/);
  }
  assert.match(directQuestions(call('get_default_value', {subject:'TR'}))[0], /example configuration value from a fallback/);
});

test('the six scopes keep scientific options in fixed order when the requested option changes', () => {
  for (let i=0; i<6; i++) {
    const rows=[0,1].map(p => selectQuestion('First source wording?', 'Second source wording?', p, i));
    assert.notEqual(...rows);
    for (const q of rows) assert.ok(q.indexOf('First source wording?') < q.indexOf('Second source wording?'));
  }
});

function explicitPick(question) {
  if (question.startsWith('I wrote down two questions.')) return +question.includes('The second one is still open;');
  if (question.startsWith('Draft A:')) return +question.endsWith('draft B alone.');
  if (question.startsWith('For orientation,')) return +question.endsWith('the second question answered.');
  if (question.startsWith("I'm listing both notes")) return +question.split('\n')[1].startsWith('Resolved:');
  if (question.startsWith('My colleague and I')) return +question.includes('things. Their note reads');
  assert.ok(question.startsWith('These were the two lines in my draft:'));
  return +question.includes('Cross out line 1.');
}

test('each scope pair really selects its requested frozen question and native scientific program', async () => {
  const originals = new Map();
  for await (const line of createInterface({ input: createReadStream(new URL('./assets/train-full-v2.jsonl', import.meta.url)) })) {
    const { case_id, question, calls } = JSON.parse(line);
    originals.set(case_id, {question, calls});
  }
  const families = new Map(), ids = new Set(), texts = new Set(), names = new Set(), ancestors = new Set();
  const groups={};
  for await (const row of expandedScience()) {
    assert.ok(!ids.has(row.case_id) && !texts.has(row.question));
    assert.equal(row.split, 'train');
    assert.deepEqual(decodeCalls(row.target), row.calls);
    const source = row.ancestor_case_ids.map(id => originals.get(id));
    source.forEach((r, i) => {assert.ok(r);ancestors.add(row.ancestor_case_ids[i]);});
    if (source.length > 1) {
      const pick = explicitPick(row.question);
      assert.ok(row.question.includes(source[0].question) && row.question.includes(source[1].question));
      assert.deepEqual(row.calls, source[pick].calls);
      assert.equal(row.group, relation(source[0].calls[0],source[1].calls[0]));
      if (!families.has(row.family_id)) families.set(row.family_id, new Set());
      families.get(row.family_id).add(canonical(row.calls));
    } else {
      assert.equal(row.group,'scientific_direct');
      assert.deepEqual(row.calls,source[0].calls);
      assert.ok(directQuestions(row.calls[0]).includes(row.question));
    }
    ids.add(row.case_id); texts.add(row.question); names.add(row.calls[0].name);
    groups[row.group]=(groups[row.group]??0)+1;
  }
  assert.deepEqual(groups,{scientific_direct:793, correct_condition:236, select_operation:1342, correct_subject:1230});
  assert.equal(names.size,57); assert.equal(ancestors.size,550);
  assert.equal(families.size,1404);
  assert.ok([...families.values()].every(answers => answers.size===2));
});
