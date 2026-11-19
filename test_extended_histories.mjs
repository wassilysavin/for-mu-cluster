import assert from 'node:assert/strict';
import { createReadStream } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { createInterface } from 'node:readline';
import test from 'node:test';
import { buildExtendedHistories } from './extended_histories.mjs';
import { decodeCalls, encodeCalls } from './evaluation/functiongemma.mjs';
import { createRegistry } from './evaluation/registry.mjs';
import { applyModelOutput, sessionHistory } from './evaluation/session.mjs';

const assets = new URL('./assets/', import.meta.url);
const original = async name => readFile(new URL(name, assets), 'utf8');
async function indexed(name, select) {
  const result = new Map();
  const source = createInterface({input: createReadStream(new URL(name, assets)), crlfDelay: Infinity});
  for await (const line of source) {
    const row = JSON.parse(line);
    result.set(row.case_id, select(row));
  }
  return result;
}
const userBoundary = '<end_of_turn>\n<start_of_turn>user\n';
const modelBoundary = '<end_of_turn>\n<start_of_turn>model\n';

test('every new dialogue preserves the displayed source history and grounds its continuation in TRAIN', async () => {
  const known = await indexed('train-full-v2.jsonl', ({question,calls}) => ({question,calls}));
  const records = await indexed('diagnostic-train-contexts.jsonl', ({before,provenance}) => ({before,provenance}));
  const bundle = JSON.parse(await original('registry.json'));
  const registry = createRegistry(bundle.knowledge, bundle.contracts);
  const histories = new Map(), ids = new Set(), sessions = new Set(), groups = new Set();
  const pairedOrders = new Set(), mixedOrders = new Set(), observedRetries = new Set();
  let explicitConditions = 0;

  await buildExtendedHistories({ render: true, emit: async row => {
    assert.equal(row.split, 'train');
    assert.ok(!ids.has(row.case_id));
    ids.add(row.case_id);
    sessions.add(row.family_id);
    groups.add(row.group);
    assert.equal(row.ancestor_case_ids[0], row.history_anchor);
    assert.ok(row.ancestor_case_ids.every(id => known.has(id)));
    assert.deepEqual(decodeCalls(row.target), row.calls);
    const record = records.get(row.history_anchor), anchor = known.get(row.history_anchor);
    assert.ok(record);
    const key = row.history_anchor + '/' + row.frozen_terminal;
    if (!histories.has(key)) {
      const state = row.frozen_terminal
        ? await applyModelOutput(registry, record.before, anchor.question, encodeCalls(anchor.calls))
        : record.before;
      histories.set(key, sessionHistory(state));
    }
    const expected = histories.get(key);
    assert.equal(row.family_id, 'replay-session:' + expected.session_id);
    assert.equal(row.history_origin, 'authenticated_' + record.provenance.history_origin +
      (row.frozen_terminal ? '_plus_frozen_gold' : ''));
    assert.ok(row.model_input.endsWith(modelBoundary));
    const start = row.model_input.indexOf(userBoundary);
    assert.ok(start >= 0);
    const data = JSON.parse(row.model_input.slice(start + userBoundary.length, -modelBoundary.length));
    assert.equal(data.question, row.question);
    assert.deepEqual(data.history, expected);

    const exchanges = data.history.exchanges;
    const byAnswer = new Map(exchanges.flatMap(ex => (ex.calls ?? []).map(c => [c.id, c])));
    const byExchange = new Map(exchanges.map(e => [e.id, e]));
    for (const {name, arguments: args} of row.calls) {
      if (name === 'repeat_exchange') assert.ok(byExchange.get(args.target)?.assistant);
      if (name === 'repeat_answer') assert.ok(byAnswer.has(args.target));
      if (name === 'retry_exchange') assert.ok(['error','provider_error'].includes(byExchange.get(args.target)?.outcome));
      if (name.startsWith('change_')) {
        const previous = byAnswer.get(args.target);
        assert.deepEqual(Object.keys(args.changes), ['subject']);
        assert.notEqual(previous.arguments.subject, args.changes.subject);
        const newScience = {name: previous.name, arguments:{...previous.arguments, ...args.changes}};
        assert.equal('change_' + newScience.name, name);
        assert.deepEqual([newScience], known.get(row.ancestor_case_ids[1]).calls);
      }
    }
    if (row.group === 'retry_stored_failure') {
      const [retry, ...scientific] = row.calls;
      const failed = byExchange.get(retry.arguments.target);
      const exemplar = known.get(row.ancestor_case_ids[1]);
      assert.equal(failed.user, exemplar.question);
      assert.deepEqual(scientific, exemplar.calls);
      assert.ok(!failed.turn_id && !failed.calls);
      if (row.history_origin === 'authenticated_observed_train_fit_model_history') observedRetries.add(row.history_anchor);
    }
    if (row.group === 'ordered_stored_messages') {
      const [first, second] = row.calls.map(c => Number(c.arguments.target.slice(1)));
      assert.notEqual(first, second);
      pairedOrders.add(first < second);
    }
    if (row.group === 'mixed_lookup_and_quote') {
      const quote = row.calls.findIndex(c => c.name === 'repeat_exchange');
      assert.ok([0,1].includes(quote));
      mixedOrders.add(quote);
      assert.deepEqual([row.calls[1 - quote]], known.get(row.ancestor_case_ids[1]).calls);
    }
    if (row.question.includes('(the condition there involved ')) {
      assert.ok(['new_operation_on_referenced_subject','mixed_lookup_and_quote'].includes(row.group));
      const {subject} = row.calls.find(c => c.name !== 'repeat_exchange').arguments;
      const role = row.question.match(/\(the condition there involved ([^)]+)\)/)[1];
      assert.ok(row.question.includes(subject + ' from the '));
      assert.ok([...byAnswer.values()].some(c =>
        c.arguments.subject === subject && c.arguments.condition_subject === role));
      explicitConditions++;
    }
    if (row.group === 'requested_absent_scientific_answer') {
      assert.ok(exchanges.some(e => !e.turn_id && e.outcome !== 'committed'));
      assert.deepEqual(row.calls, [{name:'clarify_reference', arguments:{kind:'answer'}}]);
    }
    if (row.group === 'clarify_multiple_real_answers') {
      const answers = [...byAnswer.values()].filter(c => typeof c.arguments.subject === 'string');
      assert.ok(answers.some((a, i) => row.question.includes(a.arguments.subject) &&
        answers.some((b, j) => i !== j && b.arguments.subject === a.arguments.subject)));
    }
  }});
  assert.equal(ids.size, 5472);
  assert.equal(histories.size, 920);
  assert.equal(sessions.size, 219);
  assert.equal(groups.size, 11);
  assert.deepEqual(pairedOrders, new Set([false, true]));
  assert.deepEqual(mixedOrders, new Set([0,1]));
  assert.equal(observedRetries.size, 3);
  assert.ok(explicitConditions >= 69);
});
