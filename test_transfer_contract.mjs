import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { encodeCalls } from './evaluation/functiongemma.mjs';
import { modelTools } from './evaluation/model-tools.mjs';
import { renderToolContract } from './evaluation/prompt.mjs';
import { createRegistry } from './evaluation/registry.mjs';
import { applyModelOutput, createSession } from './evaluation/session.mjs';
import { transferRegistry, upgradePacked } from './transfer_contract.mjs';

const source = JSON.parse(await readFile(new URL('./assets/registry.json', import.meta.url)));
const original = createRegistry(source.knowledge, source.contracts);
const revised = await transferRegistry(source);
const name = 'explain_parameter_data_matching';
const pairs = source.contracts.bindings.filter(b => b.call.name === name).map(b => b.call.arguments);
const send = (registry, session, call) => applyModelOutput(registry, session, 'Check the requested match.', encodeCalls([call]));

test('every admitted complete pair has a source binding', () => {
  for (const { subject } of pairs) for (const { reference } of pairs) {
    const call = { name, arguments: { subject, reference } };
    if (pairs.some(p => p.subject === subject && p.reference === reference)) {
      assert.deepEqual(revised.resolve(call), original.resolve(call));
    } else assert.throws(() => revised.validateCall(call), { code: 'invalid_arguments' });
  }
});

test('partial requests, resumption, revision and restored prompts use identical native tools', async () => {
  let old = await createSession(original, { dialogueVersion: 2, sessionId: 'schema-test' });
  let next = await createSession(revised, { dialogueVersion: 2, sessionId: 'schema-test' });
  function samePrompt() {
    const contract = session => modelTools(session === old ? original : revised, session.semantic, session.exchanges);
    assert.equal(upgradePacked(JSON.parse(renderToolContract(contract(old))), source), renderToolContract(contract(next)));
  }
  samePrompt();
  for (const call of [{ name, arguments: { subject: pairs[0].subject, _clarify: true } },
    { name, arguments: { reference: pairs[0].reference, _resolve: 'p1' } }]) {
    if (call.arguments._resolve) call.arguments._resolve = next.semantic.pending[0].id;
    old = await send(original, old, call);
    next = await send(revised, next, call);
    assert.equal(next.exchanges.at(-1).outcome, 'committed');
    samePrompt();
  }
  const target = next.semantic.turns.at(-1).answers[0].id;
  const change = args => send(revised, next, { name: 'change_' + name, arguments: { target, changes: args } });
  assert.equal((await change(pairs[1])).exchanges.at(-1).outcome, 'committed');
  const invalid = await change({ subject: pairs[1].subject });
  assert.equal(invalid.exchanges.at(-1).error_code, 'invalid_arguments');
});
