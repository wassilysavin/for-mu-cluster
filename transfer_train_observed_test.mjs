import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {readFile} from 'node:fs/promises';
import {test} from 'node:test';
import {transferRegistry} from './transfer_contract.mjs';
import {observedSession} from './transfer_train.mjs';
import {createRegistry} from './evaluation/registry.mjs';
import * as state from './evaluation/session.mjs';
import * as model from './evaluation/model-input.mjs';
import {encodeCalls} from './evaluation/functiongemma.mjs';
import {applyObservation, renderPrompt} from './evaluation/history-train.mjs';

test('receipt replay preserves displayed failure, subsequent answer and actual addresses', async () => {
  const source = JSON.parse(await readFile(new URL('./assets/registry.json', import.meta.url)));
  const old = createRegistry(source.knowledge, source.contracts);
  const revised = await transferRegistry(source);
  const api = {...state, ...model};
  const spec = {session_id:'transfer-train-receipt-test', prefix:[
    {question:'What does TR specify?'}, {question:'What does BIDS.basepath specify?'}]};
  const hash = x => createHash('sha256').update(x).digest('hex');
  const record = {specs_sha256:'fixture', registry_sha256:await old.fingerprint(),
    model_fingerprint:{checkpoint_step:7205}, observations:[]};
  let before = await state.createSession(old, {dialogueVersion:2,sessionId:spec.session_id});
  for (const [index, step] of spec.prefix.entries()) {
    const raw = index === 0 ? 'malformed' : encodeCalls([{name:'explain_purpose',arguments:{subject:'BIDS.basepath'}}]);
    const row = {question:step.question, kind:'native', raw_output:raw,
      model_input_sha256:hash(await renderPrompt(old, api, before, step.question))};
    record.observations.push(row);
    before = await applyObservation(old, api, before, step.question, row);
  }
  record.actual_session = before;
  const migrated = await observedSession(revised, old, record, spec, spec, 'fixture');
  assert.equal(migrated.registry_id, await revised.fingerprint());
  assert.deepEqual(migrated.semantic, before.semantic);
  assert.deepEqual(migrated.exchanges, before.exchanges);
  assert.equal(migrated.exchanges[0].outcome, 'error');
  assert.equal(migrated.exchanges[1].semantic_turn_id, 't1');
  const tampered = structuredClone(record);
  tampered.actual_session.exchanges[0].assistant = 'a different reply';
  await assert.rejects(observedSession(revised, old, tampered, spec, spec, 'fixture'), /Saved TRAIN dialogue differs/);
});
