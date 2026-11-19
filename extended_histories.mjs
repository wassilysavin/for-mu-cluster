/** Ground new continuations in authenticated TRAIN dialogue records and their frozen source answers. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { createInterface } from 'node:readline';
import { pathToFileURL } from 'node:url';
import * as session from './evaluation/session.mjs';
import * as native from './evaluation/functiongemma.mjs';
import * as modelInput from './evaluation/model-input.mjs';
import { createRegistry } from './evaluation/registry.mjs';
import { canonical } from './evaluation/schema.mjs';
import { ordinal, renderPrompt } from './evaluation/history-train.mjs';
import { transferRegistry } from './transfer_contract.mjs';

const api = { ...session, ...native, ...modelInput };
const assets = new URL('./assets/', import.meta.url);
const hash = data => createHash('sha256').update(data).digest('hex');
const shaObject = data => hash(canonical(data));
const trainHash = '024d21fba1d2fe73dc9f8d795552303acaeb099675d9d78df179c082c7e7afcb';
const registryHash = 'b006fbacf4fd0f7f423b9178c7f365e98a6cc91bc99d8c9a16277bfd5954d4d0';
const contextManifestHash = 'b181085faba8cdb73bcf602e83c0eb0842f52496b4d85b29f9e584804add8aa3';
const pureNames = new Set(['get_units', 'get_default_value', 'explain_purpose']);
const choose = (rows, key) => rows[parseInt(hash(key).slice(0, 8), 16) % rows.length];
const call = (name, args) => ({ name, arguments: args });
const repeat = (kind, id) => call('repeat_' + kind, { target: id });

async function pinnedJson(name, expected) {
  const bytes = await readFile(new URL(name, assets));
  assert.equal(hash(bytes), expected, name);
  return JSON.parse(bytes);
}

async function frozenRows(name, expected, project) {
  const stream = createReadStream(new URL(name, assets));
  const digest = createHash('sha256'), map = new Map();
  stream.on('data', bytes => digest.update(bytes));
  for await (const line of createInterface({ input: stream, crlfDelay: Infinity })) {
    const row = JSON.parse(line);
    assert.ok(!map.has(row.case_id), row.case_id);
    map.set(row.case_id, project(row));
  }
  assert.equal(digest.digest('hex'), expected, name);
  return map;
}

const suspect = (question, calls) => calls.some(c =>
  ['explain_change_effect', 'change_explain_change_effect'].includes(c.name) &&
  (canonical(c.arguments).includes('default_hrf.t') || /endpoint/i.test(question) && /hrf/i.test(question)));

function groundSources(rows, contexts, registry) {
  const standalone = new Map(), bySubject = new Map(), byOperation = new Map();
  for (const row of rows.values()) {
    if (contexts.has(row.case_id) || suspect(row.question, row.calls)) continue;
    if (!row.calls.length || !row.calls.every(c => registry.tool(c.name) &&
        Object.keys(c.arguments).every(k => !k.startsWith('_')))) continue;
    const label = standalone.get(row.question);
    if (label === undefined) standalone.set(row.question, row);
    else if (!label || canonical(label.calls) !== canonical(row.calls)) standalone.set(row.question, null);
    if (row.calls.length !== 1) continue;
    const c = row.calls[0], subj = c.arguments.subject;
    if (typeof subj !== 'string') continue;
    if (!byOperation.has(c.name)) byOperation.set(c.name, new Map());
    byOperation.get(c.name).set(canonical(c.arguments), row);
    if (!pureNames.has(c.name) || Object.keys(c.arguments).length !== 1) continue;
    if (!bySubject.has(subj)) bySubject.set(subj, new Map());
    bySubject.get(subj).set(c.name, row);
  }
  return { standalone, bySubject, byOperation };
}

function address(exchanges, index, key) {
  const q = exchanges[index].question, span = q.trim().split(/\s+/).slice(0, 7).join(' ');
  const pos = ordinal(index + 1);
  const choices = [`my ${pos} request`, `question ${index + 1}`];
  const back = exchanges.length - index - 1;
  if (!back) choices.push('my last question');
  if (back === 1) choices.push('the question just before my last one');
  if (!index) choices.push('my opening message');
  if (span.length >= 12 && q.includes(span) && exchanges.filter(e => e.question.includes(span)).length === 1) {
    choices.push(`the request where I wrote “${span}”`, `my question containing “${span}”`);
  }
  return choose(choices, key);
}

function term(a) {
  const subject = a.answer.call.arguments.subject;
  return typeof subject === 'string' ? subject : null;
}

function answerRef(entry, exchanges, key) {
  const section = address(exchanges, entry.index, key);
  return entry.turn.answers.length === 1 ? `the scientific answer you gave to ${section}`
    : `the ${ordinal(entry.part + 1)} scientific answer in your reply to ${section}`;
}

function parameterRef(slot, answer) {
  const { name, arguments: args } = slot.answer.call;
  if (args.condition_subject) {
    return `${args.subject} from ${answer} (the condition there involved ${args.condition_subject})`;
  }
  if (name === 'explain_parameter_data_matching') return `the configured ${args.subject} discussed in ${answer}`;
  return `the main parameter you addressed in ${answer}`;
}

function sciencePrompt(op, referent, key) {
  const choices = {
    get_units: [`What units does GEM-pRF document for ${referent}?`, `Which units go with ${referent}?`,
      `I'm checking the units now. What is the unit convention for ${referent}?`],
    get_default_value: [`What default is documented for ${referent}?`, `For ${referent}, can I see the documented default?`,
      `I also need the documented default value for ${referent}. What is it?`],
    explain_purpose: [`What is ${referent} used for in GEM-pRF?`, `Can you explain the role of ${referent} here?`,
      `What is the purpose of ${referent} in this analysis?`],
  };
  return choose(choices[op], key);
}

function casesFor(state, source, key) {
  const entries = state.exchanges, turns = new Map(state.semantic.turns.map(t => [t.id, t]));
  const slots = entries.flatMap((e, index) => {
    const t = turns.get(e.semantic_turn_id);
    return (t?.answers ?? []).flatMap((answer, part) => {
      const text = t.response_parts.find(r => r.kind === 'answer' && r.id === answer.id)?.text;
      return typeof text === 'string' && e.assistant.includes(text) ? [{ index, part, answer, text, turn: t }] : [];
    });
  });
  const plan = [], chooseAt = (rows, label) => choose(rows, key + '/' + label);
  function add(group, question, calls, expected = {}, ancestors = []) {
    plan.push({ group, question, calls, expected, ancestors });
  }
  const pos = chooseAt(entries.map((_, i) => i), 'full'), e = entries[pos];
  const ref = address(entries, pos, key + 'whole');
  add('stored_whole_reply', chooseAt([`I need the full wording of your response to ${ref}. Copy it for my notes.`,
    `Could you paste everything you sent in reply to ${ref}?`, `Please read your reply to ${ref} back verbatim.`,
    `For the record, what was your complete message after ${ref}?`], 'q-full'),
  [repeat('exchange', e.id)], { texts: [e.assistant] });

  if (slots.length) {
    const selected = chooseAt(slots, 'single'), ref = answerRef(selected, entries, key + 'single');
    add('stored_answer_part', chooseAt([`Pull out ${ref}, word for word.`, `For my notes, could you reproduce ${ref} exactly?`,
      `Would you copy just ${ref} here?`], 'q-single'), [repeat('answer', selected.answer.id)], { texts: [selected.text] });
  }

  if (entries.length >= 2) {
    const pairs = entries.flatMap((_, i) => entries.flatMap((_, j) => i === j ? [] : [[i, j]]));
    const [i, j] = chooseAt(pairs, 'pair');
    const a = address(entries, i, key+'pair-a'), b = address(entries, j, key+'pair-b');
    add('ordered_stored_messages', chooseAt([`I need two complete messages for comparison. Copy your reply to ${a}, followed by your reply to ${b}.`,
      `Read back both replies verbatim in this order: the one you sent after ${a}, then the one after ${b}.`,
      `For the transcript, put the exact response to ${a} first. After it, paste the complete response to ${b}.`], 'q-pair'),
    [repeat('exchange', entries[i].id), repeat('exchange', entries[j].id)], { texts: [entries[i].assistant, entries[j].assistant] });
  }

  const scientific = slots.filter(s => source.bySubject.has(term(s)));
  if (scientific.length) {
    const slot = chooseAt(scientific, 'new-topic'), r = answerRef(slot, entries, key + 'new');
    const targets = [...source.bySubject.get(term(slot)).values()].filter(t => t.calls[0].name !== slot.answer.call.name);
    if (targets.length) {
      const target = chooseAt(targets, 'new-op');
      const subjectRef = parameterRef(slot, r);
      add('new_operation_on_referenced_subject', sciencePrompt(target.calls[0].name, subjectRef, key),
        target.calls, { scientific: target.calls }, [target.case_id]);
      if (entries.length >= 2) {
        const otherIndex = chooseAt(entries.map((_, i) => i).filter(i => i !== slot.index), 'other');
        const other = entries[otherIndex], otherRef = address(entries, otherIndex, key + 'other');
        const question = sciencePrompt(target.calls[0].name, subjectRef, key+'two');
        const intro = `I also need your complete reply to ${otherRef} copied exactly.`;
        const scienceFirst = chooseAt([true, false], 'order');
        add('mixed_lookup_and_quote', scienceFirst ? question+' '+intro : `Start by quoting your entire reply to ${otherRef}. Then: `+question,
          scienceFirst ? [...target.calls, repeat('exchange', other.id)] : [repeat('exchange', other.id), ...target.calls],
          { scientific: target.calls, quotePart: [scienceFirst ? 1 : 0, other.assistant] }, [target.case_id]);
      }
    }
    const revisionCandidates = scientific.flatMap(slot => {
      const prior = slot.answer.call;
      const choices = [...(source.byOperation.get(prior.name)?.values() ?? [])].filter(row => {
        const alt = row.calls[0].arguments;
        return typeof alt.subject === 'string' && alt.subject !== prior.arguments.subject &&
          canonical({ ...alt, subject: prior.arguments.subject }) === canonical(prior.arguments);
      });
      return choices.length ? [{ slot, choices }] : [];
    });
    if (revisionCandidates.length) {
      const { slot, choices } = chooseAt(revisionCandidates, 'rev-slot');
      const exemplar = chooseAt(choices, 'rev-alt'), targetCall = exemplar.calls[0];
      const r = answerRef(slot, entries, key + 'rev'), from = term(slot), to = targetCall.arguments.subject;
      add('change_referenced_subject', chooseAt([`In ${r}, the parameter was ${from}. I meant ${to}. Revise that answer and keep the rest of the request the same.`,
        `Please update ${r} for ${to} in place of ${from}. Keep the operation and all other arguments as they were.`,
        `One correction to ${r}: use the parameter ${to} instead of ${from}. Everything else in that answer still applies.`], 'q-rev'),
      [call('change_'+targetCall.name, { target: slot.answer.id, changes: { subject: to } })],
      { scientific: [targetCall], revision: slot.answer.id }, [exemplar.case_id]);
    }
  }

  const fresh = entries.flatMap((ex, i) => {
    const row = source.standalone.get(ex.question);
    return row ? [{ ex, i, row }] : [];
  });
  if (fresh.length) {
    const failures = fresh.filter(f => f.ex.outcome !== 'committed');
    const { ex, i, row } = chooseAt(failures.length ? failures : fresh, 'rerun');
    const ref = address(entries, i, key + 'rerun');
    add(failures.length ? 'retry_stored_failure' : 'recompute_stored_request', chooseAt([
      `Could you work through ${ref} again from scratch using the details I originally supplied?`,
      `Please answer ${ref} anew. Use the same assumptions I gave you then.`,
      `Take another run at ${ref}; I want a newly worked answer under the original conditions.`], 'q-rerun'),
    [...(failures.length ? [call('retry_exchange', {target: ex.id})] : []), ...row.calls],
    { scientific: row.calls, ...(failures.length ? { retry: ex.id } : {}) }, [row.case_id]);
  }

  const failed = entries.flatMap((ex, i) => ex.outcome === 'committed' ? [] : [{ ex, i }]);
  if (failed.length) {
    const { ex, i } = chooseAt(failed, 'absent'), ref = address(entries, i, key+'absent');
    assert.equal(ex.semantic_turn_id, undefined);
    add('requested_absent_scientific_answer', `I want the scientific answer you actually gave to ${ref}. Can you quote that specific answer?`,
      [call('clarify_reference', {kind:'answer'})], { clarification: true });
    if (ex.id !== e.id)
      add('copy_displayed_failure', `What was the complete message shown when ${ref} failed? Please paste its text exactly.`,
        [repeat('exchange', ex.id)], { texts: [ex.assistant] });
  }

  const topicSlots = new Map();
  for (const slot of scientific) {
    const subject = term(slot);
    if (!topicSlots.has(subject)) topicSlots.set(subject, []);
    topicSlots.get(subject).push(slot);
  }
  const ambiguous = [...topicSlots].filter(([, s]) => new Set(s.map(x => x.text)).size >= 2);
  if (ambiguous.length) {
    const [topic] = chooseAt(ambiguous, 'ambiguous');
    add('clarify_multiple_real_answers', chooseAt([
      `I need one particular earlier scientific answer about ${topic}. I have several in these notes and haven't specified which one to quote.`,
      `Could you retrieve the particular ${topic} answer I have in mind? I haven't said which of the different earlier answers I'm referring to.`,
      `One of the answers about ${topic} is the one I want verbatim. Do you have enough information to tell which one I mean?`], 'q-ambig'),
    [call('clarify_reference', {kind:'answer'})], { clarification: true });
  }
  return plan;
}

async function checkContinuation(registry, state, p) {
  const target = native.encodeCalls(p.calls);
  assert.deepEqual(native.decodeCalls(target), p.calls);
  const next = await session.applyModelOutput(registry, state, p.question, target);
  const reply = next.exchanges.at(-1), turn = next.semantic.turns.at(-1);
  assert.equal(reply.outcome, 'committed', p.question+' / '+reply.error_code);
  assert.deepEqual(next.exchanges.slice(0, -1), state.exchanges);
  assert.deepEqual(next.semantic.turns.slice(0,-1), state.semantic.turns);
  assert.deepEqual(next.semantic.pending, state.semantic.pending);
  const e = p.expected;
  if (e.texts) assert.deepEqual(turn.response_parts.map(part => part.text), e.texts);
  if (e.scientific) assert.deepEqual(turn.answers.map(a => a.call), e.scientific);
  if (e.quotePart) assert.equal(turn.response_parts[e.quotePart[0]].text, e.quotePart[1]);
  if (e.clarification) assert.equal(turn.response_parts[0].kind, 'reference_clarification');
  if (e.revision) assert.equal(turn.answers[0].revises, e.revision);
  if (e.retry) assert.ok(turn.effects.some(x => x.kind === 'exchange_retried' && x.target === e.retry));
  return target;
}

export async function buildExtendedHistories({render = false, emit}) {
  const spec = await pinnedJson('diagnostic-train-contexts.json', contextManifestHash);
  assert.equal(spec.split, 'train');
  assert.equal(spec.source_dataset.sha256, trainHash);
  const [bundle, rows, contexts] = await Promise.all([pinnedJson('registry.json', registryHash),
    frozenRows('train-full-v2.jsonl', trainHash, ({case_id, question, calls, history_origin}) => ({case_id, question, calls, history_origin})),
    frozenRows('diagnostic-train-contexts.jsonl', spec.contexts.sha256, r => r)]);
  const original = createRegistry(bundle.knowledge, bundle.contracts), revised = await transferRegistry(bundle);
  assert.equal(await original.fingerprint(), spec.registry_sha256);
  const revisedId = await revised.fingerprint(), source = groundSources(rows, contexts, original);
  const states = new Map(), seenCases = new Set();
  for (const [cid, record] of contexts) {
    const row = rows.get(cid), before = record.before;
    assert.equal(shaObject(before), record.before_sha256);
    assert.equal(row.history_origin, record.provenance.history_origin);
    if (suspect(row.question, row.calls) || before.exchanges.some(e => suspect(e.question, e.model_calls ?? []))) continue;
    const origPrompt = await renderPrompt(original, api, before, row.question);
    assert.equal(hash(origPrompt), record.prompt_sha256);
    const official = await session.applyModelOutput(original, before, row.question, native.encodeCalls(row.calls));
    assert.equal(shaObject(official.exchanges), record.expected_after_exchanges_sha256);
    assert.equal(shaObject(official.semantic), record.expected_after_semantic_sha256);
    for (const [state, terminal] of [[before,false],[official,true]]) {
      const { session_id, ...visible } = session.sessionHistory(state);
      const key = shaObject(visible);
      // Prefer an originally stored before-state to an identical replayed terminal answer.
      if (states.has(key) && (terminal || !states.get(key).terminal)) continue;
      states.set(key, { state: { ...state, registry_id: revisedId }, row, key, terminal });
    }
  }
  for (const {state, row, key, terminal} of states.values()) {
    const group = 'replay-session:' + state.session_id;
    for (const p of casesFor(state,source,key)) {
      const id = 'extra-history-' + hash(key+'\n'+p.question).slice(0,24);
      assert.ok(!seenCases.has(id), id);
      seenCases.add(id);
      const target = await checkContinuation(revised,state,p);
      const output = { case_id:id, split:'train', family_id:group, group:p.group, question:p.question,
        calls:p.calls, target, history_origin:'authenticated_'+row.history_origin+(terminal?'_plus_frozen_gold':''),
        ancestor_case_ids:[...new Set([row.case_id,...p.ancestors])], history_anchor:row.case_id,
        history_fingerprint:key, frozen_terminal:terminal,
        ...(render ? {model_input:await renderPrompt(revised,api,state,p.question)} : {}) };
      await emit(output);
    }
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const args = process.argv.slice(2);
  assert.ok(args.length === 0 || (args.length === 1 && args[0] === '--render'));
  await buildExtendedHistories({render:args.length === 1,emit:row=>
    new Promise((ok,fail)=>process.stdout.write(JSON.stringify(row)+'\n',e=>e?fail(e):ok()))});
}
