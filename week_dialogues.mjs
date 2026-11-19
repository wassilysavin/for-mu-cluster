/** Build supervised continuations from TRAIN anchors and the actually displayed dialogue. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { createInterface } from 'node:readline';
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';
import * as session from './evaluation/session.mjs';
import * as native from './evaluation/functiongemma.mjs';
import * as modelInput from './evaluation/model-input.mjs';
import { canonical } from './evaluation/schema.mjs';
import { continuation, ordinal, renderPrompt } from './evaluation/history-train.mjs';
import { transferRegistry } from './transfer_contract.mjs';

const api = { ...session, ...native, ...modelInput };
const assets = new URL('./assets/', import.meta.url);
const sourceDigest = '024d21fba1d2fe73dc9f8d795552303acaeb099675d9d78df179c082c7e7afcb';
const hash = text => createHash('sha256').update(text).digest('hex');
const by = (items, key) => {
  const grouped = new Map();
  for (const item of items) {
    const k = key(item);
    if (!grouped.has(k)) grouped.set(k, []);
    grouped.get(k).push(item);
  }
  return grouped;
};
const subject = row => row.calls[0].arguments.subject;
const name = row => row.calls[0].name;
const call = (name, args) => ({ name, arguments: args });
const gold = row => ({ kind: 'gold', question: row.question, calls: row.calls, ancestor_case_ids: [row.case_id] });
const negative = row => ({ kind: 'provider_error', question: row.question,
  error_code: 'inference_failed', ancestor_case_ids: [row.case_id] });
const choose = (xs, i) => xs[i % xs.length];
const action = (kind, question, fields = {}) => ({ kind, question, ...fields });
const ord = i => ordinal(i);
const rotation = (rows, seed) => rows.map(row => [hash(seed + row.case_id), row])
  .sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0).map(([, row]) => row);

export async function scientificAnchors() {
  const stream = createReadStream(new URL('train-full-v2.jsonl', assets));
  const checked = createHash('sha256');
  stream.on('data', bytes => checked.update(bytes));
  const anchors = [];
  for await (const line of createInterface({ input: stream, crlfDelay: Infinity })) {
    const { case_id, question, calls } = JSON.parse(line);
    if (!case_id.startsWith('train-intent-') || calls.length !== 1 || typeof calls[0].arguments.subject !== 'string') continue;
    if (calls[0].name === 'explain_change_effect' &&
        (calls[0].arguments.subject === 'default_hrf.t' || (/endpoint/i.test(question) && /hrf/i.test(question)))) continue;
    anchors.push({ case_id, question, calls });
  }
  assert.equal(checked.digest('hex'), sourceDigest);
  return anchors;
}

function scientificOrder(anchors, label) {
  const queues = [...by(rotation(anchors, label), name).values()];
  return Array.from({ length: Math.max(...queues.map(q => q.length)) }, (_, i) =>
    queues.flatMap(q => q[i] ? [q[i]] : [])).flat();
}

function questionFor(kind, position, variant) {
  const order = ord(position);
  const questions = {
    full: [`What exactly did you write back to request number ${position}? Please include the entire message.`,
      `I need your complete reply to my ${order} question for my notes. Copy it as displayed.`,
      `Show the text that appeared under my ${order} request, from beginning to end.`,
      `Could you read the full response to question ${position} back to me verbatim?`],
    fresh: [`Please work out question ${position} again with the same details I supplied then.`,
      `Take another run at my ${order} request using its original assumptions.`,
      `Can you answer the question in my ${order} message afresh?`,
      `I'd like a fresh answer to request ${position}. The conditions are as before.`],
    answer: [`What was the actual scientific answer in your reply to my ${order} question? Quote it.`,
      `Retrieve the scientific answer you gave for request number ${position}, exactly as recorded.`,
      `Read back the scientific answer from your response to question ${position}.`,
      `I need the scientific answer delivered for my ${order} request, in its original words.`],
  };
  return choose(questions[kind], variant);
}

async function advance(registry, before, step) {
  const next = step.kind === 'provider_error'
    ? await session.recordProviderFailure(registry, before, step.question, step.error_code)
    : await session.applyModelOutput(registry, before, step.question,
      step.kind === 'malformed' ? step.raw_output : native.encodeCalls(step.calls));
  const expected = { gold: 'committed', malformed: 'error', provider_error: 'provider_error' }[step.kind];
  assert.equal(next.exchanges.at(-1).outcome, expected, step.question);
  return next;
}

class Scene {
  constructor(config, before, prefix = []) {
    this.config = config;
    this.before = before;
    this.prefix = prefix;
  }

  async append(step) {
    const next = await advance(this.config.registry, this.before, step);
    return new Scene(this.config, next, [...this.prefix, step]);
  }

  async example(group, policy, sources = []) {
    const { registry, family, render, emit, seen } = this.config;
    const decision = continuation(this.before, policy);
    const { question, calls, branch } = decision;
    const step = { kind: 'gold', question, calls, ancestor_case_ids: sources };
    const next = await this.append(step);
    const target = native.encodeCalls(calls);
    assert.deepEqual(native.decodeCalls(target), calls);
    assert.deepEqual(next.before.exchanges.slice(0, -1), this.before.exchanges);
    if (policy.kind === 'quote_exchange') {
      assert.deepEqual(calls, [call('repeat_exchange', { target: `e${policy.exchange_index}` })]);
      assert.equal(next.before.exchanges.at(-1).assistant, this.before.exchanges[policy.exchange_index - 1].assistant);
    }
    const { session_id, ...visibleHistory } = session.sessionHistory(this.before);
    assert.equal(session_id, family);
    const case_id = 'week-case-' + hash(canonical([visibleHistory, question])).slice(0, 20);
    assert.ok(!seen.has(case_id), 'Duplicate history and question: ' + case_id);
    seen.add(case_id);
    const ancestor_case_ids = [...new Set([...this.prefix, step].flatMap(s => s.ancestor_case_ids))].sort();
    const record = { case_id, split: 'train', family_id: family, session_id: family,
      group, question, calls, target, prefix: this.prefix, branch, ancestor_case_ids,
      history_origin: 'authored_executed_prefix' };
    if (render) record.model_input = await renderPrompt(registry, api, this.before, question);
    await emit(record);
    return next;
  }
}

async function compound(start, a, b, filler, i) {
  const intro = choose(['Two things for my analysis:', 'Please cover these questions in order.',
    'I have two separate issues.', 'Can you cover both parts of this request?'], i);
  const merged = { kind: 'gold', question: `${intro} ${a.question} ${b.question}`,
    calls: [...a.calls, ...b.calls], ancestor_case_ids: [a.case_id, b.case_id] };
  const p = start.prefix.length + 1;
  let scene = await start.append(merged);
  if (i % 3) scene = await scene.append(gold(filler));
  await scene.example('whole_message', action('quote_exchange', questionFor('full', p, i), { exchange_index: p }));
  for (const part of [1, 2]) {
    const q = choose([`Pull out just the ${ord(part)} scientific answer in your reply to request ${p}.`,
      `For my ${ord(p)} request, read back only the ${ord(part)} of the two answers you gave.`,
      `From the two answers to question ${p}, copy only answer number ${part} as it appeared.`], i);
    await scene.example('single_part', action('quote_answer', q, { exchange_index: p, part }));
  }
  await scene.example('recompute_success', action('answer_again', questionFor('fresh', p, i),
    { exchange_index: p, calls: merged.calls }));
}

async function retryTwins(start, a, filler, i) {
  const p = start.prefix.length + 1;
  const failure = i % 2 ? negative(a) : { kind: 'malformed', question: a.question,
    raw_output: 'Unable to complete this response.', ancestor_case_ids: [a.case_id] };
  for (const step of [failure, gold(a)]) {
    let scene = await start.append(step);
    if (i % 3) scene = await scene.append(gold(filler));
    const failed = step === failure;
    await scene.example(failed ? 'quote_failure' : 'whole_message',
      action('quote_exchange', questionFor('full', p, i), { exchange_index: p }));
    await scene.example(failed ? 'missing_answer' : 'single_part',
      action('quote_answer', questionFor('answer', p, i), { exchange_index: p, part: 1 }));
    const retried = await scene.example(failed ? 'retry_failure' : 'recompute_success',
      action('answer_again', questionFor('fresh', p, i), { exchange_index: p, calls: a.calls }));
    if (failed) {
      const q = choose(['Quote the scientific answer you just obtained on that second attempt.',
        'Read back the answer from the successful retry.', 'What scientific answer did the new attempt produce? Quote it exactly.'], i);
      await retried.example('recovered_answer', action('quote_answer', q, {
        exchange_index: retried.prefix.length, part: 1 }));
    }
  }
}

async function ambiguous(start, left, right, filler, i) {
  const topic = subject(left);
  const first = start.prefix.length + 1;
  let scene = await start.append(gold(left));
  if (i % 2) scene = await scene.append(gold(filler));
  scene = await scene.append(gold(right));
  const last = scene.prefix.length;
  const q = choose([`I wrote down a reference to one of the two answers on ${topic}. I didn't mark which one. Could you pull that answer up?`,
    `I need one of those two scientific answers about ${topic}. My note only gives the parameter name. Can you repeat the one I meant?`,
    `Which answer on ${topic} did I want to copy? I have two candidates in this chat and haven't specified which.`], i);
  const clarified = await scene.example('ambiguous_reference', action('clarify_reference', q, { reference_kind: 'answer' }));
  for (const [index, pos] of [first, last].entries()) {
    const order = index === 0 ? 'earlier' : 'later';
    await clarified.example('clarified_reference', action('quote_answer',
      choose([`I meant the ${order} one. Give me that scientific answer in its original words.`,
        `The ${order} answer about ${topic}, please. Read that one back.`,
        `I can narrow it down: copy just the scientific answer from request number ${pos}.`], i),
      { exchange_index: pos, part: 1 }));
  }
}

async function repeatedText(start, a, filler, i) {
  const first = start.prefix.length + 1;
  let scene = await start.append(gold(a));
  if (i % 3) scene = await scene.append(gold(filler));
  scene = await scene.append(gold(a));
  const last = scene.prefix.length;
  assert.equal(scene.before.exchanges[first - 1].assistant, scene.before.exchanges[last - 1].assistant);
  for (const [j, pos] of [first, last].entries()) {
    const which = j === 0 ? 'first' : 'second';
    await scene.example('same_text_answer_address', action('quote_answer',
      choose([`Quote the scientific answer from the ${which} time I asked that identical question.`,
        `I asked the same question twice. Copy the answer recorded on the ${which} occasion.`,
        `Take the ${which} occurrence of that repeated question. Read back its scientific answer.`], i),
      { exchange_index: pos, part: 1 }));
  }
  const pos = i % 2 ? first : last;
  await scene.example('whole_message', action('quote_exchange', questionFor('full', pos, i), { exchange_index: pos }));
}

async function revision(start, a, b, filler, i) {
  const first = start.prefix.length + 1;
  let scene = await start.append(gold(a));
  if (i % 3) scene = await scene.append(gold(filler));
  const q = choose([`In the scientific answer to question ${first}, change the parameter from ${subject(a)} to ${subject(b)}. Keep every other detail of the request.`,
    `Please correct the answer to my ${ord(first)} question: the parameter should be ${subject(b)} rather than ${subject(a)}. Everything else still applies.`,
    `That ${ord(first)} request used the wrong parameter. Revise its answer for ${subject(b)}, keeping the same operation and conditions.`], i);
  const revised = await scene.example('revise_answer', action('revise_answer', q, { exchange_index: first,
    part: 1, expected_original: a.calls[0], replacement: b.calls[0], changes: { subject: subject(b) } }), [b.case_id]);
  assert.equal(revised.before.semantic.turns.at(-1).answers[0].call.arguments.subject, subject(b));
  for (const [old, pos] of [[true, first], [false, revised.prefix.length]]) {
    const text = old ? 'original answer from before my correction' : 'corrected answer from after my change';
    await revised.example('original_or_corrected', action('quote_answer',
      choose([`Please quote the ${text}.`, `For the record I need the ${text}, in your exact words.`,
        `Which wording did you use in the ${text}? Copy the scientific answer.`], i),
      { exchange_index: pos, part: 1 }));
  }
}

async function pending(start, anchor, filler, i) {
  const fn = 'explain_boolean_effect', topic = subject(anchor);
  const question = choose([`I want the effect of my particular ${topic} setting. I haven't given you its value; ask me for it first.`,
    `Can you explain what my setting for ${topic} does? First check with me whether it's on or off.`], i);
  const pendingStep = { kind: 'gold', question,
    calls: [call(fn, { subject: topic, _clarify: true, _missing: ['value'] })], ancestor_case_ids: [anchor.case_id] };
  let scene = await start.append(pendingStep);
  const pos = scene.prefix.length;
  if (i % 2) scene = await scene.append(gold(filler));
  const item = scene.before.semantic.pending.find(p => p.id === `${scene.before.exchanges[pos - 1].semantic_turn_id}.p1`);
  assert.ok(item && item.function === fn && canonical(item.known) === canonical({ subject: topic }));
  const fields = { exchange_index: pos, part: 1, function: fn, known: item.known, missing: item.missing };
  const value = anchor.calls[0].arguments.value;
  const resolved = await scene.example('resolve_pending', action('resolve_pending',
    `For the open question about ${topic}, my setting is ${value}. Please finish that answer.`, { ...fields, values: { value } }));
  assert.equal(resolved.before.semantic.pending.some(p => p.id === item.id), false);
  const cancelled = await scene.example('cancel_pending', action('cancel_pending',
    `Drop the open question about the effect of ${topic}. I no longer need that answer.`, fields));
  assert.equal(cancelled.before.semantic.pending.some(p => p.id === item.id), false);
  await scene.example('whole_message', action('quote_exchange', questionFor('full', pos, i), { exchange_index: pos }));
}

export async function buildWeekDialogues({ emit, render = false }) {
  const anchors = await scientificAnchors();
  const registry = await transferRegistry(JSON.parse(await readFile(new URL('registry.json', assets), 'utf8')));
  const byTopic = by(anchors, subject);
  const compatible = (a, b) => name(a) === name(b) && subject(a) !== subject(b) &&
    canonical({ ...a.calls[0].arguments, subject: null }) === canonical({ ...b.calls[0].arguments, subject: null });
  const revisionAnchors = anchors.filter(a => anchors.some(b => compatible(a, b)));
  const ambiguousAnchors = anchors.filter(a => byTopic.get(subject(a)).some(b => name(a) !== name(b)));
  const flags = anchors.filter(a => name(a) === 'explain_boolean_effect' && typeof a.calls[0].arguments.value === 'boolean');
  const seen = new Set();
  const modes = ['compound', 'retry', 'ambiguous', 'same_text', 'revision', 'pending'];
  for (const mode of modes) {
    const pool = mode === 'ambiguous' ? ambiguousAnchors : mode === 'revision' ? revisionAnchors :
      mode === 'pending' ? flags : anchors;
    const ordered = scientificOrder(pool, 'week-train-v1/' + mode);
    for (let i = 0; i < 64; i++) {
      const family = 'wk-' + hash('week-train-v1/' + mode + '/' + i).slice(0, 20);
      const a = choose(ordered, i);
      const candidates = rotation(anchors, family);
      const b = candidates.find(b => mode === 'revision' ? compatible(a, b) :
        mode === 'ambiguous' ? subject(a) === subject(b) && name(a) !== name(b) : subject(a) !== subject(b));
      assert.ok(b);
      const fillers = candidates.filter(f => subject(f) !== subject(a) && subject(f) !== subject(b));
      const unique = [...by(fillers, subject).values()].map(g => g[0]);
      assert.ok(unique.length >= 3);
      const config = { family, render, emit, seen, registry };
      let start = new Scene(config, await session.createSession(registry, { dialogueVersion: 2, sessionId: family }));
      for (let j = 0; j < (i % 3); j++) {
        start = await start.append(i % 4 === 1 && j === 0 ? negative(unique[j]) : gold(unique[j]));
      }
      if (mode === 'pending' && !start.prefix.length) start = await start.append(gold(unique[0]));
      const filler = unique[2];
      if (mode === 'compound') await compound(start, a, b, filler, i);
      if (mode === 'retry') await retryTwins(start, a, filler, i);
      if (mode === 'ambiguous') await ambiguous(start, a, b, filler, i);
      if (mode === 'same_text') await repeatedText(start, a, filler, i);
      if (mode === 'revision') await revision(start, a, b, filler, i);
      if (mode === 'pending') await pending(start, a, filler, i);
    }
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const args = process.argv.slice(2);
  assert.ok(!args.length || (args.length === 1 && args[0] === '--render'), 'Usage: node week_dialogues.mjs [--render]');
  await buildWeekDialogues({ render: args.length === 1, emit: row =>
    new Promise((yes, no) => process.stdout.write(JSON.stringify(row) + '\n', e => e ? no(e) : yes())) });
}
