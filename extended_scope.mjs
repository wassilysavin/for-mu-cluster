/** Executed TRAIN dialogues combining real supported and unsupported requests. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { once } from 'node:events';
import { createReadStream } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { createInterface } from 'node:readline';
import { pathToFileURL } from 'node:url';
import { declineRequestMessage } from './evaluation/dialogue.mjs';
import * as native from './evaluation/functiongemma.mjs';
import { renderPrompt } from './evaluation/history-train.mjs';
import * as modelInput from './evaluation/model-input.mjs';
import { canonical } from './evaluation/schema.mjs';
import * as session from './evaluation/session.mjs';
import { transferRegistry } from './transfer_contract.mjs';
import { usableAnchor } from './week_compositions.mjs';

const assets = new URL('./assets/', import.meta.url);
const sha = value => createHash('sha256').update(value).digest('hex');
const api = { ...session, ...native, ...modelInput };
const call = (name, args = {}) => ({ name, arguments: args });
const decline = call('decline_request');
const sourceId = '024d21fba1d2fe73dc9f8d795552303acaeb099675d9d78df179c082c7e7afcb';
const registryId = 'b006fbacf4fd0f7f423b9178c7f365e98a6cc91bc99d8c9a16277bfd5954d4d0';
const ordered = (a, b) => a < b ? -1 : a > b ? 1 : 0;
const subject = anchor => anchor.calls[0].arguments.subject;
const gold = row => ({ kind: 'gold', question: row.question, calls: row.calls, ancestor_case_ids: [row.case_id] });
const part = (kind, text) => ({ kind, text });
const declinePart = part('declined_request', declineRequestMessage);
const mix = (science, outside, outsideFirst) => outsideFirst ? [outside, science] : [science, outside];
const intros = [
  'Could you deal with both of these requests in order?',
  'I have two separate questions. Please address each in the order listed.',
  'Please go through these two items, keeping their answers separate.',
  'I would like a reply to each of these, starting with the first.',
];

async function anchors() {
  const input = createReadStream(new URL('train-full-v2.jsonl', assets));
  const hash = createHash('sha256'), outside = [], candidates = [], labels = new Map();
  input.on('data', chunk => hash.update(chunk));
  for await (const line of createInterface({ input, crlfDelay: Infinity })) {
    const { case_id, calls, question, history_origin } = JSON.parse(line);
    const row = { case_id, calls, question, history_origin };
    if (!['fresh', 'new_standalone'].includes(history_origin)) continue;
    const key = question.toLowerCase().trim();
    if (!labels.has(key)) labels.set(key, new Set());
    labels.get(key).add(canonical(calls));
    if (/^train-outside-\d+$/.test(case_id)) {
      assert.deepEqual(calls, [decline]);
      outside.push(row);
    }
    if (usableAnchor(row)) candidates.push(row);
  }
  assert.equal(hash.digest('hex'), sourceId);
  const unique = new Map();
  for (const row of candidates) {
    if (labels.get(row.question.toLowerCase().trim()).size === 1 && !unique.has(canonical(row.calls))) {
      unique.set(canonical(row.calls), row);
    }
  }
  const byOperation = new Map();
  for (const row of [...unique.values()].sort((a, b) => ordered(sha(a.case_id), sha(b.case_id)))) {
    const key = row.calls[0].name;
    if (!byOperation.has(key)) byOperation.set(key, []);
    byOperation.get(key).push(row);
  }
  const queues = [...byOperation.values()];
  const science = Array.from({ length: Math.max(...queues.map(q => q.length)) },
    (_, i) => queues.flatMap(queue => queue[i] ? [queue[i]] : [])).flat();
  assert.equal(outside.length, 200);
  return { science, outside };
}

class Scene {
  constructor(ctx, state, prefix = []) {
    this.ctx = ctx; this.state = state; this.prefix = prefix;
  }

  async committed(step) {
    const next = await session.applyModelOutput(this.ctx.registry, this.state, step.question, native.encodeCalls(step.calls));
    const exchange = next.exchanges.at(-1);
    assert.equal(exchange.outcome, 'committed', step.question + ': ' + exchange.error_code);
    assert.deepEqual(exchange.model_calls, step.calls);
    assert.deepEqual(next.exchanges.slice(0, -1), this.state.exchanges);
    assert.deepEqual(next.semantic.turns.slice(0, -1), this.state.semantic.turns);
    return new Scene(this.ctx, next, this.prefix.concat(step));
  }

  async failure(step, malformed) {
    const trace = malformed ? { kind: 'malformed', raw_output: 'Unable to return the requested response.' }
      : { kind: 'provider_error', error_code: 'inference_failed' };
    const next = malformed
      ? await session.applyModelOutput(this.ctx.registry, this.state, step.question, trace.raw_output)
      : await session.recordProviderFailure(this.ctx.registry, this.state, step.question, trace.error_code);
    assert.equal(next.exchanges.at(-1).outcome, malformed ? 'error' : 'provider_error');
    assert.deepEqual(next.semantic, this.state.semantic);
    return new Scene(this.ctx, next, this.prefix.concat({ ...trace, question: step.question, ancestor_case_ids: step.ancestor_case_ids }));
  }

  async example(group, question, calls, ancestors, expected) {
    const { registry, emit, render, seen, family, counts } = this.ctx;
    const step = { kind: 'gold', question, calls, ancestor_case_ids: ancestors };
    const next = await this.committed(step);
    const turn = next.state.semantic.turns.at(-1), parts = turn.response_parts;
    assert.deepEqual(parts.map(p => part(p.kind, p.text)), expected.parts);
    assert.equal(next.state.exchanges.at(-1).assistant, expected.parts.map(p => p.text).join('\n\n'));
    assert.deepEqual(turn.answers.map(a => a.call), expected.answers ?? []);
    if (expected.pending === 'open') {
      assert.equal(next.state.semantic.pending.length, this.state.semantic.pending.length + 1);
      const item = next.state.semantic.pending.at(-1);
      assert.equal(parts.find(p => p.kind === 'clarification').id, item.id);
      assert.equal(item.function, expected.fn);
      assert.deepEqual(item.known, expected.known);
      assert.deepEqual(item.missing, [expected.field]);
    } else if (expected.closed) {
      assert.ok(this.state.semantic.pending.some(p => p.id === expected.closed));
      assert.deepEqual(next.state.semantic.pending, this.state.semantic.pending.filter(p => p.id !== expected.closed));
    } else {
      assert.deepEqual(next.state.semantic.pending, this.state.semantic.pending);
    }
    const links = turn.answers.map(a => Object.fromEntries(
      ['repeats', 'revises', 'resolves'].filter(k => a[k]).map(k => [k, a[k]])));
    assert.deepEqual(links, expected.links ?? turn.answers.map(() => ({})));
    assert.deepEqual(turn.effects.filter(e => e.kind === 'exchange_retried').map(e => e.target), expected.retry ? [expected.retry] : []);
    const { session_id, ...history } = session.sessionHistory(this.state);
    const case_id = 'xs-scope-' + sha(canonical([history, question])).slice(0, 20);
    assert.ok(!seen.has(case_id), 'Duplicated visible question and history: ' + question);
    seen.add(case_id);
    const target = native.encodeCalls(calls);
    assert.deepEqual(native.decodeCalls(target), calls);
    const row = { case_id, family_id: family, split: 'train', group, question, calls, target,
      prefix: this.prefix, ancestor_case_ids: [...new Set(next.prefix.flatMap(s => s.ancestor_case_ids))].sort(),
      history_origin: this.prefix.length ? 'authored_executed_prefix' : 'fresh' };
    assert.ok(row.ancestor_case_ids.length);
    if (render) row.model_input = await renderPrompt(registry, api, this.state, question);
    await emit(row);
    counts[group] = (counts[group] ?? 0) + 1;
    return next;
  }
}

function different(science, idx, except) {
  for (let i = 0; i < science.length; i++) {
    const row = science[(idx + i) % science.length];
    if (except.every(a => subject(a) !== subject(row) && a.calls[0].name !== row.calls[0].name)) return row;
  }
  throw Error('Missing independent anchor');
}

async function obligations(ctx, base, a, b, outside, filler, i) {
  let start = new Scene(ctx, base);
  if (i % 3 === 2) start = await start.committed(gold(filler));
  const outsideFirst = Math.floor(i / 2) % 2 === 0, pos = outsideFirst ? 'second' : 'first', other = outsideFirst ? 'first' : 'second';
  const q = intros[Math.floor(i / 4) % intros.length] + '\n' + mix(a.question, outside.question, outsideFirst).join('\n');
  const both = mix(a.calls[0], decline, outsideFirst), origins = [a.case_id, outside.case_id];
  const pa = await ctx.standalone(a), pb = await ctx.standalone(b);
  const mixed = { parts: mix(pa, declinePart, outsideFirst), answers: a.calls };
  const done = await start.example('scope_mixed_parts', q, both, origins, mixed);
  const exchange = done.state.exchanges.at(-1), answer = done.state.semantic.turns.at(-1).answers[0].id;
  const failed = await start.failure({ question: q, ancestor_case_ids: origins }, Math.floor(i / 8) % 2 === 1);
  const retry = call('retry_exchange', { target: exchange.id });
  const quoteQ = i % 2 ? 'Copy that entire two-part reply for me, including the part you declined.'
    : 'What did your complete reply to those two items say, word for word?';
  await done.example('scope_quote_whole', quoteQ, [call('repeat_exchange', { target: exchange.id })], [],
    { parts: [part('replayed_exchange', exchange.assistant)] });
  await done.example('scope_quote_science', 'Quote the scientific answer from that reply, just the ' + pos + ' part.',
    [call('repeat_answer', { target: answer })], [], { parts: [pa], answers: a.calls, links: [{ repeats: answer }] });
  const narrow = 'Drop the ' + other + ' item. Please try just the ' + pos + ' item again, with the same assumptions.';
  await done.example('scope_retract_outside', narrow, a.calls, [], { parts: [pa], answers: a.calls });
  await failed.example('scope_retry_subset', narrow, [retry, ...a.calls], [], { parts: [pa], answers: a.calls, retry: exchange.id });
  const again = i % 2 ? 'Please answer both items afresh in their original order, keeping all the same conditions.'
    : 'Can you take another pass at both requests as originally written? Work through them in order.';
  await done.example('scope_fresh_full', again, both, [], mixed);
  await failed.example('scope_retry_full', again, [retry, ...both], [], { ...mixed, retry: exchange.id });
  if (i % 2) {
    await done.example('scope_keep_outside', 'I withdraw the ' + pos + ' item. The ' + other + ' request still stands: ' + outside.question,
      [decline], [], { parts: [declinePart] });
  } else {
    await done.example('scope_replace_science', 'I want to replace the scientific answer in that two-part reply. Use this question in its place: ' + b.question,
      [call(b.calls[0].name, { ...b.calls[0].arguments, _replace: answer })], [b.case_id],
      { parts: [pb], answers: b.calls, links: [{ revises: answer }] });
  }
}

async function pendingScope(ctx, base, a, outside, b, i) {
  const full = a.calls[0], field = { explain_change_effect: 'direction', explain_boolean_effect: 'value',
    explain_conditional_applicability: 'condition_value' }[full.name];
  const known = { ...full.arguments }, value = known[field];
  delete known[field];
  const topic = known.subject;
  const setup = field === 'direction' ? 'What would my planned change to ' + topic + ' do? First ask which direction I have in mind.'
    : field === 'condition_value' ? 'Does ' + topic + ' apply in my setup? Ask about my ' + known.condition_subject + ' setting before deciding.'
    : 'What would my setting of ' + topic + ' do? Please ask whether I mean enabled or disabled before answering.';
  const request = call(full.name, { ...known, _clarify: true, _missing: [field] });
  const isolated = await session.applyModelOutput(ctx.registry, base, setup, native.encodeCalls([request]));
  assert.equal(isolated.exchanges.at(-1).outcome, 'committed');
  const clarification = isolated.semantic.turns[0].response_parts[0];
  assert.equal(clarification.kind, 'clarification');
  const open = await new Scene(ctx, base).example('scope_mixed_pending',
    intros[Math.floor(i / 2) % intros.length] + '\n' + mix(setup, outside.question, i % 2).join('\n'),
    mix(request, decline, i % 2), [a.case_id, outside.case_id], { parts: mix(part('clarification', clarification.text), declinePart, i % 2),
      pending: 'open', known, field, fn: full.name });
  const item = open.state.semantic.pending[0].id;
  const other = await open.example('scope_keep_pending', 'I will answer your clarification later. Meanwhile, a separate question: ' + b.question,
    b.calls, [b.case_id], { parts: [await ctx.standalone(b)], answers: b.calls });
  const start = i % 2 ? other : open;
  const detail = field === 'direction' ? 'I mean a change in this direction: ' + value + '.'
    : (known.condition_subject ?? topic) + ' is ' + (value ? 'enabled' : 'disabled') + '.';
  await start.example('scope_resolve_only', 'Let’s return to the open question about ' + topic + '. ' + detail + ' Please finish only that answer.',
    [call(full.name, { _resolve: item, [field]: value })], [], { parts: [await ctx.standalone(a)], answers: [full], links: [{ resolves: item }], closed: item });
  const cancel = call('cancel_pending', { target: item });
  await start.example('scope_cancel_then_decline', 'Cancel the open question about ' + topic + '. I still want help with this other request: ' + outside.question,
    [cancel, decline], [], { parts: [part('cancellation', 'That pending request has been cancelled.'), declinePart], closed: item });
}

export async function buildExtendedScope({ emit, render = false }) {
  const { science, outside } = await anchors();
  const bytes = await readFile(new URL('registry.json', assets));
  assert.equal(sha(bytes), registryId);
  const registry = await transferRegistry(JSON.parse(bytes));
  const empty = await session.createSession(registry, { dialogueVersion: 2, sessionId: 'scope-anchor-verification' });
  const seen = new Set(), counts = {}, cached = new Map();
  async function standalone(row) {
    if (!cached.has(row.case_id)) {
      const next = await session.applyModelOutput(registry, empty, row.question, native.encodeCalls(row.calls));
      assert.equal(next.exchanges[0].outcome, 'committed', row.case_id);
      const [answer] = next.semantic.turns[0].response_parts;
      assert.equal(answer.kind, 'answer');
      cached.set(row.case_id, part('answer', answer.text));
    }
    return cached.get(row.case_id);
  }
  const pending = science.filter(a => ['explain_boolean_effect', 'explain_change_effect', 'explain_conditional_applicability'].includes(a.calls[0].name));
  const mainFamily = async (tag, ids, callback) => {
    const family = 'xs-scope-family-' + sha(canonical([tag, ids])).slice(0,16);
    const base = await session.createSession(registry, { dialogueVersion: 2, sessionId: family });
    await callback({ registry, seen, counts, family, standalone, emit, render }, base);
  };
  for (const [i, outsideRow] of outside.entries()) {
    const a = science[i % science.length], b = different(science, i + 37, [a]), filler = different(science, i + 17, [a,b]);
    await mainFamily('obligations', [a.case_id, outsideRow.case_id], (ctx, base) => obligations(ctx, base, a, b, outsideRow, filler, i));
  }
  for (const [i, a] of pending.entries()) {
    const o = outside[(i * 37 + 11) % outside.length], b = different(science, i + 71, [a]);
    await mainFamily('pending', [a.case_id, o.case_id], (ctx, base) => pendingScope(ctx, base, a, o, b, i));
  }
  return { cases: seen.size, families: outside.length + pending.length, groups: counts };
}

if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) {
  const options = process.argv.slice(2);
  if (!options.every(o => o === '--render')) throw Error('Usage: node extended_scope.mjs [--render]');
  const render = options.includes('--render');
  const report = await buildExtendedScope({ render, emit: async row => {
    if (render && !process.stdout.write(JSON.stringify(row) + '\n')) await once(process.stdout, 'drain');
  } });
  if (!render) console.log(JSON.stringify(report));
}
