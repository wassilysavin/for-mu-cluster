/** Compose independent, ordered scientific obligations from authenticated standalone TRAIN anchors. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createReadStream } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { createInterface } from 'node:readline';
import { once } from 'node:events';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import * as native from './evaluation/functiongemma.mjs';
import * as modelInput from './evaluation/model-input.mjs';
import * as session from './evaluation/session.mjs';
import { renderPrompt } from './evaluation/history-train.mjs';
import { canonical } from './evaluation/schema.mjs';
import { transferRegistry } from './transfer_contract.mjs';

const api = { ...session, ...native, ...modelInput };
const assets = new URL('./assets/', import.meta.url);
const sourceSha = '024d21fba1d2fe73dc9f8d795552303acaeb099675d9d78df179c082c7e7afcb';
const registrySha = 'b006fbacf4fd0f7f423b9178c7f365e98a6cc91bc99d8c9a16277bfd5954d4d0';
const sha = data => createHash('sha256').update(data).digest('hex');
const ordered = (a, b) => a < b ? -1 : a > b ? 1 : 0;
const fingerprint = calls => canonical(calls.map(canonical).sort());
const terms = [
  'I have separate questions about GEM-pRF. Please answer each one in order.',
  'Please take the following GEM-pRF questions one at a time, in the order given.',
  'In my notes these are separate questions. Could you answer each under its own stated conditions?',
  'For each of these questions, give a separate answer in the same order.',
];
const excludesOtherObligations = /\b(only|exclude|instead of|rather than|do not|don’t|don't|leave out|just the|out of scope|not part of|not requesting|not asking|not the (?:documented|default|maximum|minimum)|previous|above|earlier|again)\b/i;

export function usableAnchor(row) {
  if (!['fresh','new_standalone'].includes(row.history_origin) || row.calls.length !== 1) return false;
  const [{ name, arguments: args }] = row.calls;
  if (!/^(explain_|get_|describe_|interpret_|compare_)/.test(name) || typeof args.subject !== 'string' ||
      Object.keys(args).some(k => k.startsWith('_')) || excludesOtherObligations.test(row.question)) return false;
  if (['explain_change_effect','change_explain_change_effect'].includes(name) &&
      (args.subject === 'default_hrf.t' || /endpoint.*hrf|hrf.*endpoint/i.test(row.question))) return false;
  const literal = args.subject.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return new RegExp('(?<![a-z0-9_])' + literal + '(?![a-z0-9_])', 'i').test(row.question);
}

async function inventory() {
  const hash = createHash('sha256');
  const input = createReadStream(new URL('train-full-v2.jsonl', assets));
  input.on('data', chunk => hash.update(chunk));
  const candidates = [], existing = new Set(), labels = new Map();
  for await (const line of createInterface({ input, crlfDelay: Infinity })) {
    const row = JSON.parse(line);
    existing.add(fingerprint(row.calls));
    if (['fresh', 'new_standalone'].includes(row.history_origin)) {
      const question = row.question.toLowerCase().trim();
      if (!labels.has(question)) labels.set(question, new Set());
      labels.get(question).add(canonical(row.calls));
    }
    if (usableAnchor(row)) {
      const call = row.calls[0];
      candidates.push({ id: row.case_id, question: row.question, call, subject: call.arguments.subject,
        options: Object.fromEntries(Object.entries(call.arguments).filter(([key]) => key !== 'subject')) });
    }
  }
  assert.equal(hash.digest('hex'), sourceSha, 'Changed source TRAIN');
  const unique = new Map();
  for (const a of candidates.sort((l, r) => ordered(l.id, r.id))) {
    if (labels.get(a.question.toLowerCase().trim()).size !== 1) continue;
    if (!unique.has(canonical(a.call))) unique.set(canonical(a.call), a);
  }
  return { anchors: [...unique.values()], existing };
}

function contrast(a, b) {
  if (a.subject === b.subject && a.call.name !== b.call.name) return 'science_same_subject';
  if (a.call.name !== b.call.name) return null;
  if (a.subject !== b.subject) return canonical(a.options) === canonical(b.options) ? 'science_same_operation' : null;
  const keys = Object.keys(a.options);
  return canonical(keys.sort()) === canonical(Object.keys(b.options).sort()) &&
    keys.filter(k => canonical(a.options[k]) !== canonical(b.options[k])).length === 1 ? 'science_argument_contrast' : null;
}

export async function buildFamilies() {
  const { anchors, existing } = await inventory();
  const sets = { science_argument_contrast: [], science_same_subject: [], science_same_operation: [] };
  for (let i = 0; i < anchors.length; i++) {
    const a = anchors[i];
    for (const b of anchors.slice(i + 1)) {
      const kind = contrast(a, b);
      if (kind && !existing.has(fingerprint([a.call, b.call]))) {
        sets[kind].push({ anchors: [a,b], rank: sha('week-science-v1|' + a.id + '|' + b.id) });
      }
    }
  }
  const families = [], frequencies = new Map();
  function add(group, members) {
    const key = fingerprint(members.map(a => a.call));
    if (existing.has(key)) return false;
    if (members.some(a => (frequencies.get(a.id) ?? 0) >= 5)) return false;
    existing.add(key);
    families.push({ family_id: 'wk-sci-family-' + sha(key).slice(0,20), group, members });
    for (const a of members) frequencies.set(a.id, (frequencies.get(a.id) ?? 0) + 1);
    return true;
  }
  for (const [group, proposals] of Object.entries(sets)) {
    proposals.sort((a, b) => ordered(a.rank, b.rank));
    let added = 0;
    for (const p of proposals) {
      if (add(group, p.anchors) && ++added >= 112) break;
    }
  }
  // Two operations for one parameter, also ask one of those operations for a different parameter.
  let triples = 0;
  for (const p of sets.science_same_subject) {
    const [a, b] = p.anchors;
    const other = anchors.filter(c => c.subject !== a.subject && c.call.name === a.call.name &&
      canonical(a.options) === canonical(c.options));
    other.sort((x, y) => ordered(sha(p.rank + x.id), sha(p.rank + y.id)));
    if (other.some(c => add('science_three_obligations', [a, b, c])) && ++triples >= 96) break;
  }
  return families;
}

export async function* buildCompositions(render = false) {
  const raw = await readFile(new URL('registry.json', assets));
  assert.equal(sha(raw), registrySha);
  const registry = await transferRegistry(JSON.parse(raw));
  const families = await buildFamilies();
  const singles = new Map();
  const seen = new Set();
  for (const { family_id, group, members } of families) {
    const head = terms[parseInt(sha(family_id).slice(0,8), 16) % terms.length];
    for (const items of [members, [...members].reverse()]) {
      const question = head + '\n' + items.map((a, i) => '(' + (i + 1) + ') ' + a.question).join('\n');
      const calls = items.map(a => a.call);
      const case_id = 'wk-sci-' + sha(canonical(items.map(a => a.id))).slice(0,20);
      assert.ok(!seen.has(question), 'Duplicate week composition');
      seen.add(question);
      const before = await session.createSession(registry, { dialogueVersion: 2, sessionId: case_id });
      const target = native.encodeCalls(calls);
      const after = await session.applyModelOutput(registry, before, question, target);
      const exchange = after.exchanges.at(-1);
      assert.equal(exchange.outcome, 'committed', case_id);
      assert.deepEqual(exchange.model_calls, calls);
      const turn = after.semantic.turns[0];
      const expected = [];
      for (const item of items) {
        if (!singles.has(item.id)) {
          const one = await session.applyModelOutput(registry, before, item.question, native.encodeCalls([item.call]));
          assert.equal(one.exchanges[0].outcome, 'committed', item.id);
          assert.equal(one.semantic.turns[0].answers.length, 1, item.id);
          singles.set(item.id, { text: one.exchanges[0].assistant, answer: one.semantic.turns[0].answers[0] });
        }
        expected.push(singles.get(item.id));
      }
      assert.equal(exchange.assistant, expected.map(e => e.text).join('\n\n'), 'Changed delivered science answer');
      const answers = expected.map((e, i) => ({ ...e.answer, id: 't1.c' + (i + 1) }));
      assert.deepEqual(turn.answers, answers);
      assert.deepEqual(turn.response_parts, expected.map((e, i) => ({ kind: 'answer', id: answers[i].id, text: e.text })));
      assert.deepEqual(turn.effects, []);
      assert.deepEqual(after.semantic.focus, [...new Set(answers.flatMap(a => a.focus))]);
      assert.equal(after.semantic.revision, 1);
      assert.deepEqual(after.semantic.pending, before.semantic.pending);
      const entry = { case_id, family_id, group, question, calls, target, history_origin: 'fresh',
        ancestor_case_ids: items.map(a => a.id), prefix: [], split: 'train' };
      if (render) entry.model_input = await renderPrompt(registry, api, before, question);
      yield entry;
    }
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) {
  const args = process.argv.slice(2);
  assert.ok(!args.length || (args.length === 1 && args[0] === '--render'), 'Expected only optional --render');
  for await (const row of buildCompositions(args.length === 1)) {
    if (!process.stdout.write(JSON.stringify(row) + '\n')) await once(process.stdout, 'drain');
  }
}
