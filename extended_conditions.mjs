/** Single scientific obligations under changed conditions, competing subjects and explicit scope. */
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { once } from 'node:events';
import { createReadStream } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { createInterface } from 'node:readline';
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
const order = (a,b) => a < b ? -1 : a > b ? 1 : 0;
const normalized = text => text.normalize('NFKC').toLowerCase().replace(/[^\p{L}\p{N}]+/gu, ' ').trim();
const keys = obj => Object.keys(obj).sort().join(',');
const only = (a, fields) => keys(a) === fields.split(' ').sort().join(',');
const state = value => value ? 'enabled' : 'disabled';

export function directQuestions({ name, arguments: a }) {
  const s = a.subject, what = s === 'TR' ? 'the HRF setting TR' : s;
  if (only(a, 'subject')) {
    switch (name) {
      case 'explain_purpose':
        return [`I'm reading a GEM-pRF configuration. What does ${what} represent in this analysis?`,
          `How would you explain the job of ${what} to someone seeing this parameter for the first time?`];
      case 'get_default_value':
        if (s === 'version') return [
          'For this GEM-pRF revision, which XML version value should I use as the default?',
          'I need the default XML version for the pinned GEM-pRF source. What value does this revision require?',
        ];
        return [`For ${what}, what default, if any, can I actually cite from GEM-pRF? Distinguish an example configuration value from a fallback.`,
          `A colleague labeled a value for ${what} “the default”. What do the configuration example and the runtime evidence establish?`];
      case 'explain_code_usage':
        return [`I'm following ${what} through GEM-pRF. Which code passages read or use it? Please point me to those passages.`,
          `Where should I look in the GEM-pRF source to verify how ${what} is used?`];
      case 'get_value_guidance':
        return [`I need to fill in ${what}. Which selection criteria for this setting can I justify from GEM-pRF's sources?`,
          `What should someone go by when choosing a value for ${what} in this package?`];
      case 'get_enable_guidance':
        return [`Under what circumstances should I switch ${s} on in GEM-pRF, and when would I leave it off?`,
          `I'm deciding whether to enable ${s}. What evidence can guide that choice?`];
      case 'get_units':
        return [`What unit should I attach to ${what} when reporting a GEM-pRF setup? Include any source qualifications.`,
          `How does GEM-pRF specify the units of ${what}?`];
      case 'get_allowed_values':
        return [`Which entries does GEM-pRF support for ${s}, including any qualifications about accepted values?`,
          `I need the set of supported values for ${s}. What can I enter according to GEM-pRF?`];
      case 'explain_case_sensitivity':
        return [`If capitalization changes in the value of ${s}, does GEM-pRF treat the spelling as equivalent?`,
          `For the ${s} entry, is capitalization significant when GEM-pRF reads its value?`];
      case 'explain_default_rationale':
        return [`What evidence explains why the listed default for ${s} was chosen? Mention any uncertainty in the rationale.`,
          `Can the source justify the choice of the ${s} default, beyond showing which value was used?`];
      case 'explain_path_portability':
        return [`If I take the configuration to another machine, how portable is the path for ${s}?`];
      case 'describe_path_target':
        return [`I'm checking a GEM-pRF configuration. What exactly is the path in ${s} supposed to point to?`];
      case 'explain_version_format':
        return [`How must the XML version string be written for GEM-pRF's loader to compare it correctly?`];
    }
  }
  if (name === 'explain_boolean_effect' && only(a, 'subject value') && typeof a.value === 'boolean') {
    return [`What behavior should I expect from GEM-pRF with ${s} ${state(a.value)}?`,
      `Assume the Boolean value of ${s} is ${a.value}. What effect does that have during the analysis?`];
  }
  if (name === 'explain_flag_literal_effect' && only(a, 'subject value') && typeof a.value === 'string') {
    return [`The raw XML contains ${JSON.stringify(a.value)} as the value of ${s}. How does GEM-pRF interpret that exact text?`,
      `How does the loader treat ${s} if I spell its XML value ${JSON.stringify(a.value)} with that exact capitalization?`];
  }
  if (name === 'explain_change_effect' && only(a, 'subject direction') && ['decrease', 'increase'].includes(a.direction)) {
    const higher = a.direction === 'increase';
    return [`What consequences does the implementation support if I make ${what} ${higher ? 'larger' : 'smaller'} in GEM-pRF? State the relevant assumptions.`,
      `Suppose ${what} is ${higher ? 'increased' : 'reduced'}. How would that change affect the analysis, within the conditions covered by the code?`];
  }
  if (name === 'explain_conditional_applicability' && only(a, 'condition_subject condition_value subject') && typeof a.condition_value === 'boolean') {
    return [`With ${a.condition_subject} switched ${a.condition_value ? 'on' : 'off'}, does GEM-pRF use ${s}?`,
      `I need to know if ${s} takes effect in a configuration in which ${a.condition_subject} is ${state(a.condition_value)}.`];
  }
  if (name === 'get_value_guidance') {
    let condition;
    if (only(a, 'subject refinement_enabled') && typeof a.refinement_enabled === 'boolean') condition = `refinement is ${state(a.refinement_enabled)}`;
    if (only(a, 'subject slice_time_corrected') && a.slice_time_corrected === true) condition = 'the data have undergone slice-time correction';
    if (only(a, 'subject run_labels_present') && a.run_labels_present === false) condition = 'the input filenames have no run labels';
    if (only(a, 'subject gpu_out_of_memory') && a.gpu_out_of_memory === true) condition = 'GPU memory is exhausted';
    if (only(a, 'subject preprocessing') && a.preprocessing === 'fMRIPrep') condition = 'the preprocessing pipeline is fMRIPrep';
    if (only(a, 'subject individualize') && a.individualize === true) condition = 'I want an individualized setting';
    if (condition) return [`How should I choose ${s} when ${condition}?`,
      `Assume ${condition}. Which considerations apply to choosing ${s} for GEM-pRF in that situation?`];
  }
  if (name === 'explain_item_count_limits' && only(a, 'subject bound') && ['minimum','maximum'].includes(a.bound)) {
    const low = a.bound === 'minimum';
    return [`Does GEM-pRF impose a ${low ? 'lower' : 'upper'} bound on how many ${s} entries I may supply? What are the parsing caveats?`,
      `I'm checking the ${low ? 'smallest' : 'largest'} number of ${s} entries the implementation allows. What is documented?`];
  }
  if (name === 'explain_resource_impact' && only(a, 'subject runtime memory')) {
    const quantity = a.memory && a.runtime ? 'runtime and memory use' : a.memory ? 'memory use' : a.runtime ? 'runtime' : null;
    if (quantity) return [`What does the source say about the effect of ${s} on ${quantity} in GEM-pRF?`,
      `For ${s}, my question is specifically about ${quantity}. Which effects can we infer from this implementation?`];
  }
  if (name === 'get_file_requirements' && only(a, 'subject requirement') && ['format','contents'].includes(a.requirement)) {
    const dir = s === 'stimulus.directory';
    const at = dir ? `the input files in ${s}` : `the input file named by ${s}`;
    return a.requirement === 'format' ? [`Which file formats does GEM-pRF support for ${at}?`,
      `I need the format requirements for ${at}. What is supported?`]
      : [`What contents does GEM-pRF expect in ${dir ? s : at}?`, `What needs to be present inside ${dir ? s : at} for the loader to use it?`];
  }
  if (name === 'explain_parameter_data_matching' && only(a, 'subject reference')) {
    const labels = { fmri_acquisition_repetition_time: 'the fMRI acquisition repetition time',
      stimulus_file_height: 'the height of the stimulus in the file', stimulus_file_width: 'the width of the stimulus in the file',
      measured_timecourse_length: 'the length of the measured time course', stimulus_file_frame_count: 'the frame count in the stimulus file' };
    if (labels[a.reference]) return [`Is ${what} required to match ${labels[a.reference]} in GEM-pRF?`,
      `For this implementation, does setting ${what} differently from ${labels[a.reference]} violate a matching requirement?`];
  }
  return [];
}

function oneDifference(x, y) {
  return [...new Set([...Object.keys(x), ...Object.keys(y)])].filter(k =>
    !(Object.hasOwn(x,k) && Object.hasOwn(y,k)) || canonical(x[k]) !== canonical(y[k]));
}

export function relation(a, b) {
  const x = a.arguments, y = b.arguments;
  if (a.name === b.name) {
    const diff = oneDifference(x,y);
    if (diff.length !== 1) return null;
    if (diff[0] === 'subject') return 'correct_subject';
    return 'correct_condition';
  }
  return x.subject && x.subject === y.subject ? 'select_operation' : null;
}

async function anchors() {
  const stream = createReadStream(new URL('train-full-v2.jsonl', assets));
  const hash = createHash('sha256');
  stream.on('data', chunk => hash.update(chunk));
  const records = [], old = new Set(), labels = new Map();
  for await (const line of createInterface({ input: stream, crlfDelay: Infinity })) {
    const row = JSON.parse(line), q = normalized(row.question);
    old.add(q);
    if (!['fresh','new_standalone'].includes(row.history_origin)) continue;
    const key = canonical(row.calls);
    if (!labels.has(q)) labels.set(q, new Set());
    labels.get(q).add(key);
    if (row.calls.length !== 1) continue;
    const call = row.calls[0];
    if (!/^(explain_|get_|describe_|interpret_|compare_)/.test(call.name) ||
        Object.keys(call.arguments).some(k => k.startsWith('_')) ||
        (call.name === 'explain_change_effect' && (call.arguments.subject === 'default_hrf.t' || /endpoint.*hrf|hrf.*endpoint/i.test(row.question)))) continue;
    records.push({ id: row.case_id, question: row.question, call });
  }
  assert.equal(hash.digest('hex'), sourceSha);
  const unique = new Map();
  for (const a of records.sort((x,y) => order(x.id,y.id))) {
    if (labels.get(normalized(a.question)).size !== 1) continue;
    const encoded = canonical(a.call);
    if (!unique.has(encoded)) unique.set(encoded, a);
  }
  return { anchors: [...unique.values()], old };
}

// Each pair keeps both scientific options fixed while changing which one the user asks to answer.
export function selectQuestion(a, b, pick, style) {
  const wanted = pick ? 'second' : 'first', other = pick ? 'first' : 'second';
  switch (style % 6) {
    case 0: return `I wrote down two questions. The ${wanted} one is still open; I've already dealt with the ${other}.\n(1) ${a}\n(2) ${b}\nPlease answer the open question only.`;
    case 1: return `Draft A: ${a}\nDraft B: ${b}\nI changed my mind about which question to send. Please answer draft ${pick ? 'B' : 'A'} alone.`;
    case 2: return `For orientation, here are two possible questions about GEM-pRF:\n• ${a}\n• ${b}\nFor my current analysis I only need the ${wanted} question answered.`;
    case 3: return `I'm listing both notes so you can see the distinction.\n${pick ? 'Resolved' : 'Still to answer'}: ${a}\n${pick ? 'Still to answer' : 'Resolved'}: ${b}\nCould you answer the outstanding one?`;
    case 4: return `My colleague and I are asking different things. ${pick ? "Their" : "My"} note reads “${a}”. ${pick ? "My" : "Their"} note reads “${b}”. Please answer my question; the colleague's note is background.`;
    default: return `These were the two lines in my draft:\n1. ${a}\n2. ${b}\nCross out line ${pick ? '1' : '2'}. What is the answer to the line I kept?`;
  }
}

export async function* expandedScience(render = false) {
  const { anchors: source, old } = await anchors();
  const raw = await readFile(new URL('registry.json', assets));
  assert.equal(sha(raw), registrySha);
  const registry = await transferRegistry(JSON.parse(raw));
  const programs = new Map(), used = new Set(old);
  const origin = await session.createSession(registry, { dialogueVersion: 2, sessionId: 'verify-standalone' });
  for (const a of source) {
    const after = await session.applyModelOutput(registry, origin, a.question, native.encodeCalls([a.call]));
    assert.equal(after.exchanges[0].outcome, 'committed', a.id);
    assert.equal(after.semantic.turns[0].answers.length, 1, a.id);
    programs.set(a.id, { text: after.exchanges[0].assistant, state: after.semantic });
  }
  async function entry(question, correct, ancestors, group, family) {
    const id = 'wx-sci-' + sha(canonical([family, question])).slice(0,24);
    assert.ok(!used.has(normalized(question)), 'Repeated scientific question: ' + id);
    used.add(normalized(question));
    const before = await session.createSession(registry, { dialogueVersion: 2, sessionId: family });
    const target = native.encodeCalls([correct.call]);
    const after = await session.applyModelOutput(registry, before, question, target);
    const original = programs.get(correct.id);
    assert.equal(after.exchanges[0].outcome, 'committed', id);
    assert.equal(after.exchanges[0].assistant, original.text, id);
    // The turn records the actual wording; all other semantic fields must match the original scientific answer.
    const reference = structuredClone(original.state);
    reference.turns[0].question = question;
    assert.deepEqual(after.semantic, reference, id);
    const row = { case_id: id, family_id: family, group, split: 'train', history_origin: 'fresh',
      question, calls: [correct.call], target, ancestor_case_ids: ancestors };
    if (render) row.model_input = await renderPrompt(registry, api, before, question);
    return row;
  }

  for (const a of source) {
    const family = 'wx-direct-' + sha(canonical(a.call)).slice(0,22);
    for (const question of directQuestions(a.call)) {
      if (!used.has(normalized(question))) yield await entry(question, a, [a.id], 'scientific_direct', family);
    }
  }

  const groups = new Map(['correct_condition','select_operation','correct_subject'].map(group => [group, []]));
  for (let i=0; i<source.length; i++) for (let j=i+1; j<source.length; j++) {
    const a=source[i], b=source[j], group=relation(a.call,b.call);
    if (!group || programs.get(a.id).text === programs.get(b.id).text ||
        normalized(a.question) === normalized(b.question)) continue;
    const rank=sha(canonical([a.id,b.id]));
    groups.get(group).push({ a,b,rank });
  }
  for (const [group, pairs] of groups) {
    const exposures = new Map();
    for (const {a,b,rank} of pairs.sort((p,q)=>order(p.rank,q.rank))) {
      if ([a,b].some(x => (exposures.get(x.id) ?? 0) >= 3)) continue;
      const family = 'wx-scope-' + rank.slice(0,22);
      const options = [a,b];
      const style = Number.parseInt(rank.slice(0,8),16)%6;
      for (let pick=0; pick<2; pick++) {
        yield await entry(selectQuestion(a.question,b.question,pick,style), options[pick], [a.id,b.id], group, family);
      }
      for (const x of options) exposures.set(x.id, (exposures.get(x.id) ?? 0) + 1);
    }
  }
}

if (process.argv[1] && pathToFileURL(resolve(process.argv[1])).href === import.meta.url) {
  const args = process.argv.slice(2);
  assert.ok(args.length === 0 || args.length === 1 && args[0] === '--render');
  for await (const row of expandedScience(args.length === 1)) {
    if (!process.stdout.write(JSON.stringify(row) + '\n')) await once(process.stdout, 'drain');
  }
}
