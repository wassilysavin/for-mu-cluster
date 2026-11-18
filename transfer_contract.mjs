import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';
import { createRegistry } from './evaluation/registry.mjs';
import { clone } from './evaluation/schema.mjs';
import { renderToolContract, unpackTools } from './evaluation/prompt.mjs';

const sourceFingerprint = '14f523a431292bcae2a50bbc692974a7d7dfec0251bdb305e0ae8ac17b9046a0';
const name = 'explain_parameter_data_matching';

function pairedSchema(source) {
  assert.equal(source.fingerprint, sourceFingerprint);
  const pairs = source.contracts.bindings.filter(row => row.call.name === name).map(row => row.call.arguments);
  assert.equal(pairs.length, 5);
  return pairs.map(({ subject, reference }) => ({ properties: { subject: { const: subject }, reference: { const: reference } } }));
}

export async function transferRegistry(source) {
  const original = createRegistry(source.knowledge, source.contracts);
  assert.equal(await original.fingerprint(), sourceFingerprint);
  const contracts = structuredClone(source.contracts);
  contracts.tools.find(tool => tool.name === name).parameters.anyOf = pairedSchema(source);
  return createRegistry(source.knowledge, contracts);
}

export function upgradePacked(packed, source) {
  const tools = unpackTools(packed);
  const domain = tools.find(tool => tool.name === name);
  assert.ok(domain && !Object.hasOwn(domain.parameters, 'anyOf'), 'Expected original data-matching tool');
  const original = source.contracts.tools.find(tool => tool.name === name).parameters;
  // Shared-tool expansion may restore object members in a different order than the live renderer.
  const revised = clone({ ...original, anyOf: pairedSchema(source) });
  domain.parameters = { ...revised, properties: domain.parameters.properties, required: [] };
  const change = tools.find(tool => tool.name === 'change_' + name);
  if (change) change.parameters.properties.changes = { ...revised, required: [] };
  return renderToolContract(tools);
}

export function gradeTransfer(frozenGrade, resolved, before, after) {
  if (frozenGrade.success || frozenGrade.failure_category !== 'semantic_unresolved' ||
      resolved.branch !== 'absent' || resolved.reference_targets.length !== 0) return frozenGrade;
  const clarificationOnly = resolved.allowed_batches.length > 0 && resolved.allowed_batches.every(batch =>
    batch.length === 1 && batch[0].name === 'clarify_reference' && batch[0].arguments.kind === 'answer');
  const exchange = after.exchanges.at(-1);
  const displayed = new Set(before.semantic.turns.flatMap(turn => turn.answers.map(answer => answer.id)));
  const wrongQuote = exchange.outcome === 'committed' && exchange.model_calls?.some(call =>
    call.name === 'repeat_answer' && displayed.has(call.arguments.target));
  if (clarificationOnly && wrongQuote) return { ...frozenGrade, critical_status: 'detected',
    critical_category: 'wrong_reference', failure_category: 'wrong_reference' };
  return frozenGrade;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  assert.deepEqual(process.argv.slice(2), ['--upgrade-packed']);
  const source = JSON.parse(await readFile(new URL('./assets/registry.json', import.meta.url), 'utf8'));
  const chunks = [];
  for await (const chunk of process.stdin) chunks.push(chunk);
  process.stdout.write(upgradePacked(JSON.parse(Buffer.concat(chunks).toString('utf8')), source));
}
