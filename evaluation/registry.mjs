import { assertSchema, canonical, clone, ContractError, freeze, validate } from './schema.mjs';

const callShape = {
  type: 'object', additionalProperties: false, required: ['name', 'arguments'],
  properties: { name: { type: 'string', minLength: 1 }, arguments: { type: 'object' } },
};

const codeSourceShape = { type: 'object', additionalProperties: false,
  required: ['commit', 'path', 'line_start', 'line_end', 'url'], properties: {
    commit: { type: 'string', minLength: 1 }, path: { type: 'string', minLength: 1 },
    line_start: { type: 'integer' }, line_end: { type: 'integer' }, url: { type: 'string', minLength: 1 },
  } };
const sourceShape = { anyOf: [codeSourceShape, { type: 'object', additionalProperties: false,
  required: ['kind', 'title', 'version', 'url'], properties: {
    kind: { type: 'string', const: 'documentation' }, title: { type: 'string', minLength: 1 },
    version: { type: 'string', minLength: 1 }, url: { type: 'string', minLength: 1 },
  } }] };
const deliveryShape = { type: 'object', additionalProperties: false,
  required: ['source_segments', 'display_order', 'evidence'], properties: {
    source_segments: { type: 'array', minItems: 1, items: { type: 'string' } },
    display_order: { type: 'array', minItems: 1, uniqueItems: true, items: { type: 'integer' } },
    before_text: { type: 'string' }, after_text: { type: 'string' },
    additional_sources: { type: 'array', items: sourceShape }, evidence: { type: 'object' },
    evidence_qualification: { type: 'object', additionalProperties: false,
      required: ['status', 'reason'], properties: {
        status: { type: 'string', const: 'insufficient_evidence' },
        reason: { type: 'string', minLength: 1 },
      } },
  } };

export function createRegistry(knowledgeInput, contractInput) {
  const knowledge = freeze(clone(knowledgeInput));
  const contracts = freeze(clone(contractInput));
  if (knowledge.schema_version !== 1 || contracts.schema_version !== 1) throw new Error('Unsupported registry version.');
  const records = new Map();
  for (const record of knowledge.records) {
    if (records.has(record.id)) throw new Error(`Duplicate knowledge record ${record.id}`);
    if (!record.answer || typeof record.answer.text !== 'string') throw new Error(`Missing source answer ${record.id}`);
    records.set(record.id, record);
  }
  const tools = new Map();
  for (const tool of contracts.tools) {
    if (tools.has(tool.name)) throw new Error(`Duplicate function ${tool.name}`);
    assertSchema(tool.parameters);
    tools.set(tool.name, tool);
  }
  const bindings = new Map();
  const resolvedRecords = new Map();
  const deliveries = new Map();
  const parameters = new Set(knowledge.parameters.map(p => p.id));
  let fingerprintPromise;
  function validateCall(input) {
    canonical(input);
    validate(callShape, input);
    const tool = tools.get(input.name);
    if (!tool) throw new ContractError('unknown_function', `Function is not implemented: ${input.name}`);
    validate(tool.parameters, input.arguments);
    return tool;
  }
  for (const binding of contracts.bindings) {
    validateCall(binding.call);
    if (!records.has(binding.knowledge_id)) throw new Error(`Missing source ${binding.knowledge_id}`);
    const ids = binding.knowledge_ids ?? [binding.knowledge_id];
    if (!Array.isArray(ids) || !ids.length || ids[0] !== binding.knowledge_id ||
        new Set(ids).size !== ids.length || ids.some(id => !records.has(id))) {
      throw new Error(`Invalid source composition for ${binding.knowledge_id}`);
    }
    if (ids.length > 1 && !binding.review?.equivalence_basis) {
      throw new Error(`Source composition needs an explicit equivalence justification: ${binding.knowledge_id}`);
    }
    if (!Array.isArray(binding.focus_parameters) ||
        new Set(binding.focus_parameters).size !== binding.focus_parameters.length ||
        binding.focus_parameters.some(id => !parameters.has(id))) {
      throw new Error(`Missing or invalid explicit focus contract for ${binding.knowledge_id}`);
    }
    const key = canonical(binding.call);
    if (bindings.has(key)) throw new Error(`Ambiguous exact call ${key}`);
    bindings.set(key, binding);
    const selected = ids.map(id => records.get(id));
    const unique = values => [...new Map(values.map(v => [canonical(v), v])).values()];
    const statuses = ['insufficient_evidence', 'supported_with_limitations', 'supported'];
    if (selected.some(r => !statuses.includes(r.answer.status))) throw new Error('Unknown source evidence status.');
    resolvedRecords.set(key, freeze({
      ...selected[0], source_record_ids: ids,
      meaning_ids: [...new Set(selected.map(r => r.meaning_id))],
      answer: {
        text: [...new Set(selected.map(r => r.answer.text))].join('\n\n'),
        status: statuses[Math.min(...selected.map(r => statuses.indexOf(r.answer.status)))],
        sources: unique(selected.flatMap(r => r.answer.sources)),
        limitations: unique(selected.flatMap(r => r.answer.limitations)),
      },
    }));
    if (binding.delivery) {
      const delivery = binding.delivery;
      validate(deliveryShape, delivery);
      if (delivery.source_segments.join('') !== resolvedRecords.get(key).answer.text ||
          delivery.display_order.length !== delivery.source_segments.length ||
          delivery.display_order.some(index => index < 0 || index >= delivery.source_segments.length)) {
        throw new Error(`Delivery must retain every exact source segment once: ${binding.knowledge_id}`);
      }
      const sources = delivery.additional_sources ?? [];
      if ((delivery.before_text || delivery.after_text) && !sources.length) {
        throw new Error('Added scientific explanation requires explicit supporting sources.');
      }
      if (sources.some(source => !/^https:\/\//.test(source.url) ||
          (source.kind !== 'documentation' && (source.line_start < 1 || source.line_end < source.line_start)))) {
        throw new Error('Invalid additional scientific source.');
      }
      const body = delivery.display_order.map(index => delivery.source_segments[index]).join('');
      deliveries.set(key, freeze({ text: [delivery.before_text, body, delivery.after_text].filter(Boolean).join('\n\n'),
        additional_sources: clone(sources),
        ...(delivery.evidence_qualification ? { evidence_qualification: clone(delivery.evidence_qualification) } : {}) }));
    }
  }

  return Object.freeze({
    tools: Object.freeze([...tools.values()]),
    coverage: contracts.coverage,
    parameterIds: Object.freeze(knowledge.parameters.map(p => p.id)),
    fingerprint() {
      if (!fingerprintPromise) {
        const material = canonical({ version: 1, contracts,
          records: knowledge.records.map(record => ({ id: record.id, meaning_id: record.meaning_id, answer: record.answer })),
          parameters: knowledge.parameters });
        fingerprintPromise = globalThis.crypto.subtle.digest('SHA-256', new TextEncoder().encode(material))
          .then(buffer => [...new Uint8Array(buffer)].map(byte => byte.toString(16).padStart(2, '0')).join(''));
      }
      return fingerprintPromise;
    },
    tool(name) { return tools.get(name); },
    validateCall,
    resolve(input) {
      validateCall(input);
      const key = canonical(input);
      const binding = bindings.get(key);
      if (!binding) {
        throw new ContractError('contract_not_implemented', 'This exact call has no implemented source contract.', { call: clone(input) });
      }
      return Object.freeze({ call: freeze(clone(input)), record: resolvedRecords.get(key),
        focus: binding.focus_parameters, ...(deliveries.has(key) ? { delivery: deliveries.get(key) } : {}) });
    },
  });
}
