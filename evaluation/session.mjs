import { canonical, clone, ContractError, freeze } from './schema.mjs';
import { createState, executeTurn, modelHistory } from './dialogue.mjs';
import { decodeCalls } from './functiongemma.mjs';
import { compileModelCalls, modelTools } from './model-tools.mjs';
import { IncompleteModelOutput } from './model-result.mjs';

const failureMessage = "I couldn't produce a reliable answer to that request.";
const contextLimitMessage = 'This conversation is too long to continue. Please start a new chat.';
const initial = (registryId, sessionId) => freeze({ version: 1, registry_id: registryId, session_id: sessionId,
  semantic: createState(), exchanges: [] });

export async function createSession(registry, sessionId = globalThis.crypto.randomUUID()) {
  return initial(await registry.fingerprint(), sessionId);
}

function appendExchange(session, semantic, exchange) {
  const item = { id: `e${session.exchanges.length + 1}`, ...exchange };
  return freeze({ ...session, semantic, exchanges: [...session.exchanges, item] });
}

async function assertCurrentRegistry(registry, session) {
  if (session.version !== 1 || session.registry_id !== await registry.fingerprint()) {
    throw new ContractError('registry_changed', 'This conversation belongs to a different source or contract version.');
  }
}

export async function applyModelOutput(registry, session, question, rawOutput, expectedExchangeCount = session.exchanges.length) {
  await assertCurrentRegistry(registry, session);
  if (expectedExchangeCount !== session.exchanges.length) throw new ContractError('stale_generation', 'The conversation changed during generation.');
  if (typeof question !== 'string' || !question.trim()) throw new ContractError('invalid_question', 'A nonempty question is required.');
  if (typeof rawOutput !== 'string') throw new ContractError('invalid_model_output', 'Model output must be text.');
  let modelCalls;
  let internalCalls;
  let result;
  try {
    modelCalls = decodeCalls(rawOutput);
    internalCalls = compileModelCalls(registry, session.semantic, modelCalls);
    result = executeTurn(registry, session.semantic, question, internalCalls);
    if (!result.ok) throw new ContractError(result.error.code, result.error.message, result.error.details);
  } catch (error) {
    if (!(error instanceof ContractError)) throw error;
    return appendExchange(session, session.semantic, { question,
      assistant: typeof error.details?.clarification === 'string' ? error.details.clarification : failureMessage,
      outcome: 'error', error_code: error.code, raw_output: rawOutput });
  }
  return appendExchange(session, result.state, { question, assistant: result.turn.text, outcome: 'committed',
    semantic_turn_id: result.turn.id, raw_output: rawOutput, model_calls: modelCalls, internal_calls: internalCalls });
}

export async function recordProviderFailure(registry, session, question, code = 'inference_failed') {
  await assertCurrentRegistry(registry, session);
  if (typeof question !== 'string' || !question.trim()) throw new ContractError('invalid_question', 'A nonempty question is required.');
  if (typeof code !== 'string' || !/^[a-z_]+$/.test(code)) throw new ContractError('invalid_error_code', 'Invalid provider failure classification.');
  return appendExchange(session, session.semantic, { question,
    assistant: code === 'context_overflow' ? contextLimitMessage : failureMessage,
    outcome: 'provider_error', error_code: code });
}

export async function recordIncompleteModelOutput(registry, session, question, rawOutput, generatedTokenIds, stopReason) {
  await assertCurrentRegistry(registry, session);
  if (typeof question !== 'string' || !question.trim()) throw new ContractError('invalid_question', 'A nonempty question is required.');
  const checked = new IncompleteModelOutput(rawOutput, generatedTokenIds, stopReason);
  return appendExchange(session, session.semantic, { question, assistant: failureMessage,
    outcome: 'error', error_code: checked.code, raw_output: checked.details.raw_output,
    generation: { complete_native_batch: false, token_ids: checked.details.generated_token_ids,
      stop_reason: checked.details.stop_reason } });
}

export function sessionHistory(session) {
  const semantic = modelHistory(session.semantic);
  const byId = new Map(semantic.turns.map(turn => [turn.id, turn]));
  return clone({ session_id: session.session_id,
    exchanges: session.exchanges.map(exchange => {
      const turn = byId.get(exchange.semantic_turn_id);
      return { id: exchange.id, user: exchange.question, assistant: exchange.assistant, outcome: exchange.outcome,
        ...(turn ? { turn_id: turn.id, calls: turn.calls, actions: turn.actions, clarifications: turn.clarifications } : {}) };
    }), pending: semantic.pending, focus: semantic.focus });
}

export async function createSessionKey() {
  return globalThis.crypto.subtle.generateKey({ name: 'HMAC', hash: 'SHA-256' }, false, ['sign', 'verify']);
}

function assertKey(key) {
  if (!key || key.type !== 'secret' || key.extractable || key.algorithm.name !== 'HMAC' || key.algorithm.hash.name !== 'SHA-256') {
    throw new ContractError('invalid_persistence_key', 'A non-extractable session HMAC key is required.');
  }
}

export async function sealSession(session, key) {
  assertKey(key);
  const payload = canonical(session);
  const bytes = await globalThis.crypto.subtle.sign('HMAC', key, new TextEncoder().encode(payload));
  const signature = [...new Uint8Array(bytes)].map(byte => byte.toString(16).padStart(2, '0')).join('');
  return { version: 1, payload, signature };
}

export async function restoreSession(registry, envelope, key) {
  assertKey(key);
  if (envelope?.version !== 1 || typeof envelope.payload !== 'string' || !/^[0-9a-f]{64}$/.test(envelope.signature ?? '')) {
    throw new ContractError('invalid_snapshot', 'Invalid conversation snapshot.');
  }
  const signature = Uint8Array.from(envelope.signature.match(/../g).map(byte => parseInt(byte, 16)));
  const valid = await globalThis.crypto.subtle.verify('HMAC', key, signature, new TextEncoder().encode(envelope.payload));
  if (!valid) throw new ContractError('snapshot_integrity', 'Conversation integrity verification failed.');
  let saved;
  try { saved = JSON.parse(envelope.payload); canonical(saved); }
  catch { throw new ContractError('invalid_snapshot', 'Invalid conversation data.'); }
  await assertCurrentRegistry(registry, saved);
  if (typeof saved.session_id !== 'string' || !Array.isArray(saved.exchanges)) throw new ContractError('invalid_snapshot', 'Missing conversation identity or exchanges.');
  let replay = initial(saved.registry_id, saved.session_id);
  for (const exchange of saved.exchanges) {
    if (exchange.outcome === 'provider_error') replay = await recordProviderFailure(registry, replay, exchange.question, exchange.error_code);
    else if (exchange.error_code === 'incomplete_model_output') replay = await recordIncompleteModelOutput(registry, replay,
      exchange.question, exchange.raw_output, exchange.generation?.token_ids, exchange.generation?.stop_reason);
    else replay = await applyModelOutput(registry, replay, exchange.question, exchange.raw_output);
  }
  if (canonical(replay) !== canonical(saved)) throw new ContractError('snapshot_replay_mismatch', 'The stored conversation does not match its actual calls, answers and state.');
  return replay;
}

/** One in-flight generation per session. A reset cancels ownership even if a
 * provider ignores AbortSignal; its old result can never enter the new chat. */
export class SessionController {
  constructor(registry, session) {
    this.registry = registry;
    this.session = session;
    this.generation = null;
  }
  async send(question, provider) {
    if (this.generation) throw new ContractError('generation_in_progress', 'A response is already being generated.');
    const base = this.session;
    const abort = new AbortController();
    const token = { base, abort };
    this.generation = token;
    try {
      let next;
      try {
        const raw = await provider({ question, session: base, history: sessionHistory(base), tools: modelTools(this.registry, base.semantic),
          state: base.semantic, signal: abort.signal });
        next = await applyModelOutput(this.registry, base, question, raw);
      } catch (error) {
        if (abort.signal.aborted) throw error;
        if (error instanceof IncompleteModelOutput) next = await recordIncompleteModelOutput(this.registry, base, question,
          error.details.raw_output, error.details.generated_token_ids, error.details.stop_reason);
        else {
          const code = error instanceof ContractError && error.code === 'context_overflow' ? 'context_overflow' : 'inference_failed';
          next = await recordProviderFailure(this.registry, base, question, code);
        }
      }
      if (this.generation !== token || this.session !== base) throw new ContractError('stale_generation', 'The result belongs to an earlier conversation.');
      this.session = next;
      return next;
    } finally {
      if (this.generation === token) this.generation = null;
    }
  }
  async reset() {
    const token = this.generation;
    this.generation = null;
    token?.abort.abort();
    this.session = await createSession(this.registry);
    return this.session;
  }
}
