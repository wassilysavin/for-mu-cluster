import { ContractError, freeze } from './schema.mjs';

/** An observed bounded model result, not an engine/provider failure. Text can
 * spell a control marker using ordinary pieces; completion requires real ID50. */
export class IncompleteModelOutput extends ContractError {
  constructor(rawOutput, generatedTokenIds, stopReason = 'token_limit') {
    if (typeof rawOutput !== 'string' || !Array.isArray(generatedTokenIds) ||
        !generatedTokenIds.length || generatedTokenIds.length > 512 ||
        generatedTokenIds.some(id => !Number.isSafeInteger(id) || id < 0 || id >= 262144 || id === 50) ||
        stopReason !== 'token_limit') {
      throw new ContractError('invalid_generation_receipt', 'An incomplete result needs actual bounded token IDs without the native batch stop.');
    }
    super('incomplete_model_output', 'Generation reached its token limit before the native batch stop.',
      freeze({ raw_output: rawOutput, generated_token_ids: [...generatedTokenIds], stop_reason: stopReason }));
  }
}
