from .schema import ContractError, freeze, is_safe_integer


class IncompleteModelOutput(ContractError):
    def __init__(self, rawOutput, generatedTokenIds, stopReason="token_limit"):
        if (
            not isinstance(rawOutput, str)
            or not isinstance(generatedTokenIds, list)
            or not 1 <= len(generatedTokenIds) <= 512
            or stopReason != "token_limit"
            or any(not is_safe_integer(i) or not 0 <= i < 262144 or i == 50 for i in generatedTokenIds)
        ):
            raise ContractError(
                "invalid_generation_receipt",
                "An incomplete result needs actual bounded token IDs without the native batch stop.",
            )
        super().__init__(
            "incomplete_model_output",
            "Generation reached its token limit before the native batch stop.",
            freeze(dict(raw_output=rawOutput, generated_token_ids=generatedTokenIds, stop_reason=stopReason)),
        )
