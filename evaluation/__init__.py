"""Shared synchronous interfaces for prompt construction, execution and evaluation."""

from types import SimpleNamespace
from . import schema, registry, dialogue, functiongemma, session, model_tools, model_input, prompt
from . import (
    model_result,
    evaluation,
    actual_history,
    development_evaluator,
    history_control,
    history_train,
    known_review,
)


def api():
    """Return the names used by the dataset builders and evaluator callbacks."""
    modules = (
        schema,
        registry,
        dialogue,
        functiongemma,
        session,
        model_tools,
        model_input,
        prompt,
        model_result,
        evaluation,
        actual_history,
        development_evaluator,
        history_control,
        history_train,
        known_review,
    )
    exports = {
        key: value
        for module in modules
        for key, value in vars(module).items()
        if not key.startswith("_") and getattr(value, "__module__", None) == module.__name__
    }
    exports.update(
        functionBatchStopToken=functiongemma.functionBatchStopToken,
        dialogueTools=dialogue.dialogueTools,
        declineRequestTool=dialogue.declineRequestTool,
        declineRequestMessage=dialogue.declineRequestMessage,
        modelInputFormat=model_input.modelInputFormat,
        recoveryModelInputFormat=model_input.recoveryModelInputFormat,
        pinnedTokenizerSha256=model_input.pinnedTokenizerSha256,
        pinnedTemplateSha256=model_input.pinnedTemplateSha256,
        maximumContextTokens=model_input.maximumContextTokens,
    )
    return SimpleNamespace(**exports)
