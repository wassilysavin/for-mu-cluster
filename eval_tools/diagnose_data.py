"""Prepare, score and reconstruct authenticated TRAIN-fit observations."""

import json
import sys
from types import SimpleNamespace

from evaluation import api
from evaluation.actual_history import createActualHistoryHarness
from evaluation.functiongemma import decodeCalls, encodeCalls
from evaluation.history_train import renderPrompt
from evaluation.schema import canonical, ContractError
from evaluation.session import (
    applyModelOutput, createSession, recordIncompleteModelOutput, recordProviderFailure, sessionHistory,
)
from .io import digest, emit, registry_at, require


compactObservationSchema = "diagnostic-observation-v1-compact"
USER = "<start_of_turn>user\n"
END = "<end_of_turn>\n<start_of_turn>model\n"


def _refs(calls):
    return [c["arguments"][key] for c in calls for key in ("target", "_replace", "_resolve", "_supersede")
            if isinstance(c["arguments"].get(key), str)]


def compactObservation(observation):
    before, after = observation["before"], observation["after"]
    exchange = after["exchanges"][-1] if after and after["exchanges"] else None
    keys = ("id", "question", "assistant", "outcome", "error_code", "semantic_turn_id")
    delivered = {k: exchange[k] for k in keys if k in exchange} if exchange else None
    return {
        **{k: v for k, v in observation.items() if k not in ("before", "after")},
        "observation_schema": compactObservationSchema,
        "before_sha256": digest(before) if before else None,
        "after_sha256": digest(after) if after else None,
        "after_semantic_sha256": digest(after["semantic"]) if after else None,
        "delivered_sha256": digest(delivered) if delivered else None, "delivered": delivered,
    }


def _payload(prompt):
    require(isinstance(prompt, str) and prompt.count(USER) == 1 and prompt.endswith(END),
            "Unexpected frozen user payload or generation boundary")
    payload = json.loads(prompt.split(USER)[1][:-len(END)])
    require(payload["format"] in {"gemprf-functiongemma-input-v1", "gemprf-functiongemma-input-v2"}
            and isinstance(payload["history"]["exchanges"], list), "Invalid frozen input format")
    return payload


def _replay(registry, stored):
    require(stored and stored["version"] == 1 and isinstance(stored["exchanges"], list), "Invalid stored session")
    before = createSession(registry, {"sessionId": stored["session_id"], "dialogueVersion": stored.get("dialogue_version") or 1})
    for ex in stored["exchanges"]:
        if ex["outcome"] == "provider_error":
            before = recordProviderFailure(registry, before, ex["question"], ex["error_code"])
        elif ex.get("error_code") == "incomplete_model_output":
            before = recordIncompleteModelOutput(registry, before, ex["question"], ex["raw_output"],
                                                  ex["generation"]["token_ids"], ex["generation"]["stop_reason"])
        else:
            before = applyModelOutput(registry, before, ex["question"], ex["raw_output"])
    require(canonical(before) == canonical(stored), "Frozen before-state does not reproduce from its original observations")
    return before


def _display(before, exchange):
    if not exchange or exchange["outcome"] != "committed":
        return None
    turn = next((t for t in before["semantic"]["turns"] if t["id"] == exchange.get("semantic_turn_id")), None)
    if not turn or not turn["answers"]:
        return None
    parts, answers = turn["response_parts"], turn["answers"]
    if len(parts) != len(answers) or any(part["kind"] != "answer" or part["id"] != answer["id"]
                                         for part, answer in zip(parts, answers)):
        return None
    if "\n\n".join(part["text"] for part in parts) == exchange["assistant"]:
        return turn
    return None


def _answer(before, target):
    return next((a for t in before["semantic"]["turns"] for a in t["answers"] if a["id"] == target), None)


def _call(name, **arguments):
    return {"name": name, "arguments": arguments}


def _choices(before, program):
    name, args = program["name"], program["arguments"]
    choices = [[program]]
    if name == "repeat_exchange":
        exchange = next((e for e in before["exchanges"] if e["id"] == args["target"]), None)
        displayed = _display(before, exchange)
        if displayed:
            choices.append([_call("repeat_answer", target=a["id"]) for a in displayed["answers"]])
    if name == "repeat_answer":
        for exchange in before["exchanges"]:
            displayed = _display(before, exchange)
            if displayed and any(a["id"] == args["target"] for a in displayed["answers"]):
                if len(displayed["answers"]) == 1:
                    choices.append([_call("repeat_exchange", target=exchange["id"])])
                break
    if isinstance(args.get("_replace"), str):
        previous = _answer(before, args["_replace"])
        if previous and previous["call"]["name"] == name:
            old = previous["call"]["arguments"]
            values = {k: v for k, v in args.items() if k != "_replace"}
            changes = {k: v for k, v in values.items() if k not in old or canonical(v) != canonical(old[k])}
            remove = [k for k in old if k not in values]
            if changes or remove:
                choices.append([_call("change_" + name, target=previous["id"], changes=changes,
                                      **({"remove": remove} if remove else {}))])
    if name.startswith("change_"):
        previous = _answer(before, args["target"])
        if previous and name == "change_" + previous["call"]["name"]:
            values = {**previous["call"]["arguments"], **args["changes"]}
            for key in args.get("remove", []):
                values.pop(key, None)
            choices.append([_call(previous["call"]["name"], **{**values, "_replace": previous["id"]})])
    return choices


def _allowed(before, calls):
    batches = [[]]
    for call in calls:
        choices = _choices(before, call)
        require(len(batches) * len(choices) <= 256, "Equivalent-program expansion exceeds the declared bound")
        batches = [a + b for a in batches for b in choices]
    return list({canonical(batch): batch for batch in batches}.values())


def materializeGeneration(registry, before, question, generation):
    if generation.get("error"):
        require(generation["error"] in ("context_overflow", "inference_failed"), "Unknown provider failure")
        return recordProviderFailure(registry, before, question, generation["error"])
    require(isinstance(generation["raw_text"], str) and type(generation["complete"]) is bool,
            "Provider must report native-batch completeness")
    if generation["complete"]:
        return applyModelOutput(registry, before, question, generation["raw_text"])
    state = applyModelOutput(registry, before, question, "")
    failed = {**state, "exchanges": [*state["exchanges"][:-1], dict(state["exchanges"][-1])]}
    failed["exchanges"][-1].update(error_code="incomplete_model_output", raw_output=generation["raw_text"],
                                  generation={"complete_native_batch": False,
                                              "provider_kind": generation.get("provider_kind") if generation.get("provider_kind") is not None else "unspecified",
                                              "token_ids": generation.get("raw_ids"),
                                              "stop_reason": generation.get("finish_reason") if generation.get("finish_reason") is not None else "provider_stopped_before_native_batch"})
    return failed


def verify_alternatives(registry, before, question, expected, alternatives):
    for batch in alternatives:
        alt = applyModelOutput(registry, before, question, encodeCalls(batch))
        ex = alt["exchanges"][-1]
        require(ex["outcome"] == "committed" and ex["assistant"] == expected["exchanges"][-1]["assistant"],
                "Equivalent program changed the complete delivered answer")
        require(canonical(alt["semantic"]["pending"]) == canonical(expected["semantic"]["pending"])
                and canonical(alt["semantic"]["turns"][:-1]) == canonical(expected["semantic"]["turns"][:-1]),
                "Equivalent program changed pending items or previous turns")


def createTrainDiagnostics(registry):
    core = api()
    harness = createActualHistoryHarness(core)

    def prepare(request):
        source, row = request["source"], request["row"]
        prompt = request.get("prompt", row.get("model_input"))
        require(source in ("common", "history") and isinstance(row["case_id"], str) and isinstance(row["question"], str)
                and isinstance(row["calls"], list) and row["calls"], "Invalid frozen TRAIN source")
        data = _payload(prompt)
        require(data["question"] == row["question"], "TRAIN question differs from frozen prompt")
        before, reason, allowed = None, None, None
        if source == "history" or request.get("before") is not None:
            before = _replay(registry, request.get("before") or row["before"])
        elif not data["history"]["exchanges"]:
            before = createSession(registry, {"sessionId": data["history"]["session_id"],
                                              "dialogueVersion": 2 if data["format"].endswith("v2") else 1})
        else:
            reason = "historical_common_row_requires_authenticated_before_state"
        if before is not None:
            require(canonical(sessionHistory(before)) == canonical(data["history"]), "Frozen visible history differs")
            require(renderPrompt(registry, core, before, row["question"]) == prompt, "Current prompt differs from frozen TRAIN")
            expected = applyModelOutput(registry, before, row["question"], encodeCalls(row["calls"]))
            require(expected["exchanges"][-1]["outcome"] == "committed", "TRAIN gold program failed execution")
            if row.get("after"):
                require(canonical(expected) == canonical(row["after"]), "Frozen TRAIN target state is not reproducible")
            allowed = _allowed(before, row["calls"])
            verify_alternatives(registry, before, row["question"], expected, allowed)
        family = row["family_id"] if row.get("family_id") is not None else row["case_id"]
        group = row.get("group") if row.get("group") is not None else row.get("history_origin")
        item = {"schema": "train-fit-case-v1", "case_id": row["case_id"], "family_id": family,
                "group": group if group is not None else "common", "source": source,
                "cohort": "outside_scope" if all(c["name"] == "decline_request" for c in row["calls"]) else "supported",
                "question": row["question"], "prompt": prompt, "prompt_sha256": digest(prompt), "before": before,
                "gold_calls": row["calls"], "exact_target": encodeCalls(row["calls"]), "allowed_batches": allowed,
                "semantic_eligible": before is not None, "semantic_exclusion_reason": reason,
                "label_basis": "frozen_reviewed_continuation_policy" if source == "history" else "frozen_training_gold",
                "label_independently_rechecked_here": False, "provenance": row.get("provenance")}
        return {**item, "case_sha256": digest(item)}

    def score(item, generation):
        require(item["schema"] == "train-fit-case-v1"
                and item["case_sha256"] == digest({k: v for k, v in item.items() if k != "case_sha256"}),
                "Prepared diagnostic case changed")
        if generation.get("error"):
            require(generation["error"] in ("context_overflow", "inference_failed"), "Unknown generation error")
        else:
            require(isinstance(generation["raw_text"], str) and type(generation["complete"]) is bool, "Invalid generation")
        complete = generation.get("complete") is True and not generation.get("error")
        valid = exact = False
        if complete:
            try:
                calls = decodeCalls(generation["raw_text"])
            except (ContractError, ValueError):
                pass
            else:
                valid = True
                exact = canonical(calls) == canonical(item["gold_calls"])
        literal = complete and generation["raw_text"] == item["exact_target"]
        result, after = None, None
        if item["semantic_eligible"]:
            after = materializeGeneration(registry, item["before"], item["question"], generation)
            targets = _refs(item["gold_calls"])
            resolved = {"question": item["question"], "branch": "frozen_train", "allowed_batches": item["allowed_batches"],
                        "reference_targets": targets, "reference_target_alternatives": list(map(_refs, item["allowed_batches"])),
                        "authored_reference_eligible": bool(targets)}
            result = harness.grade(registry, {**item, "panel": "train_fixed", "policy": {"kind": "fresh"},
                                              "turn_index": None, "reference_eligible": bool(targets)}, item["before"], after, resolved)
        keys = ("case_id", "case_sha256", "prompt_sha256", "source", "cohort", "family_id", "group", "semantic_eligible",
                "semantic_exclusion_reason", "before", "label_basis", "label_independently_rechecked_here")
        return {**{k: item[k] for k in keys}, "native_literal_exact": literal, "canonical_call_exact": exact,
                "native_exact_match": literal, "canonical_call_match": exact, "transport_valid": valid,
                "semantic_success": (result or {}).get("success"), "semantic_result": result,
                "success": (result or {}).get("success"), "critical_status": (result or {}).get("critical_status"),
                "error_kind": generation.get("error") or (after and after["exchanges"][-1].get("error_code")) or
                              (result or {}).get("failure_category"), "generation": generation, "after": after}

    def expand(item, observation):
        require(observation["observation_schema"] == compactObservationSchema, "Expected compact TRAIN receipt")
        full = score(item, observation["generation"])
        require(canonical(compactObservation(full)) == canonical(observation), "Compact TRAIN receipt does not reproduce")
        return full

    return SimpleNamespace(prepare=prepare, score=score, expand=expand)


def main(path):
    registry, _ = registry_at(path)
    diagnostic = createTrainDiagnostics(registry)
    for line in sys.stdin:
        request = json.loads(line)
        action = request["action"]
        if action == "prepare":
            emit({"kind": "prepared", "case": diagnostic.prepare(request)})
        elif action == "score":
            observation = diagnostic.score(request["case"], request["generation"])
            emit({"kind": "score", "observation": compactObservation(observation) if request.get("compact") is True else observation})
        elif action == "expand":
            emit({"kind": "expanded", "observation": diagnostic.expand(request["case"], request["observation"])})
        else:
            raise ValueError(f"Unknown diagnostic action: {action}")


if __name__ == "__main__":
    main(*sys.argv[1:])
