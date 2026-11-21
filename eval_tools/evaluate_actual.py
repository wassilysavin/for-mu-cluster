"""Run or self-test the independently reviewed actual-history evaluation suite."""

import json
from pathlib import Path
import sys

from evaluation import api
from evaluation.actual_history import createActualHistoryHarness
from evaluation.functiongemma import encodeCalls
from evaluation.history_train import renderPrompt
from evaluation.model_result import IncompleteModelOutput
from evaluation.schema import canonical
from evaluation.session import applyModelOutput, createSession, recordIncompleteModelOutput, recordProviderFailure
from .diagnose_dev import PANELS, full_observation, load_bundle
from .evaluate import native_output
from .io import digest, emit, read_saved, receive, require, save


def _materialize(registry, before, question, generation):
    error = generation.get("error")
    if error:
        require(error in ("context_overflow", "inference_failed"), "Unrecognized provider failure")
        return recordProviderFailure(registry, before, question, error)
    try:
        raw = native_output(generation)
    except IncompleteModelOutput:
        return recordIncompleteModelOutput(registry, before, question, generation["raw_text"], generation["raw_ids"], "token_limit")
    return applyModelOutput(registry, before, question, raw)


def _load(registry_path, data_path, panels, review):
    ctx = load_bundle(registry_path, data_path, panels, review)
    data = ctx.data["panels"]
    loop = data["closed_loop"]
    require(len(data["fixed_prefix"]) == 288 and len(loop) == 898 and sum(r["cohort"] == "supported" for r in loop) == 834
            and sum(r["cohort"] == "outside_scope" for r in loop) == 64
            and sum(not r.get("dialogue_id") and r["cohort"] == "supported" for r in loop) == 194
            and len({r["dialogue_id"] for r in loop if r.get("dialogue_id")}) == 128, "Actual-history DEV plan count differs")
    require(all(r.get("family_id") and r["semantic_obligations"] and type(r["reference_eligible"]) is bool for r in ctx.plan),
            "Missing frozen semantic labels")
    return ctx


def _summaries(context, panels, results, harness):
    return {p: harness.summarize(context.data["panels"][p], [r for r in results if r["panel"] == p]) for p in panels}


def _selftest(context, panels):
    registry = context.registry
    harness = createActualHistoryHarness(api())
    sessions, results = {}, []
    alternatives = absent = wrong = atomic = pending = scope = 0

    def candidate(row, before, resolved, calls):
        after = applyModelOutput(registry, before, resolved["question"], encodeCalls(calls))
        return after, harness.grade(registry, row, before, after, resolved)

    def reject(grade):
        require(grade["success"] is False and grade["critical_status"] == "detected", "Incorrect reference was accepted")

    def call(name, **args):
        return {"name": name, "arguments": args}

    for row in context.plan:
        before = sessions.get(row.get("dialogue_id")) or harness.initial(registry, row)
        resolved = harness.resolve(row, before)
        for index, batch in enumerate(resolved["allowed_batches"]):
            after, grade = candidate(row, before, resolved, batch)
            require(grade["success"] is True, row["case_id"])
            if row["reference_eligible"] and resolved["reference_targets"]:
                require(grade["reference_success"] is True, "Gold reference failed")
            alternatives += 1
            if index == 0:
                results.append(grade)
                if row.get("dialogue_id"):
                    sessions[row["dialogue_id"]] = after
        if row["reference_eligible"]:
            empty = createSession(registry, {"dialogueVersion": 2, "sessionId": "absent-branch"})
            alternative = harness.resolve(row, empty)
            require(candidate(row, empty, alternative, alternative["allowed_batches"][0])[1]["success"] is True,
                    row["case_id"] + "/absent")
            absent += 1
        policy = row["policy"]
        if policy["kind"].startswith("pending_"):
            state = createSession(registry, {"dialogueVersion": 2, "sessionId": "wrong-pending-branch"})
            operation = "get_default_value" if policy.get("function") == "get_units" else "get_units"
            for _ in range(policy["target_exchange"]):
                state = applyModelOutput(registry, state, "Please ask which parameter I mean.", encodeCalls([call(operation, _clarify=True)]))
            alternate = harness.resolve(row, state)
            after, grade = candidate(row, state, alternate, alternate["allowed_batches"][0])
            require(alternate["branch"] == "absent" and grade["success"] is True and
                    len(after["semantic"]["pending"]) == len(state["semantic"]["pending"]), "Wrong pending branch")
            pending += 1
        quote = next((b for b in resolved["allowed_batches"] if len(b) > 1 and all(c["name"] == "repeat_answer" for c in b)), [])
        if quote:
            reject(candidate(row, before, resolved, quote[:1])[1])
            scope += 1
        if policy["kind"] == "repeat_answer":
            index = policy["target_exchange"] - 1
            ex = before["exchanges"][index] if 0 <= index < len(before["exchanges"]) else {}
            turn = next((t for t in before["semantic"]["turns"] if t["id"] == ex.get("semantic_turn_id")), {})
            if len(turn.get("answers", [])) > 1:
                reject(candidate(row, before, resolved, [call("repeat_exchange", target=ex["id"])])[1])
                scope += 1
        if row["panel"] == "fixed_prefix" and policy["kind"] == "quote":
            other = next(ex for ex in before["exchanges"] if ex["id"] != resolved["reference_targets"][0])
            reject(candidate(row, before, resolved, [call("repeat_exchange", target=other["id"])])[1])
            wrong += 1
        if row["panel"] == "fixed_prefix" and row.get("variant") == 1:
            after = applyModelOutput(registry, before, resolved["question"], "invalid candidate output")
            grade = harness.grade(registry, row, before, after, resolved)
            require(grade["success"] is False and grade["failure_category"] == "execution_error", "Expected atomic failure")
            atomic += 1
    duplicate = next(r for r in context.data["panels"]["fixed_prefix"] if r["group"] == "ambiguous_address")
    before = harness.initial(registry, duplicate)
    text = before["exchanges"][0]["assistant"]
    require(text == before["exchanges"][1]["assistant"], "Expected duplicate texts")
    occurrence = {**duplicate, "question": "Quote exactly the first reply.", "policy": {"kind": "quote", "target_exchange": 1},
                  "reference_eligible": True}
    resolved = harness.resolve(occurrence, before)
    after, grade = candidate(occurrence, before, resolved, [call("repeat_exchange", target="e2")])
    require(after["exchanges"][-1]["assistant"] == text, "Expected identical quotation")
    reject(grade)
    return {"bundle_sha256": context.bundleHash, "cases": len(context.plan), "alternatives": alternatives,
            "absent_branches": absent, "wrong_target_mutations": wrong, "identical_text_wrong_target_rejected": True,
            "wrong_pending_branches": pending, "incomplete_or_excess_quote_scope_rejected": scope, "atomic_failure_mutations": atomic,
            "panels": _summaries(context, panels, results, harness), "model_inference_performed": False}


def _evaluate(context, panels, output, reserve):
    saved = read_saved(output)
    require(len(saved) <= len(context.plan), "Observation file has extra rows")
    core = api()
    harness = createActualHistoryHarness(core)
    sessions, results = {}, []
    with Path(output).open("ab") as handle:
        for idx, row in enumerate(context.plan):
            before = sessions.get(row.get("dialogue_id")) or harness.initial(context.registry, row)
            resolved = harness.resolve(row, before)
            prompt = renderPrompt(context.registry, core, before, resolved["question"])
            previous = saved[idx] if idx < len(saved) else None
            if previous is not None:
                require(previous["case_id"] == row["case_id"] and previous["bundle_sha256"] == context.bundleHash and
                        previous["prompt_sha256"] == digest(prompt) and
                        previous["before_visible_history_sha256"] == resolved["before_visible_history_sha256"],
                        "Saved observation belongs to a different plan, bundle or actual history")
                generation = previous["generation"]
            else:
                emit({"kind": "generate", "case_id": row["case_id"], "prompt": prompt})
                generation = receive()
            obs = full_observation(context, harness, row, before, resolved, prompt, generation, _materialize)
            if previous is None:
                save(handle, obs, reserve)
            else:
                require(canonical(previous) == canonical(obs), "Saved observation is not exactly reproducible")
            if row.get("dialogue_id"):
                sessions[row["dialogue_id"]] = obs["after"]
            results.append({k: v for k, v in obs.items() if k not in ("before", "after", "generation", "expected_model_call_batches")})
    scores = _summaries(context, panels, results, harness)
    emit({"kind": "result", "score": {**scores["closed_loop"], "summary": scores["closed_loop"], "panels": scores,
                                     "observed_records": len(results), "bundle_sha256": context.bundleHash,
                                     "diagnostic_reference_panel": "fixed_prefix",
                                     "review_status": "automatic_conservative_unresolved_retained", "release_ready": False}})


def main(registry_path, data_path, output, config="{}", reserve="0"):
    panels = PANELS[:2] if json.loads(config).get("challenge") is False else PANELS
    context = _load(registry_path, data_path, panels, output != "--selftest")
    if output == "--selftest":
        emit({"kind": "selftest", **_selftest(context, panels)})
    elif output == "--check":
        emit({"kind": "check", "cases": len(context.plan), "panels": {p: len(context.data["panels"][p]) for p in panels},
              "bundle_sha256": context.bundleHash})
    else:
        _evaluate(context, panels, output, int(reserve))


if __name__ == "__main__":
    main(*sys.argv[1:])
