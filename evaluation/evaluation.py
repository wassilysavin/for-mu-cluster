"""Grade complete responses against the approved programs on the actual prior state."""

from collections import defaultdict
from hashlib import sha256

from .functiongemma import encodeCalls
from .schema import ContractError, canonical
from .session import applyModelOutput


def textHash(text):
    return sha256(text.encode("utf-8")).hexdigest()


def metric(planned, passed):
    return {"planned": planned, "passed": passed, "accuracy": passed / planned if planned else None}


def gradeTurn(registry, before, after, gold, observation=None):
    review = gold.get("review") or {}

    def invalid(message):
        raise ContractError("unapproved_gold", message)

    if (
        review.get("status") != "approved"
        or not gold.get("author")
        or not review.get("reviewer")
        or gold["author"] == review["reviewer"]
    ):
        invalid("Independent approved interpretation is required.")
    if (
        review.get("question_sha256") != textHash(gold["question"])
        or review.get("registry_sha256") != registry.fingerprint()
    ):
        invalid("Question or source registry changed after review.")
    batches = gold.get("allowed_model_call_batches")
    if not isinstance(batches, list) or not batches:
        invalid("At least one independently admitted complete program is required.")
    observation = observation or {}

    def result(success, reasons, critical):
        return dict(
            case_id=gold["case_id"],
            success=success,
            reasons=reasons,
            critical_status=critical,
            observation_origin=observation.get("origin", "session_log"),
        )

    if (
        any(before.get(k) != after.get(k) for k in ("registry_id", "session_id", "dialogue_version"))
        or len(after["exchanges"]) != len(before["exchanges"]) + 1
        or canonical(after["exchanges"][:-1]) != canonical(before["exchanges"])
    ):
        return result(False, ["actual_history_rewritten_or_exchange_missing"], "detected")
    actual = after["exchanges"][-1]
    if actual["question"] != gold["question"]:
        return result(False, ["different_question"], "detected")
    best = None
    for calls in batches:
        expected = applyModelOutput(registry, before, gold["question"], encodeCalls(calls))
        required = expected["exchanges"][-1]
        if required["outcome"] != "committed":
            invalid(
                "An admitted program cannot execute on the actual previous history; adjudicate or mark blocked."
            )
        reasons = []
        if actual["outcome"] != required["outcome"]:
            reasons.append("missing_complete_outcome")
        if observation.get("visible_text", actual["assistant"]) != required["assistant"]:
            reasons.append("visible_answer_differs")
        if actual["assistant"] != required["assistant"]:
            reasons.append("stored_answer_differs")
        if canonical(after["semantic"]) != canonical(expected["semantic"]):
            reasons.append("complete_semantic_state_differs")
        if not reasons:
            return result(True, [], "none_observed")
        if best is None or len(reasons) < len(best):
            best = reasons
    unchanged = actual["outcome"] != "committed" and canonical(after["semantic"]) == canonical(before["semantic"])
    return result(False, best, "none_observed" if unchanged else "review_required")


def summarizeEvaluation(planned, results):
    ids = {r["case_id"] for r in planned}
    if len(ids) != len(planned):
        raise ValueError("Duplicate planned case.")
    if any(r["cohort"] not in ("supported", "outside_scope") for r in planned):
        raise ValueError("Unknown planned cohort.")
    passed, observed = set(), set()
    for r in results:
        cid = r["case_id"]
        if cid not in ids or cid in observed:
            raise ValueError("Unknown or duplicate result.")
        if r["critical_status"] not in ("none_observed", "review_required", "detected"):
            raise ValueError("Unclassified critical status.")
        observed.add(cid)
        if r["success"] is True and r["critical_status"] == "none_observed":
            passed.add(cid)
    groups = defaultdict(set)
    episodes = defaultdict(set)
    for r in planned:
        groups[r["cohort"]].add(r["case_id"])
        if r.get("dialogue_id"):
            episodes[r["dialogue_id"]].add(r["case_id"])
    return {
        **{c: metric(len(groups[c]), len(passed & groups[c])) for c in ("supported", "outside_scope")},
        "whole_dialogues": metric(len(episodes), sum(cases <= passed for cases in episodes.values())),
        "missing": len(ids - observed),
        "critical_detected": sum(r["critical_status"] == "detected" for r in results),
        "critical_review_required": sum(r["critical_status"] == "review_required" for r in results),
        "release_ready": False,
    }
