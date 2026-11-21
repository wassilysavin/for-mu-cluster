"""Apply previously adjudicated DEV interpretations to a saved observation."""

from .evaluation import gradeTurn, textHash
from .schema import ContractError, canonical


def applyKnownReview(registry, row, record, policy):
    key = f"{record['case_id']}:{record['before_visible_history_sha256']}"
    meaning = policy["contexts"].get(key)
    raw = record["after"]["exchanges"][-1].get("raw_output") or ""
    verdict = policy["verdicts"].get(f"{key}:{textHash(raw)}", {})
    amendments = [r for r in policy["amendments"] if row["case_id"] in r["case_ids"]]
    extra = [b for rule in amendments for b in rule["additional_allowed_batches"]]
    sharing = policy["sharing_amendment"]
    if row["case_id"] in sharing["affected_case_ids"]:
        original = next((x for x in record["before"]["exchanges"] if x["id"] == "e1"), {})
        failed = row["case_id"].endswith("-t3") and original and original["outcome"] != "committed"
        extra.extend(
            sharing[
                "allowed_batches_for_t3_when_actual_original_e1_failed"
                if failed
                else "allowed_batches_when_original_not_failed"
            ]
        )
    if not meaning and not extra:
        return record
    question_hash = textHash(row["question"])
    if meaning and meaning["question_sha256"] != question_hash:
        raise ValueError("Known interpretation belongs to another question")
    batches = (meaning or {}).get("allowed_batches", row["allowed_model_call_batches"])
    allowed = {canonical(b): b for b in [*batches, *extra]}
    review = dict(
        status="approved",
        reviewer=policy["reviewer"],
        registry_sha256=policy["registry_sha256"],
        question_sha256=question_hash,
        before_visible_history_sha256=record["before_visible_history_sha256"],
    )
    gold = dict(
        row,
        author="known-development-author",
        history_policy="reviewed_actual_candidate",
        allowed_model_call_batches=list(allowed.values()),
        review=review,
    )
    try:
        graded = gradeTurn(registry, record["before"], record["after"], gold)
    except ContractError as error:
        if error.code == "unapproved_gold":
            return record
        raise
    if graded["success"] and verdict.get("critical_status") == "detected":
        raise ValueError("Known review contradicts semantic acceptance")
    status = "none_observed" if graded["success"] else verdict.get("critical_status", graded["critical_status"])
    return dict(
        record,
        result=dict(graded, critical_status=status),
        known_review=dict(
            policy_version=policy["schema_version"],
            known_context=bool(meaning),
            known_output=bool(verdict),
            amendments=[r["rule_id"] for r in amendments],
            reviewer=policy["reviewer"],
            category=verdict.get("category"),
        ),
    )
