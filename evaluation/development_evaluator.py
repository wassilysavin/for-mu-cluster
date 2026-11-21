"""Validate reviewed DEV plans and evaluate each provider on its own dialogue history."""

import re
from copy import deepcopy

from .evaluation import gradeTurn, summarizeEvaluation, textHash
from .model_result import IncompleteModelOutput
from .schema import ContractError, canonical
from .session import SessionController, createSession, sessionHistory

RECOVERY = "reviewed_recovery_control"


def _fail(message):
    raise ContractError("invalid_evaluation_plan", message)


def visibleHistoryHash(session):
    return textHash(canonical({k: v for k, v in sessionHistory(session).items() if k != "session_id"}))


def _approved(registry_id, gold):
    review = gold.get("review") or {}
    if (
        review.get("status") != "approved"
        or not gold.get("author")
        or not review.get("reviewer")
        or review["reviewer"] == gold["author"]
        or review.get("registry_sha256") != registry_id
        or review.get("question_sha256") != textHash(gold["question"])
        or not isinstance(gold.get("allowed_model_call_batches"), list)
        or not gold["allowed_model_call_batches"]
    ):
        _fail("Every planned question requires current independent approved interpretation.")


def _approve_control(registry, row, version):
    control = row.get("actual_history_control") or {}
    match = re.fullmatch(r"e([1-9][0-9]*)", control.get("target") or "")
    kind = control.get("kind")
    keys = {"kind", "target", *(("match", "replace") if kind == "correct_or_fresh" else ())}
    batches = row["allowed_model_call_batches"]
    binding = textHash(canonical(dict(control=control, batches=batches)))
    if (
        version != 2
        or not row.get("dialogue_id")
        or not match
        or int(match[1]) >= row["turn_index"]
        or kind not in ("retry_exchange", "correct_or_fresh")
        or control.keys() - keys
        or control.get("replace", True) is not True
        or row["review"].get("actual_history_control_sha256") != binding
    ):
        _fail("A recovery control must bind its reviewed condition, earlier exchange and complete fresh programs.")
    if kind == "correct_or_fresh":
        registry.validateCall(control.get("match"))
    for batch in batches:
        if not isinstance(batch, list) or not batch or (kind == "correct_or_fresh" and len(batch) != 1):
            _fail("A recovery control requires complete programs, with one fresh call for a correction.")
        for c in batch:
            registry.validateCall(c)
        if kind == "correct_or_fresh" and canonical(control["match"]) == canonical(batch[0]):
            _fail("A correction control must change the independently identified call.")


def _recovery_gold(row, before, history_hash):
    control = row["actual_history_control"]
    previous = next((x for x in before["exchanges"] if x["id"] == control["target"]), None)
    if previous is None:
        return None
    batches = row["allowed_model_call_batches"]
    if control["kind"] == "retry_exchange" and previous["outcome"] != "committed":
        batches = [[dict(name="retry_exchange", arguments=dict(target=previous["id"])), *b] for b in batches]
    elif control["kind"] == "correct_or_fresh" and previous["outcome"] == "committed":
        turn = next(t for t in before["semantic"]["turns"] if t["id"] == previous["semantic_turn_id"])
        matches = [a for a in turn["answers"] if canonical(a["call"]) == canonical(control["match"])]
        if len(matches) > 1:
            return None
        if matches:
            answer = matches[0]

            def update(fresh):
                args = fresh["arguments"]
                if control.get("replace") or fresh["name"] != answer["call"]["name"]:
                    return dict(name=fresh["name"], arguments=dict(args, _replace=answer["id"]))
                known = answer["call"]["arguments"]
                changes = {k: v for k, v in args.items() if k not in known or canonical(known[k]) != canonical(v)}
                remove = [k for k in known if k not in args]
                return dict(
                    name="change_" + fresh["name"],
                    arguments=dict(target=answer["id"], changes=changes, **({"remove": remove} if remove else {})),
                )

            batches = [[update(b[0])] for b in batches]
    return dict(
        row,
        allowed_model_call_batches=batches,
        review=dict(row["review"], before_visible_history_sha256=history_hash),
    )


def validateDevelopmentPlan(registry, planned, options=None, **kwargs):
    options = {**(options or {}), **kwargs}
    split, version = options.get("split"), options.get("dialogueVersion", 1)
    if (
        split not in ("train", "dev")
        or type(version) is not int
        or version not in (1, 2)
        or not isinstance(planned, list)
        or not planned
    ):
        _fail("Explicit TRAIN-fit or DEV plan required; acceptance is separate.")
    registry_id = registry.fingerprint()
    ids, episodes = set(), {}
    for row in planned:
        cid = row.get("case_id")
        if (
            row.get("split") != split
            or not cid
            or cid in ids
            or row.get("cohort") not in ("supported", "outside_scope")
            or not isinstance(row.get("question"), str)
            or not row["question"].strip()
        ):
            _fail("Unknown split/cohort, missing question or duplicate planned case.")
        ids.add(cid)
        _approved(registry_id, row)
        if row.get("history_policy") == RECOVERY:
            _approve_control(registry, row, version)
        elif "actual_history_control" in row:
            _fail("Unexpected recovery control without its reviewed history policy.")
        if did := row.get("dialogue_id"):
            if row.get("turn_index") != episodes.get(did, 0) + 1 or row.get("history_policy") not in (
                "context_invariant",
                "exact_visible_prefix_or_independent_adjudication",
                RECOVERY,
            ):
                _fail("Complete ordered episodes and explicitly reviewed history policy required.")
            if not re.fullmatch("[a-f0-9]{64}", row["review"].get("before_visible_history_sha256") or ""):
                _fail("Dialogue gold must bind its complete independently reviewed visible prefix.")
            episodes[did] = row["turn_index"]
        elif row.get("turn_index") != 0:
            _fail("Standalone questions start in fresh sessions.")
    if any(n != 5 for n in episodes.values()):
        _fail("The declared development episodes have exactly five turns.")
    return registry_id


def evaluateDevelopment(registry, planned, options=None, **kwargs):
    opts = {**(options or {}), **kwargs}
    provider = opts.get("provider")
    on_record = opts.get("onRecord", lambda record: None)
    adjudicate, session_id = opts.get("adjudicate"), opts.get("sessionIdFor")
    stop, version, split = (
        opts.get("stopOnProviderFailure", True),
        opts.get("dialogueVersion", 1),
        opts.get("split"),
    )
    if (
        not callable(provider)
        or not callable(on_record)
        or type(stop) is not bool
        or any(f is not None and not callable(f) for f in (adjudicate, session_id))
    ):
        _fail("A valid provider and evaluation callbacks are required.")
    planned = deepcopy(planned)
    registry_id = validateDevelopmentPlan(registry, planned, split=split, dialogueVersion=version)
    sessions, blocked, results = {}, {}, []
    aborted, generated = None, 0
    for row in planned:
        cid, did, question = row["case_id"], row.get("dialogue_id"), row["question"]
        before = sessions.get(did)
        if before is None:
            config = {"dialogueVersion": version}
            if session_id:
                config["sessionId"] = session_id(did or cid)
            before = createSession(registry, config)
        history_hash = visibleHistoryHash(before)
        gold, interpretation, reason = row, "reviewed_original", blocked.get(did)
        if not reason and row.get("history_policy") == RECOVERY:
            gold = _recovery_gold(row, before, history_hash)
            if gold:
                interpretation = "pre_reviewed_recovery_control"
        if not reason and (
            not gold
            or (
                did
                and row["history_policy"] not in ("context_invariant", RECOVERY)
                and history_hash != row["review"]["before_visible_history_sha256"]
            )
        ):
            gold, interpretation = None, "actual_history_review_required"
            if adjudicate:
                evidence = dict(
                    case_id=cid,
                    question=question,
                    cohort=row["cohort"],
                    dialogue_id=did,
                    turn_index=row["turn_index"],
                    actual_before=deepcopy(before),
                    before_visible_history_sha256=history_hash,
                    registry_sha256=registry_id,
                )
                decision = adjudicate(evidence)
                if not decision or decision.get("decision") not in ("grade", "blocked", "review"):
                    _fail("Unknown actual-history adjudication.")
                if decision["decision"] == "grade":
                    gold = deepcopy(decision["gold"])
                    _approved(registry_id, gold)
                    if (
                        any(
                            row.get(k) != gold.get(k)
                            for k in (
                                "case_id",
                                "split",
                                "cohort",
                                "dialogue_id",
                                "turn_index",
                                "question",
                                "author",
                            )
                        )
                        or gold.get("history_policy") != "reviewed_actual_candidate"
                        or gold["review"].get("before_visible_history_sha256") != history_hash
                    ):
                        _fail(
                            "Adjudication must preserve the exact planned question and bind its actual visible history."
                        )
                    interpretation = "independently_reviewed_actual_history"
                elif decision["decision"] == "blocked":
                    reason = decision.get("reason")
                    if (
                        not decision.get("reviewer")
                        or decision["reviewer"] == row["author"]
                        or not isinstance(reason, str)
                        or not reason
                        or decision.get("question_sha256") != row["review"]["question_sha256"]
                        or decision.get("registry_sha256") != registry_id
                        or decision.get("before_visible_history_sha256") != history_hash
                    ):
                        _fail("Blocking a continuation requires an independent reason on its exact actual prefix.")
                    blocked[did] = reason
        if reason:
            result = dict(
                case_id=cid,
                success=False,
                reasons=["blocked_continuation"],
                critical_status="none_observed",
                observation_origin="planned_but_not_generated",
            )
            results.append(result)
            on_record(
                deepcopy(
                    dict(
                        case_id=cid,
                        status="blocked",
                        reason=reason,
                        before=before,
                        after=before,
                        question_shown=False,
                        before_visible_history_sha256=history_hash,
                        result=result,
                    )
                )
            )
            continue
        failure = None

        def generate(request):
            nonlocal failure
            try:
                raw = provider(
                    request, dict(case_id=cid, split=split, dialogue_id=did, turn_index=row["turn_index"])
                )
                if not isinstance(raw, str):
                    raise ContractError(
                        "invalid_provider_result", "Provider must return its exact raw decoded string."
                    )
                return raw
            except IncompleteModelOutput:
                raise
            except Exception as error:
                failure = getattr(error, "code", "inference_failed")
                raise

        after = SessionController(registry, before).send(question, generate)
        last = after["exchanges"][-1]
        if last["outcome"] == "provider_error":
            failure = failure or last["error_code"]
        generated += 1
        if did:
            sessions[did] = after
        if gold:
            try:
                result = gradeTurn(registry, before, after, gold)
            except ContractError as error:
                if error.code != "unapproved_gold":
                    raise
                result = dict(
                    case_id=cid,
                    success=False,
                    reasons=["actual_history_gold_requires_review"],
                    critical_status="review_required",
                    observation_origin="session_log",
                )
                interpretation = "actual_history_review_required"
        else:
            result = dict(
                case_id=cid,
                success=False,
                reasons=["actual_history_requires_independent_review"],
                critical_status="review_required",
                observation_origin="session_log",
            )
        results.append(result)
        on_record(
            deepcopy(
                dict(
                    case_id=cid,
                    status="observed",
                    interpretation=interpretation,
                    before=before,
                    after=after,
                    question_shown=True,
                    before_visible_history_sha256=history_hash,
                    adjudicated_review=gold["review"] if gold and gold is not row else None,
                    provider_failure=failure,
                    result=result,
                )
            )
        )
        if failure and stop:
            aborted = dict(case_id=cid, reason=failure)
            break
    return dict(
        schema_version=1,
        split=split,
        planned_cases=len(planned),
        generated_cases=generated,
        observed_records=len(results),
        aborted=aborted,
        stop_on_provider_failure=stop,
        results=results,
        summary=summarizeEvaluation(planned, results),
        model_quality_measured_by_this_core_alone=False,
        limitation="Provider identity/resource/decoding evidence and independent critical adjudication remain external requirements.",
    )
