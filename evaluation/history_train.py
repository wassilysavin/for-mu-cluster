"""Materialize authored or observed TRAIN histories and resolve their continuations."""

from .schema import canonical


def ordinal(n):
    words = ("first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth")
    return words[n - 1] if isinstance(n, int) and 1 <= n <= len(words) else f"{n}th"


def renderPolicy(policy, exchangeIndex=None):
    index = policy.get("exchange_index") if exchangeIndex is None else exchangeIndex
    result = dict(policy)
    if index is not None:
        result["exchange_index"] = index
    for k in ("question", "replacement_question", "missing_question", "already_correct_question"):
        template = policy.get("address_templates", {}).get(k)
        if template:
            result[k] = template.replace("{exchange_ordinal}", ordinal(index)).replace(
                "{exchange_number}", str(index)
            )
    return result


def continuation(session, policy):
    p = renderPolicy(policy)
    kind = p["kind"]
    idx = p.get("exchange_index") or 0
    ex = session["exchanges"][idx - 1] if 0 < idx <= len(session["exchanges"]) else {}
    question = p.get("question")

    def result(branch, calls, targets=(), q=None):
        return dict(
            question=question if q is None else q, calls=calls, branch=branch, reference_targets=list(targets)
        )

    def call(name, **arguments):
        return dict(name=name, arguments=arguments)

    def clarify(noun):
        return [call("clarify_reference", kind=noun)]

    if kind in ("quote_exchange", "answer_again"):
        if not ex:
            raise ValueError("Predeclared exchange is absent")
        target = ex["id"]
        if kind == "quote_exchange":
            return result("actual_exchange", [call("repeat_exchange", target=target)], [target])
        if ex["outcome"] in ("error", "provider_error"):
            return result("retry_actual_failure", [call("retry_exchange", target=target), *p["calls"]], [target])
        return result("fresh_answer_to_actual_request", p["calls"])
    if kind in ("quote_answer", "revise_answer", "replace_answer"):
        turn = next((t for t in session["semantic"]["turns"] if t["id"] == ex.get("semantic_turn_id")), {})
        answers = turn.get("answers", [])
        part = p.get("part", 1)
        if not 0 < part <= len(answers):
            return result("requested_answer_missing", clarify("answer"))
        answer = answers[part - 1]
        target = answer["id"]
        repeat = [call("repeat_answer", target=target)]
        if kind == "quote_answer":
            return result("actual_answer", repeat, [target])
        fresh = p["replacement"]
        if canonical(answer["call"]) == canonical(fresh):
            return result("requested_content_already_present", repeat, [target], p["already_correct_question"])
        if kind == "revise_answer" and canonical(answer["call"]) == canonical(p["expected_original"]):
            return result(
                "exact_original_delta",
                [call("change_" + answer["call"]["name"], target=target, changes=p["changes"])],
                [target],
            )
        branch = "explicit_replacement" if kind == "replace_answer" else "wrong_original_explicit_replacement"
        return result(
            branch,
            [call(fresh["name"], **dict(fresh["arguments"], _replace=target))],
            [target],
            p.get("replacement_question"),
        )
    if kind in ("resolve_pending", "cancel_pending"):
        wanted = f"{ex.get('semantic_turn_id')}.p{p.get('part', 1)}"
        item = next((x for x in session["semantic"]["pending"] if x["id"] == wanted), {})
        if not item or any(canonical(item.get(k)) != canonical(p.get(k)) for k in ("function", "known", "missing")):
            return result("requested_pending_missing_or_different", clarify("pending"), q=p["missing_question"])
        if kind == "cancel_pending":
            return result("cancel_actual_pending", [call("cancel_pending", target=wanted)], [wanted])
        return result(
            "resolve_actual_pending", [call(p["function"], **dict(p["values"], _resolve=wanted))], [wanted]
        )
    if kind == "clarify_reference":
        return result("preauthored_ambiguous_reference", clarify(p["reference_kind"]))
    if kind == "fresh":
        return result("fresh", p["calls"])
    raise ValueError(f"Unknown preauthored continuation kind: {kind}")


def applyObservation(registry, api, before, question, observed):
    if observed["kind"] == "provider_error":
        return api.recordProviderFailure(registry, before, question, observed["error_code"])
    if observed["kind"] == "incomplete":
        return api.recordIncompleteModelOutput(
            registry, before, question, observed["raw_output"], observed["raw_ids"], "token_limit"
        )
    return api.applyModelOutput(registry, before, question, observed["raw_output"])


def materializeScenario(registry, api, spec, rawOutputs=None):
    before = api.createSession(registry, dict(dialogueVersion=2, sessionId=spec["session_id"]))
    if rawOutputs is not None and len(rawOutputs) != len(spec["prefix"]):
        raise ValueError("Partial observations cannot be frozen")
    for i, step in enumerate(spec["prefix"]):
        q = step["question"]
        if rawOutputs is not None:
            if rawOutputs[i]["question"] != q:
                raise ValueError("Observed question differs from frozen prefix")
            before = applyObservation(registry, api, before, q, rawOutputs[i])
        elif step["kind"] == "provider_error":
            before = api.recordProviderFailure(registry, before, q, step["error_code"])
        elif step["kind"] == "malformed":
            before = api.applyModelOutput(registry, before, q, step["raw_output"])
        else:
            calls = continuation(before, step["policy"])["calls"] if step.get("policy") else step["calls"]
            before = api.applyModelOutput(registry, before, q, api.encodeCalls(calls))
            if before["exchanges"][-1]["outcome"] != "committed":
                raise ValueError(f"{spec['case_id']}: prefix {i + 1}")
    target = continuation(before, spec["continuation"])
    raw = api.encodeCalls(target["calls"])
    after = api.applyModelOutput(registry, before, target["question"], raw)
    if after["exchanges"][-1]["outcome"] != "committed" or canonical(after["exchanges"][:-1]) != canonical(
        before["exchanges"]
    ):
        raise ValueError(f"Invalid gold continuation: {spec['case_id']}")
    origin = (
        "observed_frozen_checkpoint_train_prefix" if rawOutputs is not None else "reconstructed_actual_history_v3"
    )
    return dict(
        **{k: spec[k] for k in ("case_id", "family_id", "group") if k in spec},
        split="train",
        question=target["question"],
        before=before,
        actual_before=api.sessionHistory(before),
        calls=target["calls"],
        target=raw,
        after=after,
        history_origin=origin,
        reference_targets=target["reference_targets"],
        branch=target["branch"],
        **({"match": spec["match"]} if "match" in spec else {}),
        provenance=dict(spec.get("provenance", {}), actual_prefix_generated=rawOutputs is not None),
    )


def renderPrompt(registry, api, before, question):
    class Captured(Exception):
        pass

    def capture(text, options):
        if options != {"add_special_tokens": False, "truncation": False}:
            raise ValueError("Unexpected native tokenizer options")
        raise Captured(text)

    tokenizer = {"sha256": api.pinnedTokenizerSha256, "encode": capture}
    try:
        api.buildModelInput(
            registry, before, question, dict(contextWindow=32768, completionTokens=512, tokenizer=tokenizer)
        )
    except Captured as prompt:
        return prompt.args[0]
    raise ValueError("Model-input builder did not produce a prompt")
