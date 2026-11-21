"""Reconstruct the earliest legal exchange order that preserves the displayed content."""

from .history_train import continuation, renderPolicy
from .schema import canonical

REFS = {"target", "_replace", "_resolve", "_supersede"}


def _remap(calls, mapping):
    return [
        dict(
            c,
            arguments={
                k: mapping.get(v, v) if k in REFS and isinstance(v, str) else v for k, v in c["arguments"].items()
            },
        )
        for c in calls
    ]


class RejectedOrder(ValueError):
    pass


def _require(ok, reason):
    if not ok:
        raise RejectedOrder(reason)


def matchedControl(registry, api, spec, treatment):
    state = treatment["before"]
    exchanges, prefix = state["exchanges"], spec["prefix"]
    if len(exchanges) != len(prefix):
        raise ValueError("Observed prefix length differs")
    turns = {t["id"]: t for t in state["semantic"]["turns"]}
    owner = {}
    for i, ex in enumerate(exchanges):
        owner[ex["id"]] = i
        turn = turns.get(ex.get("semantic_turn_id"), {})
        for item in turn.get("answers", []):
            owner[item["id"]] = i
        for item in turn.get("response_parts", []):
            if item["kind"] == "clarification":
                owner[item["id"]] = i
        for item in turn.get("effects", []):
            if item["kind"] == "pending_created":
                owner[item["target"]] = i
        for item in state["semantic"]["pending"]:
            if turn and item["id"].startswith(turn["id"] + "."):
                owner[item["id"]] = i
    dependencies = []
    last_pending = None
    for i, (ex, source) in enumerate(zip(exchanges, prefix)):
        calls = ex.get("model_calls") or []
        deps = {
            owner[v]
            for c in calls
            for k, v in c["arguments"].items()
            if k in REFS and isinstance(v, str) and v in owner and owner[v] < i
        }
        if source.get("address_policy"):
            deps.add(source["address_policy"]["exchange_index"] - 1)
        if any(
            c["name"] == "cancel_pending" or {"_clarify", "_resolve", "_supersede"} & c["arguments"].keys()
            for c in calls
        ):
            if last_pending is not None:
                deps.add(last_pending)
            last_pending = i
        dependencies.append(deps)
    original = tuple(range(len(exchanges)))

    def orders(prefix):
        if len(prefix) == len(original):
            yield prefix
        for i in original:
            if i not in prefix and dependencies[i].issubset(prefix):
                yield from orders((*prefix, i))

    wanted = spec["continuation"].get("exchange_index")

    def priority(order):
        inversions = sum(a > b for i, a in enumerate(order) for b in order[i + 1 :])
        return order.index(wanted - 1), inversions, ",".join(map(str, order))

    candidates = sorted(orders(()), key=priority) if wanted else [original]
    rejected = []
    for order in candidates:
        try:
            result = _reconstruct(registry, api, spec, treatment, order)
        except RejectedOrder as e:
            rejected.append(dict(exchange_order=[exchanges[i]["id"] for i in order], reason=str(e)))
            continue
        result["match"].update(
            rejected_reorders=len(rejected),
            reconstruction_fallback="earliest_content_preserving_legal_order" if rejected else None,
            rejection_examples=rejected[:3],
        )
        return result
    raise ValueError(f"{spec['case_id']}: no content-preserving reconstruction: {rejected}")


def _reconstruct(registry, api, spec, treatment, order):
    original = treatment["before"]
    exchanges = original["exchanges"]
    positions = {old + 1: new + 1 for new, old in enumerate(order)}
    mapping = {}
    before = api.createSession(registry, dict(dialogueVersion=2, sessionId=spec["session_id"]))
    cid = spec["case_id"]
    for i in order:
        old, source = exchanges[i], spec["prefix"][i]
        question = old["question"]
        if policy := source.get("address_policy"):
            question = renderPolicy(policy, positions[policy["exchange_index"]])["question"]
        if old["outcome"] == "provider_error":
            before = api.recordProviderFailure(registry, before, question, old["error_code"])
        elif old.get("error_code") == "incomplete_model_output":
            before = api.recordIncompleteModelOutput(
                registry,
                before,
                question,
                old["raw_output"],
                old["generation"]["token_ids"],
                old["generation"]["stop_reason"],
            )
        else:
            raw = (
                api.encodeCalls(_remap(old["model_calls"], mapping))
                if old["outcome"] == "committed"
                else old["raw_output"]
            )
            before = api.applyModelOutput(registry, before, question, raw)
        current = before["exchanges"][-1]
        for k, desc in (("outcome", "outcome"), ("error_code", "failure type"), ("assistant", "displayed content")):
            _require(current.get(k) == old.get(k), f"{cid}: reordered {desc}")
        mapping[old["id"]] = current["id"]
        if turn_id := old.get("semantic_turn_id"):
            old_turn = next(t for t in original["semantic"]["turns"] if t["id"] == turn_id)
            new_turn = next(t for t in before["semantic"]["turns"] if t["id"] == current["semantic_turn_id"])
            mapping[turn_id] = new_turn["id"]
            _require(len(old_turn["answers"]) == len(new_turn["answers"]), "Expected values to be strictly equal:")
            for old_a, new_a in zip(old_turn["answers"], new_turn["answers"]):
                mapping[old_a["id"]] = new_a["id"]
                _require(
                    canonical(old_a["call"]) == canonical(new_a["call"]),
                    "Scientific arguments changed during reconstruction",
                )
            for p in before["semantic"]["pending"]:
                if p["id"].startswith(new_turn["id"] + "."):
                    mapping[turn_id + p["id"][len(new_turn["id"]) :]] = p["id"]
    policy = spec["continuation"]
    if policy.get("exchange_index"):
        policy = renderPolicy(policy, positions[policy["exchange_index"]])
    gold = continuation(before, policy)
    pending = [
        dict(
            p,
            id=mapping[p["id"]],
            **({"revision_target": mapping[p["revision_target"]]} if p.get("revision_target") else {}),
        )
        for p in original["semantic"]["pending"]
    ]
    _require(
        canonical(pending) == canonical(before["semantic"]["pending"]),
        "Pending lifecycle changed during reconstruction",
    )
    _require(
        canonical(gold["calls"]) == canonical(_remap(treatment["calls"], mapping)),
        f"{cid}: matched continuation differs",
    )
    raw = api.encodeCalls(gold["calls"])
    after = api.applyModelOutput(registry, before, gold["question"], raw)
    _require(after["exchanges"][-1]["outcome"] == "committed", "Expected values to be strictly equal:")
    _require(
        after["exchanges"][-1]["assistant"] == treatment["after"]["exchanges"][-1]["assistant"],
        f"{cid}: matched continuation text differs",
    )
    return dict(
        case_id=treatment["case_id"] + "-control",
        paired_case_id=treatment["case_id"],
        family_id=treatment["family_id"],
        group=treatment["group"],
        split="train",
        question=gold["question"],
        before=before,
        actual_before=api.sessionHistory(before),
        calls=gold["calls"],
        target=raw,
        after=after,
        history_origin="reconstructed_earliest_legal_order_from_paired_history",
        branch=gold["branch"],
        reference_targets=gold["reference_targets"],
        match=dict(
            treatment.get("match", {}),
            identity_order_pair=all(i == n for n, i in enumerate(order)),
            exchange_order=[exchanges[i]["id"] for i in order],
            reference_mapping=mapping,
            exact_scientific_arguments_preserved=True,
            displayed_content_preserved=True,
            current_action_arity_preserved=True,
            pending_lifecycle_order_preserved=True,
        ),
        provenance=dict(
            treatment.get("provenance", {}),
            actual_prefix_generated=False,
            reconstructed_prefix=True,
            parent_history_origin=treatment["history_origin"],
            authoring="Prospective matched-control order reconstruction",
            estimator="position/order and corresponding reference wording; not pure provenance or literal old-data replay",
        ),
    )
