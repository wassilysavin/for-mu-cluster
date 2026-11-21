"""Materialize paired TRAIN decisions from authenticated displayed histories."""

from pathlib import Path

from .common import ASSETS, check, read_json, render, rows, runtime, send, session, sha
from .transfer_contract import transferRegistry


def observedSession(registry, original, record, root, spec, specsDigest):
    from evaluation.history_train import applyObservation

    check(record["specs_sha256"] == specsDigest and record["registry_sha256"] == original.fingerprint())
    check(record["model_fingerprint"]["checkpoint_step"] == 7205)
    prefix = root["prefix"]
    check(len(record["observations"]) == len(prefix))
    check([s["question"] for s in prefix] == [s["question"] for s in spec["prefix"]])
    before = session(original, spec["session_id"])
    for step, observed in zip(prefix, record["observations"]):
        question = step["question"]
        check(
            observed["question"] == question
            and observed["model_input_sha256"] == sha(render(original, before, question))
        )
        before = applyObservation(original, runtime(), before, question, observed)
    check(before == record["actual_session"], "Saved TRAIN dialogue differs from original native replay")
    return {**before, "registry_id": registry.fingerprint()}


def authored_session(registry, root):
    from evaluation.history_train import continuation

    api = runtime()
    before = session(registry, root["root_id"])
    for step in root["prefix"]:
        question = step["question"]
        if step.get("kind") == "provider_error":
            before = api.recordProviderFailure(registry, before, question, step["error_code"])
            continue
        malformed = step.get("kind") == "malformed"
        if malformed:
            raw = step["raw_output"]
        else:
            calls = continuation(before, step["policy"])["calls"] if step.get("policy") else step["calls"]
            raw = api.encodeCalls(calls)
        before = api.applyModelOutput(registry, before, question, raw)
        check(before["exchanges"][-1]["outcome"] == ("error" if malformed else "committed"))
    return before


def check_meaning(policy, before, calls):
    kind = policy["kind"]
    if kind == "fresh":
        check(calls == policy["calls"])
    elif kind == "clarify_reference":
        check(sum(e["outcome"] == "committed" for e in before["exchanges"]) >= 2)
        check(calls == [{"name": "clarify_reference", "arguments": {"kind": "answer"}}])
    else:
        exchange = before["exchanges"][policy["exchange_index"] - 1]
        if kind == "answer_again":
            retry = [{"name": "retry_exchange", "arguments": {"target": exchange["id"]}}]
            check(calls == (retry if exchange["outcome"] in ("error", "provider_error") else []) + policy["calls"])
        elif kind == "quote_exchange":
            check(calls == [{"name": "repeat_exchange", "arguments": {"target": exchange["id"]}}])
        elif kind in ("quote_answer", "revise_answer"):
            turn = next(
                (t for t in before["semantic"]["turns"] if t["id"] == exchange.get("semantic_turn_id")), None
            )
            part = policy["part"] - 1
            answer = turn["answers"][part] if turn and 0 <= part < len(turn["answers"]) else None
            if kind == "revise_answer":
                check(answer and answer["call"] == policy["expected_original"])
                check(
                    calls
                    == [
                        {
                            "name": "change_" + answer["call"]["name"],
                            "arguments": {"target": answer["id"], "changes": policy["changes"]},
                        }
                    ]
                )
                check({**answer["call"]["arguments"], **policy["changes"]} == policy["replacement"]["arguments"])
            else:
                expected = (
                    {"name": "repeat_answer", "arguments": {"target": answer["id"]}}
                    if answer
                    else {"name": "clarify_reference", "arguments": {"kind": "answer"}}
                )
                check(calls == [expected])


def materialize(options):
    from evaluation.registry import createRegistry
    from evaluation.history_train import continuation

    receipts = options.get("receipts")
    emit = options.get("emit", lambda row: None)
    source = read_json("registry.json")
    registry = transferRegistry(source)
    original = createRegistry(source["knowledge"], source["contracts"])
    roots = list(rows(ASSETS / "transfer-train-roots.jsonl"))
    ids = {r["observed_prefix_id"] for r in roots if r.get("observed_prefix_id")}
    specs = ASSETS / "history-train-specs.jsonl"
    by_prefix = {r["prefix_id"]: r for r in rows(specs)}
    digest = sha(specs.read_bytes())
    selected = {}
    if receipts:
        for record in rows(Path(receipts)):
            key = record["prefix_id"]
            if key in ids:
                check(key not in selected)
                selected[key] = record
        check(selected.keys() == ids)
    else:
        check(options.get("fixture"), "Real TRAIN requires the original 7205 prefix receipts")
    prefixes = {}
    for root in roots:
        key = root.get("observed_prefix_id")
        spec = by_prefix[key] if key else None
        observed = bool(spec and receipts)
        before = (
            observedSession(registry, original, selected[key], root, spec, digest)
            if observed
            else authored_session(registry, {**root, "prefix": spec["prefix"]} if spec else root)
        )
        prefixes[root["root_id"]] = before, root["group"], observed
    api = runtime()
    for arm in ("ordinary", "contrast"):
        for row in rows(ASSETS / f"transfer-train-{arm}.jsonl"):
            before, group, observed = prefixes[row["root_id"]]
            decision = continuation(before, row["continuation"])
            question, calls = decision["question"], decision["calls"]
            check_meaning(row["continuation"], before, calls)
            target = api.encodeCalls(calls)
            after = api.applyModelOutput(registry, before, question, target)
            check(after["exchanges"][-1]["outcome"] == "committed", row["case_id"])
            check(after["exchanges"][:-1] == before["exchanges"])
            emit(
                {
                    "arm": arm,
                    "case_id": row["case_id"],
                    "root_id": row["root_id"],
                    "group": group,
                    "question": question,
                    "calls": calls,
                    "branch": decision["branch"],
                    "observed": observed,
                    "model_input": render(registry, before, question),
                    "target": target,
                }
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fixture", action="store_true")
    group.add_argument("receipts", nargs="?")
    materialize({**vars(parser.parse_args()), "emit": send})
