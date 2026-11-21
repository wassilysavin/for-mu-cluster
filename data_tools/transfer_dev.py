"""Replay and score the frozen semantic-transfer development panels."""

import json
import os
import re
import sys
from pathlib import Path

from evaluation import api
from evaluation.actual_history import createActualHistoryHarness
from evaluation.functiongemma import encodeCalls
from evaluation.schema import canonical
from evaluation.session import applyModelOutput, recordIncompleteModelOutput, recordProviderFailure
from .common import check, compact, render, send, session, sha
from .transfer_contract import gradeTransfer, transferRegistry

PANELS = ("closed_loop", "challenge")
DATA_SHA = "9df60f65e30c8b0d54966b2a2076af1318810ac1cb3cfecd77457dc299c0ff7f"


def load(registry_path, bundle_path):
    source = json.loads(Path(registry_path).read_bytes())
    registry = transferRegistry(source)
    raw = Path(bundle_path).read_bytes()
    check(sha(raw) == DATA_SHA, "Fresh DEV asset differs from frozen version")
    data = json.loads(raw)
    check(data["schema"] == "semantic-transfer-dev-v1" and data["source_registry_sha256"] == source["fingerprint"])
    registry_id = registry.fingerprint()
    bundle_id = sha(
        canonical({"asset_sha256": DATA_SHA, "registry_sha256": registry_id, "version": data["schema"]})
    )
    ids, turns, plan = set(), {}, []
    for panel in PANELS:
        for row in data["panels"][panel]:
            cid = row["case_id"]
            check(cid not in ids and row["panel"] == panel and row["split"] == "dev")
            check(row["family_id"] and row["bootstrap_family_id"] and row["question"].strip())
            check(row["cohort"] in ("supported", "outside_scope") and type(row["reference_eligible"]) is bool)
            dialogue = row.get("dialogue_id")
            if dialogue:
                check(not row.get("initial_prefix") and panel == "closed_loop")
                check(row["turn_index"] == turns.get(dialogue, 0) + 1)
                turns[dialogue] = row["turn_index"]
            else:
                check(row["turn_index"] == 1 and bool(row.get("initial_prefix")) == (panel == "challenge"))
            ids.add(cid)
            plan.append(row)
    check(len(turns) == 24 and set(turns.values()) == {4} and len(ids) == 233)
    return registry, data, plan, bundle_id, registry_id


def score(harness, registry, row, before, after, resolved):
    frozen = harness.grade(registry, row, before, after, resolved)
    result = gradeTransfer(frozen, resolved, before, after)
    return {**result, "standalone": False} if row.get("initial_prefix") else result


def materialize(registry, before, question, generation):
    if generation.get("error"):
        check(generation["error"] in ("context_overflow", "inference_failed"))
        return recordProviderFailure(registry, before, question, generation["error"])
    ids = generation["raw_ids"]
    check(isinstance(ids, list) and 0 < len(ids) <= 512 and all(type(i) is int and 0 <= i < 262144 for i in ids))
    check(isinstance(generation["raw_text"], str) and 50 not in ids[:-1])
    if ids[-1] == 50:
        return applyModelOutput(registry, before, question, generation["raw_text"])
    check(len(ids) == 512)
    return recordIncompleteModelOutput(registry, before, question, generation["raw_text"], ids, "token_limit")


def selftest(reg, plan, harness):
    sessions = {}
    counters = dict.fromkeys(
        (
            "alternatives",
            "unavailable_answers",
            "retry_after_error",
            "retry_after_success",
            "wrong_scope",
            "clarified",
            "failed_live_predecessors",
            "retry_quoted_instead",
            "partial_compound_answers",
            "necessary_clarifications",
            "unnecessary_clarifications",
            "ambiguous_guesses_retained",
        ),
        0,
    )
    for row in plan:
        did = row.get("dialogue_id")
        before = sessions[did] if did in sessions else harness.initial(reg, row)
        resolved = harness.resolve(row, before)
        question = resolved["question"]
        policy = row["policy"]
        kind = policy["kind"]
        challenge = row["panel"] == "challenge"

        def grade(calls):
            after = applyModelOutput(reg, before, question, encodeCalls(calls))
            return score(harness, reg, row, before, after, resolved), after

        for i, calls in enumerate(resolved["allowed_batches"]):
            verdict, after = grade(calls)
            check(verdict["success"] is True, row["case_id"])
            if i == 0 and did:
                sessions[did] = after
            counters["alternatives"] += 1
        if kind == "retry":
            counters["retry_after_error" if resolved["branch"] == "failed" else "retry_after_success"] += 1
        if kind == "ambiguous":
            counters["clarified"] += 1
        if challenge and kind == "ambiguous" and policy["object_kind"] == "answer":
            target = before["semantic"]["turns"][0]["answers"][0]["id"]
            verdict, _ = grade([{"name": "repeat_answer", "arguments": {"target": target}}])
            check(
                not verdict["success"]
                and verdict["failure_category"] == "semantic_unresolved"
                and verdict["critical_status"] == "review_required"
            )
            counters["ambiguous_guesses_retained"] += 1
        if challenge and kind == "repeat_answer":
            v, _ = grade([{"name": "clarify_reference", "arguments": {"kind": "answer"}}])
            check(v["critical_status"] == "none_observed")
            absent = resolved["branch"] == "absent"
            check(
                v["success"] is absent
                and v["failure_category"] == (None if absent else "unnecessary_clarification")
            )
            counters["necessary_clarifications" if absent else "unnecessary_clarifications"] += 1
            selected = before["semantic"]["turns"][-1]["answers"][0]["id"]
            if any(selected in refs for refs in resolved["reference_target_alternatives"]):
                continue
            verdict, _ = grade([{"name": "repeat_answer", "arguments": {"target": selected}}])
            check(not verdict["success"] and verdict["critical_category"] == "wrong_reference", row["case_id"])
            counters["unavailable_answers" if absent else "wrong_scope"] += 1
        if challenge and kind == "retry":
            ex = before["exchanges"][policy["target_exchange"] - 1]
            v, _ = grade([{"name": "repeat_exchange", "arguments": {"target": ex["id"]}}])
            check(not v["success"])
            counters["retry_quoted_instead"] += 1
        if did and row["turn_index"] == 1 and len(policy.get("calls") or []) == 2:
            v, _ = grade(policy["calls"][:1])
            check(not v["success"])
            counters["partial_compound_answers"] += 1
        if (
            did
            and row["turn_index"] > 1
            and kind in ("retry", "quote", "repeat_answer", "pending_cancel", "pending_resolve", "replace")
        ):
            failed = session(reg, "failed-prefix-control")
            for ex in before["exchanges"][: row["turn_index"] - 1]:
                failed = applyModelOutput(reg, failed, ex["question"], "invalid output")
            actual = harness.resolve(row, failed)
            after = applyModelOutput(reg, failed, actual["question"], encodeCalls(actual["allowed_batches"][0]))
            check(score(harness, reg, row, failed, after, actual)["success"])
            counters["failed_live_predecessors"] += 1
    check(all(counters.values()))
    return counters


def saved_prefix(path):
    if not path.exists():
        return []
    raw = path.read_bytes()
    boundary = raw.rfind(b"\n") + 1
    saved = [json.loads(r) for r in raw[:boundary].splitlines()]
    if boundary != len(raw):
        with path.open("r+b") as out:
            try:
                saved.append(json.loads(raw[boundary:]))
            except (json.JSONDecodeError, UnicodeDecodeError):
                out.truncate(boundary)
            else:
                out.seek(0, os.SEEK_END)
                out.write(b"\n")
            out.flush()
            os.fsync(out.fileno())
    return saved


def evaluate(context, path, reserve, binding, harness):
    from eval_tools.diagnose_data import compactObservation

    reg, data, plan, bundle_hash, _ = context
    saved = saved_prefix(path)
    check(len(saved) <= len(plan), "Fresh DEV receipts extend past the frozen panel")
    sessions, records = {}, []
    with path.open("ab") as out:
        for i, row in enumerate(plan):
            did = row.get("dialogue_id")
            before = sessions[did] if did in sessions else harness.initial(reg, row)
            resolved = harness.resolve(row, before)
            question = resolved["question"]
            prompt = render(reg, before, question)
            prompt_hash = sha(prompt)
            if i < len(saved):
                previous = saved[i]
                check(
                    previous["case_id"] == row["case_id"]
                    and previous["bundle_sha256"] == bundle_hash
                    and previous["prompt_sha256"] == prompt_hash
                )
                generation = previous["generation"]
            else:
                send({"kind": "generate", "case_id": row["case_id"], "prompt": prompt})
                reply = sys.stdin.readline()
                check(reply, "Model provider disconnected; saved prefix can be resumed")
                generation = json.loads(reply)
            if binding:
                check(
                    generation.get("diagnostic_worker_sha256") == binding, "Provider or checkpoint identity changed"
                )
            after = materialize(reg, before, question, generation)
            grade = score(harness, reg, row, before, after, resolved)
            receipt = compactObservation(
                {
                    **grade,
                    "bundle_sha256": bundle_hash,
                    "prompt_sha256": prompt_hash,
                    "before_visible_history_sha256": resolved["before_visible_history_sha256"],
                    "question": question,
                    "policy_branch": resolved["branch"],
                    "expected_model_call_batches": resolved["allowed_batches"],
                    "generation": generation,
                    "before": before,
                    "after": after,
                }
            )
            if i < len(saved):
                check(canonical(previous) == canonical(receipt), "Fresh DEV replay differs from saved observation")
            else:
                raw = (compact(receipt) + "\n").encode("utf-8")
                disk = os.statvfs(path.parent)
                check(disk.f_bavail * disk.f_frsize >= reserve + len(raw), "Fresh DEV disk reserve exhausted")
                out.write(raw)
                out.flush()
                os.fsync(out.fileno())
            if did:
                sessions[did] = after
            records.append(receipt)
    panels = {}
    for panel in PANELS:
        scored = [r for r in records if r["panel"] == panel]
        summary = harness.summarize(data["panels"][panel], scored)
        available = [r for r in scored if r["reference_eligible"] and r["resolved_reference_available"]]
        passed = sum(bool(r["reference_success"]) for r in available)
        summary["exact_reference_when_available"] = {
            "planned": len(available),
            "passed": passed,
            "accuracy": passed / len(available) if available else None,
        }
        panels[panel] = summary
    panels["challenge"]["standalone"] = {"planned": 0, "passed": 0, "accuracy": None}
    summary = panels["closed_loop"]
    failed = [r["case_id"] for r in records if r["generation"].get("error") == "inference_failed"]
    send(
        {
            "kind": "result",
            "score": {
                **summary,
                "summary": summary,
                "panels": panels,
                "observed_records": len(records),
                "bundle_sha256": bundle_hash,
                "infrastructure_failures": len(failed),
                "infrastructure_failure_case_ids": failed,
                "quality_valid": not failed,
                "review_status": "automatic_conservative_unresolved_retained",
                "release_ready": False,
            },
        }
    )


def main(args):
    check(
        3 <= len(args) <= 5,
        "Expected registry, DEV asset, output or --check/--selftest, optional binding and reserve",
    )
    registry_path, data_path, output = args[:3]
    context = load(registry_path, data_path)
    registry, data, plan, bundle_hash, registry_hash = context
    header = {
        "cases": len(plan),
        "bundle_sha256": bundle_hash,
        "registry_sha256": registry_hash,
        "panels": {p: len(data["panels"][p]) for p in PANELS},
    }
    if output == "--check":
        send({"kind": "check", **header})
        return
    harness = createActualHistoryHarness(api())
    if output == "--selftest":
        send({"kind": "selftest", **header, **selftest(registry, plan, harness)})
        return
    expected = json.loads(args[3]) if len(args) > 3 else {}
    reserve = int(args[4]) if len(args) > 4 else 0
    check(not expected.get("bundle_sha256") or expected["bundle_sha256"] == bundle_hash)
    binding = expected.get("diagnostic_worker_sha256")
    check(reserve >= 0 and (binding is None or re.fullmatch("[0-9a-f]{64}", binding)))
    evaluate(context, Path(output), reserve, binding, harness)


if __name__ == "__main__":
    main(sys.argv[1:])
