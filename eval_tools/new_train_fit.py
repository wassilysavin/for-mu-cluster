"""Grade frozen TRAIN continuations against the session actually shown to the model."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

from evaluation import api
from evaluation.actual_history import createActualHistoryHarness
from evaluation.functiongemma import decodeCalls, encodeCalls
from evaluation.history_train import continuation, materializeScenario, renderPrompt
from evaluation.registry import createRegistry
from evaluation.schema import canonical, ContractError
from evaluation.session import applyModelOutput, createSession
from data_tools.transfer_contract import gradeTransfer, transferRegistry, upgradePacked
from .diagnose_data import createTrainDiagnostics, materializeGeneration, verify_alternatives
from .io import digest, emit, records, require


USER = "<end_of_turn>\n<start_of_turn>user\n"
END = "<end_of_turn>\n<start_of_turn>model\n"
CONTRACT = '\n\n{"format":"gemprf-shared-tool-contract",'


def _insert(table, key, value):
    require(isinstance(key, str) and key not in table, f"Duplicate or missing TRAIN identifier: {key}")
    table[key] = value


def _payload(prompt):
    require(prompt.count(USER) == 1 and prompt.endswith(END), "Unexpected frozen TRAIN prompt boundaries")
    return json.loads(prompt.split(USER)[1][:-len(END)])


def _shared_prompt(original, bundle, cache):
    require(original.count(CONTRACT) == 1, "Expected exactly one packed shared tool contract")
    prefix, rest = original.split(CONTRACT)
    require(rest.count(USER) == 1 and rest.endswith(END), "Unexpected shared TRAIN prompt boundaries")
    tools, payload = rest.split(USER)
    if tools not in cache:
        packed = json.loads(CONTRACT[2:] + tools)
        cache[tools] = upgradePacked(packed, bundle).replace("<", r"\u003c")
    return prefix + "\n\n" + cache[tools] + USER + payload


def createScorer(bundle):
    original = createRegistry(bundle["knowledge"], bundle["contracts"])
    original_id = original.fingerprint()
    require(original_id == bundle["fingerprint"], "Original registry fingerprint changed")
    registry = transferRegistry(bundle)
    revised_id = registry.fingerprint()
    core = api()
    harness = createActualHistoryHarness(core)
    diagnostics = [createTrainDiagnostics(registry), createTrainDiagnostics(original)]
    cases, contracts = {}, {}

    def prepare(request, row, before, options):
        case_id, prompt, target = (request[key] for key in ("case_id", "prompt", "target"))
        section, branch = options["section"], options["branch"]
        migrated = bool(options.get("migrated"))
        old_registry = original if migrated else registry
        old_before = {**before, "registry_id": original_id} if migrated else before
        revised_before = {**before, "registry_id": revised_id}
        require(target == encodeCalls(row["calls"]), f"Frozen TRAIN target differs: {case_id}")
        historical = renderPrompt(old_registry, core, old_before, row["question"])
        if section == "shared":
            expected = _shared_prompt(historical, bundle, contracts)
        elif migrated:
            expected = renderPrompt(registry, core, revised_before, row["question"])
        else:
            expected = historical
        require(prompt == expected, f"Frozen TRAIN prompt differs: {case_id}")
        item = diagnostics[migrated].prepare({"source": "common" if section == "shared" else "history",
                                              "row": row, "before": old_before, "prompt": historical})
        require(item["semantic_eligible"] and item["allowed_batches"], f"Missing TRAIN oracle: {case_id}")
        if migrated:
            after = applyModelOutput(registry, revised_before, row["question"], target)
            require(after["exchanges"][-1]["outcome"] == "committed", "Updated gold program failed")
            verify_alternatives(registry, revised_before, row["question"], after, item["allowed_batches"][1:])
        meta = {"case_id": case_id, "section": section, "group": row["group"], "cohort": item["cohort"],
                "prompt_sha256": digest(prompt), "target_sha256": digest(target)}
        require(case_id not in cases or cases[case_id]["meta"] == meta, "Inconsistent repeated TRAIN row")
        targets = harness.refs(row["calls"])
        resolved = {"question": row["question"], "branch": "absent" if branch == "requested_answer_missing" else branch,
                    "allowed_batches": item["allowed_batches"], "reference_targets": targets,
                    "reference_target_alternatives": list(map(harness.refs, item["allowed_batches"]))}
        cases[case_id] = {"meta": meta, "before": revised_before, "target": target, "calls": row["calls"], "resolved": resolved}
        return {"kind": "prepared", **meta}

    def score(case_id, generation):
        require(case_id in cases, f"Case was never prepared: {case_id}")
        item = cases[case_id]
        meta, resolved, before = (item[key] for key in ("meta", "resolved", "before"))
        require(generation["provider_kind"] == "functiongemma" and not generation.get("error"), "TRAIN inference failure")
        require(generation["input_sha256"] == meta["prompt_sha256"], "Generated against a different TRAIN input")
        ids = generation["raw_ids"]
        require(isinstance(generation["raw_text"], str) and isinstance(ids, list) and len(ids) <= 512,
                "Invalid generation payload")
        complete = bool(ids and ids[-1] == 50)
        require(generation["complete"] is complete and generation["finish_reason"] == ("complete_batch" if complete else "length"),
                "Invalid generation termination")
        after = materializeGeneration(registry, before, resolved["question"], generation)
        result = harness.grade(registry, {**meta, "family_id": case_id, "panel": "train_fixed", "policy": {"kind": "fresh"},
                                          "turn_index": None, "reference_eligible": bool(resolved["reference_targets"])},
                               before, after, resolved)
        grade = gradeTransfer(result, resolved, before, after)
        calls = None
        if complete:
            try:
                calls = decodeCalls(generation["raw_text"])
            except (ContractError, ValueError):
                pass
        return {"kind": "scored", "observation": {**meta, **{k: grade[k] for k in (
            "success", "critical_status", "critical_category", "failure_category") if k in grade},
            "native_literal_exact": complete and generation["raw_text"] == item["target"],
            "canonical_call_exact": calls is not None and canonical(calls) == canonical(item["calls"]),
            "transport_valid": calls is not None}}

    return SimpleNamespace(original=original, registry=registry, prepare=prepare, score=score)


def _load_sources(registry_path, init):
    from data_tools.transfer_train import observedSession
    assets = Path(registry_path).parent
    scorer = createScorer(json.loads(Path(registry_path).read_bytes()))
    shared, contexts, roots, cases, observed, specs, states = {}, {}, {}, {}, {}, {}, {}
    for row in records(assets / "train-full-v2.jsonl"):
        key = "shared:" + row["case_id"]
        _insert(shared, key, {"case_id": key, "group": "shared_replay", "question": row["question"], "calls": row["calls"]})
    for row in records(init.get("contexts", assets / "diagnostic-train-contexts.jsonl")):
        _insert(contexts, "shared:" + row["case_id"], row["before"])
    for row in records(assets / "transfer-train-roots.jsonl"):
        _insert(roots, row["root_id"], row)
    observed_ids = {r["observed_prefix_id"] for r in roots.values() if r.get("observed_prefix_id")}
    spec_file = assets / "history-train-specs.jsonl"
    for row in records(spec_file):
        key = row["prefix_id"]
        if key in observed_ids:
            if key in specs:
                require(row["prefix"] == specs[key]["prefix"] and row["session_id"] == specs[key]["session_id"],
                        "Inconsistent repeated prefix")
            else:
                specs[key] = row
    specs_digest = digest(spec_file.read_bytes())
    for row in records(init["receipts"]):
        if row["prefix_id"] in observed_ids:
            _insert(observed, row["prefix_id"], row)
    require(len(observed) == len(observed_ids), "Real 7205 TRAIN prefix receipts are incomplete")
    for arm in ("ordinary", "contrast"):
        for row in records(assets / f"transfer-train-{arm}.jsonl"):
            _insert(cases, row["case_id"], row)

    def prepare(request):
        key = request["case_id"]
        if key in shared:
            data = _payload(request["prompt"])
            before = contexts.get(key) or createSession(scorer.original, {"sessionId": data["history"]["session_id"],
                                               "dialogueVersion": 2 if data["format"].endswith("v2") else 1})
            return scorer.prepare(request, shared[key], before, {"section": "shared", "migrated": True, "branch": "frozen_train"})
        require(key in cases, f"Unknown TRAIN row: {key}")
        row = cases[key]
        root = roots[row["root_id"]]
        old = root.get("observed_prefix_id")
        if root["root_id"] not in states:
            if old:
                state = observedSession(scorer.registry, scorer.original, observed[old], root, specs[old], specs_digest)
            else:
                state = materializeScenario(scorer.registry, api(), {**root, "case_id": key, "session_id": root["root_id"],
                                                                    "continuation": row["continuation"]})["before"]
            states[root["root_id"]] = state
        before = states[root["root_id"]]
        target = continuation(before, row["continuation"])
        return scorer.prepare(request, {"case_id": key, "group": root["group"], **target}, before,
                              {"section": "variable", "migrated": bool(old), "branch": target["branch"]})

    return SimpleNamespace(prepare=prepare, score=scorer.score)


def main(path):
    server = None
    for line in sys.stdin:
        request = json.loads(line)
        action = request["action"]
        if action == "init" and server is None:
            server = _load_sources(path, request)
            emit({"kind": "initialized"})
        elif server is not None and action == "prepare":
            emit(server.prepare(request))
        elif server is not None and action == "score":
            emit(server.score(request["case_id"], request["generation"]))
        else:
            raise ValueError(f"Unexpected TRAIN-fit RPC action: {action}")


if __name__ == "__main__":
    main(*sys.argv[1:])
