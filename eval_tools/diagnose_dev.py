"""Shard, grade, audit and expand the actual-history DEV panel from saved receipts."""

from contextlib import nullcontext
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

from evaluation import api
from evaluation.actual_history import createActualHistoryHarness
from evaluation.history_train import renderPrompt
from evaluation.schema import canonical

from .diagnose_data import compactObservation, compactObservationSchema, materializeGeneration
from .io import digest, emit, read_saved, receive, records, registry_at, require, save


PANELS = ("fixed_prefix", "closed_loop", "challenge")
BINDING = re.compile("[a-f0-9]{64}")
MIN_RESERVE = 536870912


def shardOf(row, count=4):
    key = "dialogue:" + row["dialogue_id"] if row.get("dialogue_id") else "case:" + row["case_id"]
    return int(digest(row["panel"] + ":" + key)[:8], 16) % count


def load_bundle(registry_path, bundle_path, panels, review=True):
    registry, source = registry_at(registry_path)
    data = json.loads(Path(bundle_path).read_bytes())
    require(data["schema"] == "actual-history-v3-dev-1" and data["registry_sha256"] == source["fingerprint"],
            "DEV schema or registry binding differs")
    fingerprint = digest({**data, "review": None})
    if review:
        verdict = data["review"]
        require(verdict["status"] == "approved" and verdict["reviewer"] != data["author"]
                and verdict["bundle_sha256"] == fingerprint, "DEV changed after independent review")
    plan = [row for panel in panels for row in data["panels"][panel]]
    require(len({r["case_id"] for r in plan}) == len(plan), "Duplicate DEV case")
    for panel in panels:
        turns = {}
        for row in data["panels"][panel]:
            require(row["panel"] == panel and row["split"] == "dev", "Unexpected panel or split")
            if row.get("dialogue_id"):
                key = row["dialogue_id"]
                require(row["turn_index"] == turns.get(key, 0) + 1, "Disordered DEV dialogue")
                turns[key] = row["turn_index"]
        require(all(n == (8 if panel == "challenge" else 5) for n in turns.values()), "Incomplete DEV dialogue")
    return SimpleNamespace(registry=registry, data=data, plan=plan, bundleHash=fingerprint)


def loadPanel(registryPath, bundlePath):
    context = load_bundle(registryPath, bundlePath, PANELS)
    require(len(context.plan) == 1250, "Incomplete actual-history panel")
    return context


def summarize(data, records):
    harness = createActualHistoryHarness(api())
    scores = {panel: harness.summarize(data["panels"][panel], [r for r in records if r["panel"] == panel])
              for panel in PANELS}
    return {"panels": scores, "summary": scores["closed_loop"], "observed_records": len(records),
            "review_status": "automatic_conservative_unresolved_retained", "release_ready": False}


def full_observation(context, harness, row, before, resolved, prompt, generation, materialize=materializeGeneration):
    after = materialize(context.registry, before, resolved["question"], generation)
    grade = harness.grade(context.registry, row, before, after, resolved)
    return {**grade, "bundle_sha256": context.bundleHash, "prompt_sha256": digest(prompt),
            "before_visible_history_sha256": resolved["before_visible_history_sha256"], "question": resolved["question"],
            "policy_branch": resolved["branch"], "expected_model_call_batches": resolved["allowed_batches"],
            "generation": generation, "before": before, "after": after}


def expandRecords(context, records, options=None):
    binding = (options or {}).get("binding")
    positions = {r["case_id"]: i for i, r in enumerate(context.plan)}
    sessions, turns = {}, {}
    last = -1
    core = api()
    harness = createActualHistoryHarness(core)
    for record in records:
        require(record["observation_schema"] == compactObservationSchema, "Expected compact diagnostic receipt")
        index = positions.get(record["case_id"], -1)
        require(index > last, "DEV receipts contain an unknown, duplicate or reordered case")
        last = index
        row = context.plan[index]
        key = (row["panel"], row["dialogue_id"]) if row.get("dialogue_id") else None
        if key:
            require(row["turn_index"] == turns.get(key, 0) + 1, "Replay requires the preceding dialogue receipts")
        before = sessions.get(key) or harness.initial(context.registry, row)
        resolved = harness.resolve(row, before)
        prompt = renderPrompt(context.registry, core, before, resolved["question"])
        worker = record["generation"].get("diagnostic_worker_sha256") or ""
        require(BINDING.fullmatch(worker), "Incomplete worker binding")
        require(binding is None or binding == worker, "Receipt belongs to another provider")
        full = full_observation(context, harness, row, before, resolved, prompt, record["generation"])
        require(canonical(compactObservation(full)) == canonical(record), "Compact DEV receipt does not reproduce")
        if key:
            sessions[key], turns[key] = full["after"], row["turn_index"]
        yield full


def _evaluate(context, output, shard, count, reserve, binding, relay):
    selected = [row for row in context.plan if shardOf(row, count) == shard]
    saved = read_saved(output)
    require(len(saved) <= len(selected), "Extra saved DEV observations")
    sessions = {}
    core = api()
    harness = createActualHistoryHarness(core)
    with nullcontext() if relay else Path(output).open("ab") as handle:
        for index, row in enumerate(selected):
            before = sessions.get(row.get("dialogue_id")) or harness.initial(context.registry, row)
            resolved = harness.resolve(row, before)
            prompt = renderPrompt(context.registry, core, before, resolved["question"])
            previous = saved[index] if index < len(saved) else None
            if previous is not None:
                require(previous["case_id"] == row["case_id"] and previous["bundle_sha256"] == context.bundleHash
                        and previous["prompt_sha256"] == digest(prompt), "DEV resume binding mismatch")
                generation = previous["generation"]
            else:
                emit({"kind": "generate", "case_id": row["case_id"], "prompt": prompt})
                generation = receive()
            require(generation["diagnostic_worker_sha256"] == binding, "DEV generation belongs to a different worker")
            full = full_observation(context, harness, row, before, resolved, prompt, generation)
            record = compactObservation(full)
            if previous is not None:
                require(canonical(previous) == canonical(record), "Saved DEV is not reproducible")
            elif relay:
                emit({"kind": "persist", "observation": record})
                ack = receive()
                require(ack["kind"] == "persisted" and ack["case_id"] == row["case_id"], "Invalid persistence acknowledgement")
            else:
                save(handle, record, reserve, check_space=True)
            if row.get("dialogue_id"):
                sessions[row["dialogue_id"]] = full["after"]
    emit({"kind": "result", "score": {"observed_records": len(selected), "shard": shard, "shards": count,
                                     "bundle_sha256": context.bundleHash}})


def main(registry_path, bundle_path, output, *args):
    context = loadPanel(registry_path, bundle_path)
    if output == "--plan":
        fields = ("case_id", "panel", "dialogue_id", "family_id", "cohort")
        emit({"kind": "plan", "bundle_sha256": context.bundleHash,
              "rows": [{**{k: r[k] for k in fields if k in r}, "shard": shardOf(r)} for r in context.plan]})
    elif output == "--summarize":
        saved = read_saved(args[0])
        require([r["case_id"] for r in saved] == [r["case_id"] for r in context.plan]
                and all(r["bundle_sha256"] == context.bundleHash for r in saved), "Incomplete or reordered DEV results")
        emit({"kind": "result", "score": summarize(context.data, saved)})
    elif output in ("--audit", "--expand"):
        selected = args[1] if len(args) > 1 else None
        count = emitted = 0
        for obs in expandRecords(context, records(args[0])):
            count += 1
            if output == "--expand" and (selected is None or selected == obs["case_id"]):
                emit({"kind": "expanded", "observation": obs})
                emitted += 1
        require(output != "--expand" or selected is None or emitted == 1, "Requested case is absent")
        emit({"kind": "audit", "observations": count, "bundle_sha256": context.bundleHash,
              "model_inference_performed": False, "independent_label_review_performed": False})
    else:
        shard_text, count_text, reserve_text, binding, *extra = args
        shard, count, reserve = int(shard_text), int(count_text), int(reserve_text)
        require(count == 4 and 0 <= shard < count and MIN_RESERVE <= reserve <= 2**53 - 1 and BINDING.fullmatch(binding)
                and extra in ([], ["--relay-persist"]), "Invalid sharded evaluation arguments")
        _evaluate(context, output, shard, count, reserve, binding, bool(extra))


if __name__ == "__main__":
    main(*sys.argv[1:])
