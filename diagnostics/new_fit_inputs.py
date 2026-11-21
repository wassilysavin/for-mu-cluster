"""Locate the exact TRAIN rows, saved weights and semantic evidence of a completed transfer pair."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from history import checkpoint_header
from ledger import file_record
from diagnostics.saved_transfer import ARMS, ROOT, RUN, expect, load_saved_transfer
from diagnostics.student_evidence import SOURCE, checked_path, digest, read, verify
from transfer_train import EXCLUDED_SHARED, FROZEN, SEEDS, arrangements, rows

CONTEXT_MANIFEST_SHA = "b181085faba8cdb73bcf602e83c0eb0842f52496b4d85b29f9e584804add8aa3"
PREFIX_RECEIPTS = SOURCE / "collection/prefix-receipts.jsonl"


def pinned(root, relative, expected):
    path = checked_path(root, root / relative)
    record = file_record(path, relative_to=root)
    expect(record["sha256"] == expected, f"Frozen TRAIN source changed: {relative}")
    return record


def schedule_index(path, common_ids, declarations, labels, seeds, receipt):
    positions = arrangements(common_ids, declarations)
    schedules = {f"{arm}-{seed}": [] for seed in seeds for arm in ARMS}
    groups = {name: Counter() for name in schedules}
    tokens = {name: Counter() for name in schedules}
    seen = set()
    fingerprint = sha256()
    with path.open("rb") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            fingerprint.update(line)
            row = json.loads(line)
            case = row["case_id"]
            expect(case in positions and case not in seen and row["positions"] == positions[case],
                   f"Unexpected TRAIN position or repeated case: {case}")
            seen.add(case)
            ids, prompt = row["input_ids"], row["prompt_tokens"]
            expect(type(prompt) is int and 0 < prompt <= 32768 - 512 and isinstance(ids, list)
                   and 0 < len(ids) - prompt <= 512 and ids[-1] == 50
                   and all(type(i) is int and 0 <= i < 262144 for i in ids), f"Invalid TRAIN tokens: {case}")
            label = labels[case] if case in labels else {"group": "shared_replay", "section": "shared"}
            meta = dict(label, case_id=case, offset=offset, row_sha256=sha256(line).hexdigest())
            for name, position in positions[case].items():
                if name in schedules:
                    schedules[name].append(dict(meta, position=position))
                    groups[name][label["group"]] += 1
                    tokens[name].update(prompt=prompt, target=len(ids) - prompt)
    expect(fingerprint.hexdigest() == receipt["sha256"] and seen == set(positions), "Prepared TRAIN differs")
    for name, records in schedules.items():
        records.sort(key=lambda r: r["position"])
        arm = name.split("-")[0]
        expect([r["position"] for r in records] == list(range(receipt["count"]))
               and groups[name] == receipt["presented"][arm]["groups"]
               and tokens[name] == receipt["token_balance"][arm], f"TRAIN exposure differs: {name}")
    return schedules


def arm_checkpoint(root, name, model, records, manifest):
    common = manifest["common"]
    identity_path = root / "runs" / f"semantic-transfer-v1-{name}" / "run.json"
    run = read(root, identity_path)
    identity = run["identity"]
    expected_train = dict(common["train"], rows=len(records), arm=name,
                          schedule_sha256=digest([r["case_id"] for r in records]))
    recipe = identity["recipe"]
    expect(digest(identity) == run["identity_sha256"] == manifest["identities"][name]
           and identity["train"] == expected_train and recipe["examples_per_update"] == 4
           and recipe["initial_step"] == 7991 and recipe["training"] == {"arm": name, "updates": 160}
           and recipe["milestones"] == [160], f"TRAIN update recipe or draw order changed: {name}")
    checkpoint = model["checkpoint"]
    path = verify(root, identity_path.parent / "checkpoint-step-8151.safetensors", checkpoint)
    header = checkpoint_header(path)["__metadata__"]
    lineage = {"identity_sha256": run["identity_sha256"],
               "initial_checkpoint_sha256": common["initial_checkpoint"]["sha256"]}
    expect(header["step"] == "8151" and json.loads(header["lineage"]) == lineage
           and header["format"] == "gemprf-torch-volta-fp32-checkpoint-v2", f"Checkpoint lineage differs: {name}")
    return {"checkpoint": checkpoint, "identity_sha256": run["identity_sha256"],
            "schedule_sha256": expected_train["schedule_sha256"],
            "fresh_dev": model["scores"]["fresh"]["summary"], "cases": records}


def load_inputs(root=ROOT):
    root = Path(root).resolve(strict=True)
    manifest, result, _, _ = load_saved_transfer(root)
    common = manifest["common"]
    receipt_path = verify(root, root / RUN / "train/train-receipt.json", manifest["train_receipt"])
    receipt = json.loads(receipt_path.read_bytes())
    train_path = checked_path(root, root / RUN / "train/train.jsonl")
    training = manifest["training"]
    common_ids = receipt["common_cases"]
    expect(receipt["schema"] == "semantic-transfer-train-v1" and receipt["count"] == 640
           and receipt["updates"] == 160 and len(common_ids) == len(set(common_ids)) == 320
           and not set(common_ids) & EXCLUDED_SHARED and training["variable_cases_per_arm"] == 320
           and receipt["identity"]["frozen_sha256"] == training["frozen_sha256"] == FROZEN
           and all(receipt["identity"][key] == val for key, val in training.items())
           and manifest["token_balance"] == receipt["token_balance"]
           and receipt["schedule_keys"] == {arm: {str(s): f"{arm}-{s}" for s in SEEDS} for arm in ARMS}
           and common["train"] == {"path": train_path.relative_to(root).as_posix(),
                                   "bytes": receipt["bytes"], "sha256": receipt["sha256"]}
           and train_path.stat().st_size == receipt["bytes"], "TRAIN receipt differs from saved experiment")

    from source_provenance import verify_sources
    verify_sources(root, common["code"])
    recorded = {r["path"]: r["sha256"] for r in common["code"]["files"]}
    for filename, expected in receipt["identity"]["code"].items():
        if recorded.get(filename) != expected:
            pinned(root, filename, expected)
    assets = {name: pinned(root, Path("assets") / name, expected) for name, expected in FROZEN.items()}
    config = read(root, root / "experiments.json")
    for name in ("base", "tokenizer"):
        ref = config["assets"][name]
        expect(common[name]["sha256"] == ref["sha256"], f"Unexpected {name} digest")
        verify(root, root / ref["path"], common[name])

    context_manifest = pinned(root, "assets/diagnostic-train-contexts.json", CONTEXT_MANIFEST_SHA)
    contexts = read(root, root / context_manifest["path"])["contexts"]
    verify(root, root / "assets/diagnostic-train-contexts.jsonl", contexts)
    roots = {r["root_id"]: r for r in rows(root / assets["transfer-train-roots.jsonl"]["path"])}
    declarations = {arm: list(rows(root / assets[f"transfer-train-{arm}.jsonl"]["path"])) for arm in ARMS}
    labels = {}
    for records in declarations.values():
        for row in records:
            source = roots[row["root_id"]]
            labels[row["case_id"]] = {"section": "variable", "group": source["group"], "root_id": row["root_id"],
                                      "lineage": source["lineage"], "slot": row["slot"], "meaning": row["meaning"],
                                      "policy": row["continuation"]["kind"], "observed": "observed_prefix_id" in source}
    seeds = [p["seed"] for p in result["pairs"]]
    schedules = schedule_index(train_path, common_ids, declarations, labels, seeds, receipt)
    arms = {}
    for pair in result["pairs"]:
        for arm, model in pair["arms"].items():
            name = f"{arm}-{pair['seed']}"
            arms[name] = arm_checkpoint(root, name, model, schedules[name], manifest)
    terminal = checked_path(root, root / RUN / "result.json")
    terminal_bytes = terminal.read_bytes()
    expect(json.loads(terminal_bytes) == result, "Source result changed during verification")
    return {"source_result_sha256": sha256(terminal_bytes).hexdigest(), "train": common["train"],
            "train_receipt": manifest["train_receipt"], "base": common["base"], "tokenizer": common["tokenizer"],
            "registry": assets["registry.json"], "source_train": assets["train-full-v2.jsonl"],
            "context_manifest": context_manifest, "contexts": {k: contexts[k] for k in ("path", "bytes", "sha256")},
            "prefix_receipts": pinned(root, PREFIX_RECEIPTS, receipt["identity"]["observed_receipts_sha256"]),
            "transfer_roots": assets["transfer-train-roots.jsonl"],
            **{f"transfer_{arm}": assets[f"transfer-train-{arm}.jsonl"] for arm in ARMS},
            "train_provenance": {"initial_step": 7991, "checkpoint_step": 8151, "examples_per_update": 4,
                                 "updates": 160, "origin_checkpoint": common["initial_checkpoint"],
                                 "fresh_dev_bundle_sha256": manifest["fresh"]["bundle_sha256"],
                                 "conditions": {"ordinary": "paraphrase", "contrast": "semantic_contrast"}},
            "arms": arms}


def iter_cases(inputs, arm_name, root=ROOT):
    root = Path(root).resolve(strict=True)
    path = checked_path(root, root / inputs["train"]["path"])
    with path.open("rb") as stream:
        for meta in inputs["arms"][arm_name]["cases"]:
            stream.seek(meta["offset"])
            line = stream.readline()
            expect(sha256(line).hexdigest() == meta["row_sha256"], f"TRAIN changed at {meta['case_id']}")
            row = json.loads(line)
            expect(row["case_id"] == meta["case_id"] and row["positions"][arm_name] == meta["position"],
                   "Incorrect prepared TRAIN row")
            yield {**meta, "input_ids": row["input_ids"], "prompt_tokens": row["prompt_tokens"]}
