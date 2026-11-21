"""Read and cross-check the completed semantic-transfer experiment from saved evidence."""

from collections import Counter
from itertools import zip_longest
from pathlib import Path
import json
import math

from diagnostics.student_evidence import SOURCE, digest, read, verify

ROOT = Path(__file__).resolve().parents[1]
RUN = Path("runs/semantic-transfer-v1")
DEV = Path("assets/transfer-dev-v1.json")
DEV_SHA256 = "9df60f65e30c8b0d54966b2a2076af1318810ac1cb3cfecd77457dc299c0ff7f"
REGISTRY_SHA256 = "86e93e72bcd5b8c219e1ab7403b6ac6ea2d26bf5c3ccb589906c3f7852d670a5"
PANELS = {"closed_loop": 161, "challenge": 72}
SIZES = {"fresh": 233, "legacy155": 155, "legacy1250": 1250}
ARMS = ("ordinary", "contrast")
SEEDS = [1729, 2718]


def expect(condition, message):
    if not condition:
        raise ValueError(message)


def load_plan(root, manifest):
    record = manifest["common"]["evaluation"]
    expect(record["sha256"] == DEV_SHA256, "Fresh DEV has a different identity")
    bundle = json.loads(verify(root, root / DEV, record).read_bytes())
    fresh = manifest["fresh"]
    expect(fresh["registry_sha256"] == REGISTRY_SHA256 and fresh["cases"] == SIZES["fresh"] and fresh["panels"] == PANELS and
           fresh["bundle_sha256"] == digest({"asset_sha256": DEV_SHA256, "registry_sha256": fresh["registry_sha256"],
                                              "version": bundle["schema"]}), "Fresh DEV specification differs")
    rows = [row for panel in PANELS for row in bundle["panels"][panel]]
    plan = {row["case_id"]: row for row in rows}
    expect(len(plan) == len(rows) == SIZES["fresh"] and Counter(r["panel"] for r in rows) == PANELS,
           "Fresh DEV duplicates or omits planned cases")
    return plan


def load_fresh(path, plan, bundle, binding, score):
    rows, last = {}, {}
    fields = ("case_id", "panel", "cohort", "family_id", "bootstrap_family_id", "dialogue_id", "turn_index",
              "question", "reference_eligible")
    with path.open(encoding="utf-8") as stream:
        for line, case in zip_longest(stream, plan.values()):
            expect(line is not None and case is not None, f"Incomplete or extra rows in {path.name}")
            row = json.loads(line)
            expect(all(row[key] == case.get(key) for key in fields) and row["bundle_sha256"] == bundle and
                   row["generation"]["diagnostic_worker_sha256"] == binding and
                   row["observation_schema"] == "diagnostic-observation-v1-compact",
                   f"Fresh observation differs from its case or checkpoint: {case['case_id']}")
            expect(type(row["success"]) is bool and row["critical_status"] in
                   ("none_observed", "review_required", "detected") and
                   (not row["success"] or row["critical_status"] == "none_observed"), "Invalid fresh verdict")
            dialogue = case.get("dialogue_id")
            if dialogue:
                if dialogue in last:
                    expect(last[dialogue] == row["before_sha256"], "Broken generated dialogue history")
                last[dialogue] = row["after_sha256"]
            rows[row["case_id"]] = row
    for panel in PANELS:
        records = [r for r in rows.values() if r["panel"] == panel]
        summary = score["panels"][panel]
        dialogues = {r["dialogue_id"] for r in records if r["dialogue_id"]}
        whole = sum(all(r["success"] for r in records if r["dialogue_id"] == d) for d in dialogues)
        for key in ("supported", "outside_scope"):
            part = [r for r in records if r["cohort"] == key]
            metric = summary[key]
            expect((metric["passed"], metric["planned"]) == (sum(r["success"] for r in part), len(part)),
                   f"Fresh {panel}/{key} score differs from saved rows")
        expect((summary["whole_dialogues"]["passed"], summary["whole_dialogues"]["planned"]) == (whole, len(dialogues))
               and summary["critical_review_required"] == sum(r["critical_status"] == "review_required" for r in records)
               and summary["critical_detected"] == sum(r["critical_status"] == "detected" for r in records)
               and summary["missing"] == 0, f"Fresh {panel} summary differs from saved observations")
    expect(score["summary"] == score["panels"]["closed_loop"] and score["bundle_sha256"] == bundle,
           "Fresh primary score or bundle mismatch")
    return rows


def stage_paths(root, directory, name, split):
    if split != "legacy1250":
        return [directory / (split + ".jsonl")]
    if name == "baseline":
        return [root / SOURCE / "evaluations/treatment-1729/observations.jsonl"]
    return [directory / f"history-{i}.jsonl" for i in range(4)]


def load_model(root, manifest, result, plan, name, model):
    directory = root / RUN / name
    checkpoint = model["checkpoint"]
    if name == "baseline":
        expect(checkpoint == manifest["common"]["initial_checkpoint"], "Different source checkpoint")
    else:
        run_dir = root / "runs" / f"semantic-transfer-v1-{name}"
        trained = read(root, run_dir / "run.json")
        identity = trained["identity"]
        expect(digest(identity) == trained["identity_sha256"] == manifest["identities"][name] and
               identity["experiment_id"] == name and identity["initial_checkpoint"] == manifest["common"]["initial_checkpoint"]
               and identity["recipe"]["initial_step"] == 7991 and identity["recipe"]["training"]["updates"] == 160
               and checkpoint["path"] == (run_dir / "checkpoint-step-8151.safetensors").relative_to(root).as_posix(),
               f"Training identity or checkpoint mismatch for {name}")

    binding = digest({"manifest": result["identity_sha256"], "checkpoint": checkpoint, "model": name})
    expect(read(root, directory / "input.json") == {"binding": binding, "checkpoint": checkpoint},
           f"Evaluation identity mismatch for {name}")
    expect(model["scores"].keys() == SIZES.keys(), f"Missing panel for {name}")
    for split, size in SIZES.items():
        receipt = read(root, directory / (split + ".json"))
        checksum = receipt.pop("receipt_sha256")
        score = receipt["score"]
        expect(digest(receipt) == checksum and receipt["binding"] == binding and score == model["scores"][split]
               and score["observed_records"] == size and score["summary"]["missing"] == 0,
               f"Stage receipt differs for {name}/{split}")
        paths = stage_paths(root, directory, name, split)
        for path, record in zip(paths, receipt["observations"], strict=True):
            verify(root, path, record)
        if split == "fresh":
            rows = load_fresh(paths[0], plan, manifest["fresh"]["bundle_sha256"], binding, score)
    return rows


def load_saved_transfer(root=ROOT):
    root = Path(root).resolve(strict=True)
    directory = root / RUN
    manifest = read(root, directory / "experiment.json")
    result = read(root, directory / "result.json")
    common = manifest["common"]
    origin = common["origin"]
    expect(manifest["protocol"] == RUN.name and result["identity_sha256"] == digest(manifest) and
           manifest["policy"]["seeds"] == SEEDS and origin["step"] == 7991 and origin["name"] == "treatment-1729" and
           origin["checkpoint"] == common["initial_checkpoint"] and result["release_ready"] is False,
           "Unexpected semantic-transfer experiment or original checkpoint")
    original = verify(root, root / SOURCE / "completion-v1/result.json", origin["source_result"])
    original_pair = json.loads(original.read_bytes())["pairs"][0]
    expect(original_pair["seed"] == 1729, "Unexpected parent experiment seed")
    saved_parent = original_pair["treatment"]
    expect(saved_parent["checkpoint"] == origin["checkpoint"] and
           saved_parent["score"] == result["baseline"]["scores"]["legacy1250"], "Inherited source evidence differs")
    plan = load_plan(root, manifest)
    pairs = result["pairs"]
    expect([pair["seed"] for pair in pairs] in ([SEEDS[0]], SEEDS), "Incomplete or unexpected seed sequence")
    rows = {"baseline": load_model(root, manifest, result, plan, "baseline", result["baseline"])}
    for pair in pairs:
        seed = pair["seed"]
        expect(pair == read(root, directory / f"pair-{seed}.json") and pair["arms"].keys() == set(ARMS)
               and pair["decision"]["seed"] == seed, f"Saved comparison differs for seed {seed}")
        for arm in ARMS:
            name = f"{arm}-{seed}"
            rows[name] = load_model(root, manifest, result, plan, name, pair["arms"][arm])
        supported = [pair["arms"][arm]["scores"]["fresh"]["summary"]["supported"] for arm in ARMS]
        gain = (supported[1]["passed"] - supported[0]["passed"]) / supported[0]["planned"]
        expect(math.isclose(gain, pair["decision"]["supported_gain"], rel_tol=0, abs_tol=1e-12),
               f"Saved gain differs for seed {seed}")
    expect((len(pairs) == 2) == (pairs[0]["decision"]["positive_signal"] is True),
           "Seed sequence differs from the saved first-pair decision")
    pending = any(summary["critical_review_required"] for pair in pairs
                  for model in (result["baseline"], *pair["arms"].values()) for score in model["scores"].values()
                  for summary in score.get("panels", {"main": score["summary"]}).values())
    status = "not_promoted"
    if len(pairs) == 2 and pairs[-1]["decision"]["positive_signal"]:
        status = "inconclusive_pending_review" if pending else "confirmed_diagnostic_gain"
    expect(result["status"] == status, "Terminal status differs from completed pairs")
    return manifest, result, plan, rows
