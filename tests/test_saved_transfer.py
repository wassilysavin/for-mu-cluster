"""Saved evidence checks run without model weights or model libraries."""

from hashlib import sha256
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import unittest

from diagnostics import saved_transfer as saved


def write(root, path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = data if isinstance(data, bytes) else json.dumps(data).encode()
    path.write_bytes(raw)
    return {"path": path.relative_to(root).as_posix(), "bytes": len(raw), "sha256": sha256(raw).hexdigest()}


def fresh_fixture(arm, plan, bundle, binding, confirm):
    result, panels, last = [], {}, {}
    counters = {}
    for panel in saved.PANELS:
        group = [r for r in plan if r["panel"] == panel]
        for item in group:
            key = (panel, item["cohort"])
            index = counters.get(key, 0)
            counters[key] = index + 1
            passed = index % 2 == 0
            if arm == "baseline":
                passed = index % 3 != 1
            elif arm.startswith("contrast"):
                if confirm:
                    passed = True
                elif index in (0, 1, 2, 4):
                    passed = not passed
            row = {k: item[k] for k in ("case_id", "cohort", "panel", "dialogue_id", "turn_index", "reference_eligible",
                                       "question", "family_id", "bootstrap_family_id")}
            lineage = item["dialogue_id"] or item["case_id"]
            before = last.get(lineage, saved.digest(lineage))
            after = saved.digest([before, item["case_id"], passed])
            last[lineage] = after
            row.update(success=passed, critical_status="none_observed", critical_category=None,
                       failure_category=None if passed else "execution_error",
                       delivered={"error_code": None if passed else "invalid_arguments"},
                       observation_schema="diagnostic-observation-v1-compact",
                       generation={"diagnostic_worker_sha256": binding}, bundle_sha256=bundle,
                       before_sha256=before, before_visible_history_sha256=before, after_sha256=after)
            result.append(row)
        summary = {"critical_detected": 0, "critical_review_required": 0, "missing": 0}
        these = [row for row in result if row["panel"] == panel]
        for cohort in ("supported", "outside_scope"):
            selected = [r for r in these if r["cohort"] == cohort]
            summary[cohort] = {"planned": len(selected), "passed": sum(r["success"] for r in selected)}
        dialogues = {r["dialogue_id"] for r in these if r["dialogue_id"]}
        whole = sum(all(r["success"] for r in these if r["dialogue_id"] == dialogue) for dialogue in dialogues)
        summary["whole_dialogues"] = {"planned": len(dialogues), "passed": whole}
        panels[panel] = summary
    return result, {"panels": panels, "summary": panels["closed_loop"], "observed_records": len(plan), "bundle_sha256": bundle}


def fixture(root, confirm):
    directory = root / saved.RUN
    dev = write(root, root / saved.DEV, (saved.ROOT / saved.DEV).read_bytes())
    data = json.loads((root / saved.DEV).read_bytes())
    plan = [row for k in saved.PANELS for row in data["panels"][k]]
    registry = saved.REGISTRY_SHA256
    bundle = saved.digest({"asset_sha256": dev["sha256"], "registry_sha256": registry, "version": data["schema"]})
    parent = {"path": "runs/source/checkpoint-step-7991.safetensors", "sha256": "a"*64, "bytes": 100}
    legacy = {"observed_records": 1250, "summary": {"missing": 0, "critical_review_required": 0,
                                                  "supported": {"passed": 507, "planned": 834}}}
    source = {"pairs": [{"seed": 1729, "treatment": {"checkpoint": parent, "score": legacy}}]}
    source_record = write(root, root / saved.SOURCE / "completion-v1/result.json", source)
    common = {"origin": {"step": 7991, "name": "treatment-1729", "checkpoint": parent, "source_result": source_record},
              "initial_checkpoint": parent, "evaluation": dev}
    identifiers, checkpoints = {}, {"baseline": parent}
    seeds = saved.SEEDS if confirm else [1729]
    for seed in seeds:
        for arm in saved.ARMS:
            name = f"{arm}-{seed}"
            train = root / "runs" / ("semantic-transfer-v1-" + name)
            identity = {"initial_checkpoint": parent, "experiment_id": name,
                        "recipe": {"initial_step": 7991, "training": {"updates": 160}}}
            identifiers[name] = saved.digest(identity)
            write(root, train / "run.json", {"identity": identity, "identity_sha256": saved.digest(identity)})
            checkpoints[name] = {"path": (train / "checkpoint-step-8151.safetensors").relative_to(root).as_posix(),
                                 "sha256": saved.digest(name), "bytes": 101}
    manifest = {"protocol": "semantic-transfer-v1", "common": common, "policy": {"seeds": saved.SEEDS},
                "identities": identifiers, "fresh": {"cases": 233, "panels": saved.PANELS,
                                                     "registry_sha256": registry, "bundle_sha256": bundle}}
    write(root, directory / "experiment.json", manifest)
    identity = saved.digest(manifest)
    models = {}
    for name, checkpoint in checkpoints.items():
        folder = directory / name
        binding = saved.digest({"manifest": identity, "model": name, "checkpoint": checkpoint})
        write(root, folder / "input.json", {"binding": binding, "checkpoint": checkpoint})
        observations, new_score = fresh_fixture(name, plan, bundle, binding, confirm)
        scores = {"fresh": new_score, "legacy155": {"observed_records": 155, "summary":
                  dict(legacy["summary"], supported={"passed": 53, "planned": 131})}, "legacy1250": legacy}
        encoding = b"".join(json.dumps(row).encode() + b"\n" for row in observations)
        for split, score in scores.items():
            paths = saved.stage_paths(root, folder, name, split)
            stage = {"binding": binding, "score": score, "observations":
                     [write(root, p, encoding if split == "fresh" else b'{"case_id":"legacy"}\n') for p in paths]}
            stage["receipt_sha256"] = saved.digest(stage)
            write(root, folder / f"{split}.json", stage)
        models[name] = {"checkpoint": checkpoint, "scores": scores}
    pairs = []
    for seed in seeds:
        arms = {arm: models[f"{arm}-{seed}"] for arm in saved.ARMS}
        metrics = [arm["scores"]["fresh"]["summary"]["supported"] for arm in arms.values()]
        gain = (metrics[1]["passed"] - metrics[0]["passed"]) / metrics[0]["planned"]
        pair = {"seed": seed, "arms": arms,
                "decision": {"seed": seed, "positive_signal": confirm, "supported_gain": gain,
                             "paired_95pct": {"lower": gain - .05, "upper": gain + .05}}}
        pairs.append(pair)
        write(root, directory / f"pair-{seed}.json", pair)
    result = {"identity_sha256": identity, "baseline": models["baseline"], "pairs": pairs,
              "release_ready": False, "status": "confirmed_diagnostic_gain" if confirm else "not_promoted"}
    write(root, directory / "result.json", result)
    return directory


class SavedTransferTests(unittest.TestCase):
    def setUp(self):
        private = saved.ROOT / ".runtime/tmp"
        private.mkdir(parents=True, exist_ok=True)
        temp = TemporaryDirectory(dir=private)
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_first_pair_and_repeated_read_leave_files_unchanged(self):
        fixture(self.root, False)
        before = {path: path.stat().st_mtime_ns for path in self.root.rglob("*")}
        for _ in range(2):
            _, result, plan, runs = saved.load_saved_transfer(self.root)
            self.assertEqual(result["status"], "not_promoted")
            self.assertEqual(len(plan), 233)
            self.assertEqual(set(runs), {"baseline", "ordinary-1729", "contrast-1729"})
        self.assertEqual(before, {path: path.stat().st_mtime_ns for path in self.root.rglob("*")})

    def test_short_report_command_reads_only_saved_cases(self):
        fixture(self.root, False)
        for name in ("diagnostics/student_evidence.py", "diagnostics/saved_transfer.py", "inspect_transfer.py"):
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(saved.ROOT / name, self.root / name)
        before = {path: path.stat().st_mtime_ns for path in self.root.rglob("*")}
        proc = subprocess.run([sys.executable, "-B", "inspect_transfer.py"], cwd=self.root, capture_output=True,
                              check=True, text=True)
        for label in ("parent 7991", "contrast vs ordinary", "contrast vs parent", "ordinary vs parent",
                      "-1.4 pp", "+1/-3", "primary gain", "primary loss"):
            self.assertIn(label, proc.stdout)
        self.assertEqual(before, {path: path.stat().st_mtime_ns for path in self.root.rglob("*")})

    def test_two_pairs(self):
        fixture(self.root, True)
        _, result, plan, runs = saved.load_saved_transfer(self.root)
        self.assertEqual(len(runs), 5)
        self.assertEqual(list(runs["contrast-2718"]), list(plan))
        self.assertEqual(result["status"], "confirmed_diagnostic_gain")

    def test_tampered_legacy_shard_or_checkpoint_identity(self):
        directory = fixture(self.root, False)
        shard = directory / "ordinary-1729/history-2.jsonl"
        raw = shard.read_bytes()
        shard.write_bytes(raw.replace(b"legacy", b"forged"))
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            saved.load_saved_transfer(self.root)
        shard.write_bytes(raw)
        path = directory / "contrast-1729/input.json"
        write(self.root, path, {"binding": "other", "checkpoint": {}})
        with self.assertRaisesRegex(ValueError, "Evaluation identity"):
            saved.load_saved_transfer(self.root)

    def test_rehashed_missing_case_or_changed_history_rejected(self):
        directory = fixture(self.root, False)
        folder = directory / "contrast-1729"
        path = folder / "fresh.jsonl"
        source = path.read_bytes().splitlines()
        stage_path = folder / "fresh.json"
        stage = json.loads(stage_path.read_bytes())
        for kind in ("missing", "history", "verdict"):
            rows = [json.loads(line) for line in source]
            if kind == "missing":
                rows.pop()
            elif kind == "history":
                next(r for r in rows if r["turn_index"] == 2)["before_sha256"] = "different"
            else:
                rows[0]["success"] = not rows[0]["success"]
            raw = b"".join(json.dumps(r).encode() + b"\n" for r in rows)
            stage["observations"] = [write(self.root, path, raw)]
            stage.pop("receipt_sha256")
            stage["receipt_sha256"] = saved.digest(stage)
            write(self.root, stage_path, stage)
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                saved.load_saved_transfer(self.root)
