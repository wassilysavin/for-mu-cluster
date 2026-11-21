"""CPU regressions for experiment identity, crash recovery and the GPU-free evaluator bridge."""

import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import transfer_run as runner


class TransferRunnerTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.runs = self.root / "runs"
        self.directory = self.runs / runner.NAME
        self.directory.mkdir(parents=True)
        for key, value in (("ROOT", self.root), ("RUNS", self.runs)):
            self.enterContext(patch.object(runner, key, value))

    def prepared(self, *, invalid_replay=False, too_long=False):
        path = self.directory / "train.jsonl"
        row_list = []
        for pos in range(8):
            def row(case_id, keys, prompt=3):
                return {"case_id": case_id, "input_ids": [1]*prompt + [50], "prompt_tokens": prompt,
                        "positions": {key: (7 - pos if key.endswith("2718") else pos) for key in keys}}

            keys = {arm: [f"{arm}-{seed}" for seed in runner.SEEDS] for arm in runner.ARMS}
            if pos % 2 == 0 or (invalid_replay and pos == 7):
                row_list.append(row(f"common-{pos}", keys["ordinary"] + keys["contrast"]))
            else:
                for arm in runner.ARMS:
                    row_list.append(row(f"{arm}-{pos}", keys[arm], 32 if too_long and arm == "contrast" else 3))
        path.write_text("".join(json.dumps(row) + "\n" for row in row_list))
        return {"path": str(path), "sha256": runner.record(path)["sha256"], "count": 8, "updates": 2,
                "schedule_keys": {arm: {str(seed): f"{arm}-{seed}" for seed in runner.SEEDS} for arm in runner.ARMS}}

    def test_schedules_bind_actual_common_bytes_and_pilot(self):
        plans = runner.schedules(self.prepared())
        self.assertEqual(plans["pilot"].rows, plans["ordinary-1729"].rows[:8])
        self.assertNotEqual(plans["ordinary-1729"].sha256, plans["contrast-1729"].sha256)
        with self.assertRaisesRegex(ValueError, "half"):
            runner.schedules(self.prepared(invalid_replay=True))
        with self.assertRaisesRegex(ValueError, "input token"):
            runner.schedules(self.prepared(too_long=True))

    def test_saved_checkpoint_cursor_and_optimizer_lineage(self):
        path = self.directory / "weights"
        lineage = {"identity_sha256": "parent", "initial_checkpoint_sha256": "source"}
        header = json.dumps({"__metadata__": {"step": "7991", "lineage": json.dumps(lineage)}}).encode()
        path.write_bytes(len(header).to_bytes(8, "little") + header)
        runner.check_header(path, 7991, lineage)
        for step, expected_lineage in ((7992, lineage), (7991, {})):
            with self.assertRaisesRegex(ValueError, "step or lineage"):
                runner.check_header(path, step, expected_lineage)

    def test_disk_budget_includes_reference_artifacts_and_deduplicates_snapshots(self):
        file = self.directory / "checkpoint"
        file.write_bytes(b"a" * runner.MIB)
        space = object.__new__(runner.Space)
        space.previous = Mock()
        space.previous.usage.return_value = {"charged_bytes": runner.diagnose.DISK_LIMIT - 10*runner.MIB}
        self.enterContext(patch.object(runner.shutil, "disk_usage", return_value=Mock(free=10**12)))
        before = space.require(runner.MIB)
        os.link(file, file.with_suffix(".snapshot"))
        self.assertLess(abs(space.available() - before), 65536)
        space.previous.usage.return_value = {"charged_bytes": runner.diagnose.DISK_LIMIT}
        with self.assertRaisesRegex(RuntimeError, "3 GB"):
            space.require(1)

    def test_bridge_preserves_saved_prefix_on_provider_failure(self):
        file = self.directory / "observations.jsonl"
        script = self.root / "mock.py"
        script.write_text("""import json, sys
from pathlib import Path
path = Path(sys.argv[1])
previous = path.read_text().splitlines() if path.exists() else []
for i in range(len(previous), 2):
    print(json.dumps(dict(kind='generate',case_id='row'+str(i),prompt=str(i))), flush=True)
    reply = json.loads(input())
    with path.open('a') as out:
        print(json.dumps(reply), file=out)
print(json.dumps(dict(kind='result',score=dict(observed_records=2))), flush=True)
""")
        space = Mock(lease=None)
        space.require.return_value = 32 * runner.MIB
        cmd = [sys.executable, script, file]
        with self.assertRaisesRegex(RuntimeError, "interrupted"):
            runner.pump(cmd, file, Mock(side_effect=[{"raw_text": "a"}, RuntimeError("interrupted")]), "bound", space)
        self.assertEqual(len(file.read_text().splitlines()), 1)
        provider = Mock(return_value={"raw_text": "b"})
        self.assertEqual(runner.pump(cmd, file, provider, "bound", space)["observed_records"], 2)
        provider.assert_called_once_with("1")
        rows = [json.loads(line) for line in file.read_text().splitlines()]
        self.assertEqual([r["raw_text"] for r in rows], ["a", "b"])
        self.assertTrue(all(r["diagnostic_worker_sha256"] == "bound" for r in rows))

    def test_legacy_guard_uses_observed_standalone_and_detected_reasons(self):
        sample = self.directory / "dev.jsonl"
        samples = [{"case_id": "solo", "cohort": "supported"},
                   {"case_id": "turn", "cohort": "supported", "dialogue_id": "dialogue"}]
        sample.write_text("".join(json.dumps(row) + "\n" for row in samples))
        rows = [{"case_id": "solo", "result": {"success": False, "critical_status": "detected",
                                                "reasons": ["unexpected_address"]}},
                {"case_id": "turn", "result": {"success": True, "critical_status": "none_observed"}}]
        score = {"summary": {"critical_detected": 1}}
        runner.enrich_known(score, rows, sample)
        self.assertEqual(score["summary"]["standalone"], {"passed": 0, "planned": 1, "accuracy": 0.0})
        self.assertEqual(score["summary"]["critical_types"], {"unexpected_address": 1})

    def test_evaluation_reuses_complete_stages_and_detects_changed_scores(self):
        target = self.root / "weights"
        target.write_bytes(b"checkpoint")
        registry = self.root / "registry"
        registry.touch()
        known = self.root / "known"
        known.write_text("".join(json.dumps({"case_id": str(i), "cohort": "supported"}) + "\n" for i in range(155)))
        assets = {"registry": (registry, {}), "regression_dev": (known, {})}
        directory = self.directory
        old_rows = self.root / "old.jsonl"
        old_rows.touch()
        inherited = ({"observed_records": 1250, "summary": {"missing": 0}}, old_rows)

        def fake_pump(cmd, path, generate, binding, space):
            fresh = path.stem == "fresh"
            n = 8 if fresh else 155
            rows = [{"case_id": str(i), "result": {"success": True, "critical_status": "none_observed"}}
                    for i in range(n)]
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            return {"observed_records": n, "bundle_sha256": "bundle",
                    "summary": {"missing": 0, "critical_detected": 0, "critical_review_required": 0, "supported": {"passed": n, "planned": n}}}

        args = (directory, "parent", target, 7991, assets, {"regression": {}}, {"run": 1},
                {"cases": 8, "bundle_sha256": "bundle"}, Mock(), inherited)
        with patch.object(runner, "pump", side_effect=fake_pump) as pump:
            first = runner.evaluate(*args)
            self.assertEqual(pump.call_count, 2)
        with patch.object(runner, "pump", side_effect=AssertionError("regenerated")):
            self.assertEqual(first, runner.evaluate(*args))
        receipt = directory / "parent" / "fresh.json"
        damaged = json.loads(receipt.read_text())
        damaged["score"]["summary"]["supported"]["passed"] -= 1
        receipt.write_text(json.dumps(damaged))
        with self.assertRaisesRegex(ValueError, "Saved DEV stage"):
            runner.evaluate(*args)

    def test_coordinator_trains_and_evaluates_both_seeds_from_one_bound_parent(self):
        import transfer_train
        initial = self.root / "parent.safetensors"
        initial.write_bytes(b"saved parent")
        entry = runner.record(initial)
        assets = {key: (initial, entry) for key in ("base", "tokenizer", "history_dev", "regression_dev")}
        origin = {"step": 7991, "checkpoint": entry}
        fresh = self.root / "dev"
        fresh.write_text("fresh")
        self.enterContext(patch.object(runner, "DEV", fresh))
        self.enterContext(patch.object(runner, "check", return_value={"train": {}, "dev": {"cases": 1}, "separation": {}}))
        self.enterContext(patch.object(runner.history, "load", return_value=({}, {"architecture": {"id": "arch"}, "recipe": {}}, assets)))
        saved = {"baseline-7205": ({}, None), "treatment-1729": ({}, None)}
        self.enterContext(patch.object(runner, "source", return_value=(origin, saved)))
        budget = Mock()
        budget.require.return_value = 16 * runner.MIB
        self.enterContext(patch.object(runner, "Space", return_value=budget))
        def prepare(folder, *, available_bytes):
            data = folder / "train.jsonl"
            data.write_text("train")
            (folder / "train-receipt.json").write_text("receipt")
            return {"path": str(data), "updates": 1, "token_balance": {}}
        self.enterContext(patch.object(transfer_train, "prepare", side_effect=prepare))
        self.enterContext(patch.object(runner.run, "four_gpu_identity", return_value={"devices": [0,1,2,3]}))
        self.enterContext(patch.object(runner, "bindings", return_value={"files": [entry]}))
        train = Mock(identities={})
        def new_backend(assets, prepared, common, recipe, space):
            self.assertEqual(assets["initial_checkpoint"], (initial, entry))
            self.assertEqual(common["execution"]["devices"], [0,1,2,3])
            return train
        self.enterContext(patch.object(runner, "Training", side_effect=new_backend))
        evaluated = []
        def evaluate(directory, name, checkpoint, step, *rest):
            model = directory / name
            model.mkdir()
            (model / "fresh.jsonl").write_text('{"case_id":"id"}\n')
            evaluated.append((name, step))
            return {}
        self.enterContext(patch.object(runner, "evaluate", side_effect=evaluate))
        self.enterContext(patch.object(runner, "compare_pair", return_value={"positive_signal": True, "unresolved": {"contrast": {"fresh": 1}},
            "paired_95pct":{"lower":0.01,"upper":0.2}, "supported_gain":0.1}))
        runner.execute()
        names = [f"{arm}-{seed}" for seed in runner.SEEDS for arm in runner.ARMS]
        self.assertEqual(evaluated, [("baseline", 7991)] + [(name,7992) for name in names])
        self.assertEqual([call.args[0] for call in train.train_arm.call_args_list], names)
        self.assertEqual(runner.history.read(self.directory/"result.json")["status"], "inconclusive_pending_review")


if __name__ == "__main__":
    unittest.main()
