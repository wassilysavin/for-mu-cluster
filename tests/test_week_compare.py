"""CPU checks for the weekly comparison's immutable schedules and handoff."""

from hashlib import sha256
from copy import copy
import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import week_compare as week


class WeeklyComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plans, cls.exposure = week.prepare_schedules(week.ROOT / week.TRAIN["path"])

    def test_published_draws_use_frozen_sizes_and_token_totals(self):
        self.assertEqual(
            [
                (value["distinct_cases"], value["presentations"], value["target_tokens"])
                for value in self.exposure.values()
            ],
            [(5169, 16176, 445925), (16174, 16176, 446500)],
        )
        lengths = {row[0]: row[2] for row in self.plans["full"].rows}
        self.assertEqual(
            [row[2] for row in self.plans["pilot"].rows[:4]], sorted(lengths.values(), reverse=True)[:4]
        )
        self.assertNotEqual(self.plans["matched"].sha256, self.plans["full"].sha256)

    def test_batches_reconstruct_exact_targets_at_a_resume_cursor(self):
        for arm in (*week.ARMS, "pilot"):
            plan = self.plans[arm]
            offset = len(plan.rows) - week.BATCH_SIZE
            with plan.path.open("rb") as stream:
                batch, ids = plan.batch(stream, offset, week.BATCH_SIZE)
            self.assertEqual(ids, [row[0] for row in plan.rows[offset:]])
            self.assertEqual(
                [len(prompt) + len(target) for prompt, target in batch], [row[2] for row in plan.rows[offset:]]
            )
            self.assertTrue(all(target[-1] == 50 for _, target in batch))

    def test_budget_does_not_reset_on_retry(self):
        with TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            with patch.object(week, "ROOT", root), patch.object(week, "RUNS", root):
                first = week.open_budget(root, "binding", 100.0)
                week.run.claim_optimizer_budget(SimpleNamespace(optimizer_budget=first.account), 1)
                later = week.open_budget(root, "binding", 101.0)
                self.assertEqual(later.deadline, 100.0 + 168 * 3600)
                self.assertEqual(later.account["record"]["charged_updates"], 1)
                with self.assertRaisesRegex(ValueError, "budget changed"):
                    week.open_budget(root, "different", 102.0)

    def test_training_uses_identical_origin_and_reuses_its_verified_final_checkpoint(self):
        with TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            checkpoint = {"path": "source.safetensors", "bytes": 32, "sha256": "c" * 64}
            origin = {"step": 7991, "checkpoint": checkpoint, "lineage": {"parent": "source"}}
            common = {
                "origin": origin,
                "initial_checkpoint": checkpoint,
                "runtime": {"devices": []},
                "architecture_id": "FunctionGemma",
                "code": {"files": [checkpoint]},
                "train": checkpoint,
                "evaluation": dict(checkpoint, split="DEV-HISTORY"),
            }
            space = SimpleNamespace(require=lambda count: None)
            plans = {arm: copy(plan) for arm, plan in self.plans.items()}
            starts = []

            def train(job, identity, experiment, assets, schedule, *, until, lease_held, deadline):
                self.assertEqual(job.fork_lineage, origin["lineage"])
                self.assertEqual(assets["initial_checkpoint"][1], checkpoint)
                self.assertEqual(identity["recipe"]["initial_step"], 7991)
                self.assertEqual(identity["recipe"]["examples_per_update"], 4)
                self.assertEqual((until, lease_held, deadline), (4044, True, 12345))
                week.ledger._check_identity(identity)
                arm = experiment["training"]["arm"]
                self.assertIs(schedule, plans[arm])
                starts.append(arm)
                output = root / job.run_id
                output.mkdir()
                lineage = {
                    "identity_sha256": week.history.digest(identity),
                    "initial_checkpoint_sha256": checkpoint["sha256"],
                }
                header = json.dumps({"__metadata__": {"step": "12035", "lineage": json.dumps(lineage)}}).encode()
                (output / "checkpoint-step-12035.safetensors").write_bytes(struct.pack("<Q", len(header)) + header)

            with (
                patch.object(week, "ROOT", root),
                patch.object(week, "RUNS", root),
                patch.object(week.run, "train", side_effect=train),
            ):
                assets = {"initial_checkpoint": (root / checkpoint["path"], checkpoint)}
                backend = week.Training(plans, common, {}, assets, space)
                backend.budget = week.open_budget(root, "binding", 100.0)
                first = [backend.train_arm(arm, 12345) for arm in week.ARMS]
                retry = [backend.train_arm(arm, 12345) for arm in week.ARMS]
                self.assertEqual(first, retry)
                self.assertEqual(starts, ["matched", "full"])

    def test_completed_fit_binds_both_checkpoints_and_the_transfer_result(self):
        with TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            original = root / week.saved_transfer.RUN
            original.mkdir(parents=True)
            raw = b"{}"
            (original / "result.json").write_bytes(raw)
            checkpoints = {"ordinary": {"sha256": "a" * 64}, "contrast": {"sha256": "b" * 64}}
            source = {"pairs": [{"seed": 1729, "arms": {k: {"checkpoint": v} for k, v in checkpoints.items()}}]}
            fit = root / "runs" / "new-train-fit-v1"
            revision = fit / week.new_fit_workers.PYTHON_REVISION
            revision.mkdir(parents=True)
            arms = {f"{k}-1729": {"checkpoint": v} for k, v in checkpoints.items()}
            manifest = {
                "protocol": "new-train-fit-v1",
                "arms": arms,
                "source_result_sha256": sha256(raw).hexdigest(),
            }
            result = {
                "status": "complete",
                "step": 8151,
                "updates": 160,
                "experiment_sha256": week.history.digest(manifest),
                "arms": {k: {**v, "subsets": {"all": {"total": 640}}} for k, v in arms.items()},
            }
            with (
                patch.object(week, "ROOT", root),
                patch.object(week.transfer, "ROOT", root),
                patch.object(week.new_fit_workers, "OUTPUT", fit),
            ):
                with self.assertRaisesRegex(RuntimeError, "Finish check_new_train_fit"):
                    week.completed_fit(source)
                (revision / "experiment.json").write_text(json.dumps(manifest))
                receipt = revision / "result.json"
                receipt.write_text(json.dumps(result))
                self.assertEqual(week.completed_fit(source), week.transfer.record(receipt))
                result["arms"]["ordinary-1729"]["subsets"]["all"]["total"] = 639
                receipt.write_text(json.dumps(result))
                with self.assertRaisesRegex(ValueError, "completion differs"):
                    week.completed_fit(source)


if __name__ == "__main__":
    unittest.main()
