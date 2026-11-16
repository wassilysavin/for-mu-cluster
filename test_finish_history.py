"""CPU-only completion/recovery tests; synthetic provider failures are not inference."""

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import struct
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import history as h
import finish_history as f
import run as runner


class Clock:
    def __init__(self, value=1_000_000.0):
        self.value = value

    def __call__(self):
        return self.value


def old_account(*, charged=792, elapsed=140_000.0, inflight=None):
    return {"identity_sha256": "original-budget-identity", "maximum_updates": h.MAX_UPDATES,
            "charged_updates": charged, "maximum_seconds": h.MAX_SECONDS,
            "elapsed_seconds": elapsed, "inflight": inflight,
            "stages": [{"name": "old-completed-phase", "seconds": elapsed, "interrupted": False}]}


def score_fixture(passed, *, unresolved=0):
    def metric(count, planned):
        return {"passed": count, "planned": planned}

    summary = {"supported": metric(passed, 100), "outside_scope": metric(10, 10),
               "whole_dialogues": metric(4 if passed > 50 else 0, 10), "standalone": metric(20, 20),
               "exact_reference": metric(1, 1), "critical_review_required": unresolved,
               "critical_types": {}, "critical_detected": 0, "missing": 0}
    fixed = dict(deepcopy(summary), exact_reference=metric(passed, 100), critical_review_required=0)
    return {"score": {"summary": summary, "panels": {"closed_loop": summary, "fixed_prefix": fixed}},
            "seconds": 1.0}


def observation_fixture(passed):
    return {f"case-{index}": {"case_id": f"case-{index}", "family_id": f"family-{index % 10}",
            "panel": "closed_loop", "cohort": "supported", "success": index < passed,
            "critical_status": "none_observed"} for index in range(100)}


def schedule_fixture():
    schedules = {}
    for seed in h.SEEDS:
        for arm in ("control", "treatment"):
            name = f"{arm}-{seed}"
            schedules[name] = SimpleNamespace(sha256=name,
                rows=[("common", 0, 10, 0, 2)] * (h.UPDATES * 2)
                     + [(arm, 0, 10, 0, 2)] * (h.UPDATES * 2))
    totals = {name: {"prompt": h.UPDATES * 4 * 8, "target": h.UPDATES * 4 * 2} for name in schedules}
    return schedules, totals


class SchedulePreservationTests(unittest.TestCase):
    def test_original_matching_token_and_seed_guards_remain(self):
        schedules, totals = schedule_fixture()
        f.validate_schedules(schedules, totals)
        wrong = deepcopy(schedules)
        wrong["treatment-1729"].rows = [(row[0], 0, 20, 0, 2) for row in wrong["treatment-1729"].rows]
        with self.assertRaisesRegex(ValueError, "5% prompt"):
            f.validate_schedules(wrong, totals)
        wrong = deepcopy(schedules)
        wrong["treatment-1729"].rows[0] = ("different", 0, 10, 0, 2)
        with self.assertRaisesRegex(ValueError, "50% common"):
            f.validate_schedules(wrong, totals)
        wrong = deepcopy(schedules)
        wrong["control-2718"].sha256 = wrong["control-1729"].sha256
        with self.assertRaisesRegex(ValueError, "distinct schedules"):
            f.validate_schedules(wrong, totals)
        wrong_totals = deepcopy(totals)
        wrong_totals["control-1729"]["prompt"] += 1
        with self.assertRaisesRegex(ValueError, "token-budget receipt"):
            f.validate_schedules(schedules, wrong_totals)


class SyntheticBackend:
    """Disk-backed receipts with tiny JSON checkpoints; never imports a model runtime."""

    def __init__(self, root, budget, *, fail_at=None, unresolved=0):
        self.root, self.budget = root, budget
        self.fail_at, self.unresolved = fail_at, unresolved
        self.calls, self.optimizer_calls = [], []

    def after(self, name, value):
        self.calls.append(name)
        if name == self.fail_at:
            self.fail_at = None
            raise OSError("synthetic interruption after atomic artifact publication")
        return value

    def checkpoint(self, name, step):
        path = self.root / f"{name}-{step}.json"
        h.immutable(path, {"name": name, "step": step})
        return h.file_record(path, relative_to=self.root)

    def validate_artifact(self, record):
        if h.file_record(self.root / record["path"], relative_to=self.root) != record:
            raise ValueError("Synthetic artifact changed")

    def remaining_updates(self, name):
        if (self.root / f"{name}-7991.json").exists():
            return 0
        return h.MIDPOINT if (self.root / f"{name}-7598.json").exists() else h.UPDATES

    def train(self, name, milestone, deadline):
        path = self.root / f"{name}-{h.INITIAL_STEP + milestone}.json"
        if not path.exists():
            count = milestone - (h.UPDATES - self.remaining_updates(name))
            self.budget.claim_updates(count)
            self.optimizer_calls.append((name, count))
        value = {"checkpoint": self.checkpoint(name, h.INITIAL_STEP + milestone),
                 "step": h.INITIAL_STEP + milestone}
        return self.after(f"{name}-{milestone}", value)

    def evaluate(self, name, checkpoint, deadline, *, known=False):
        passed = 20 if name == "baseline" else 30 if name.startswith("control") else 60
        value = score_fixture(passed, unresolved=self.unresolved if name.startswith("treatment") else 0)
        value["checkpoint"] = h.file_record(checkpoint, relative_to=self.root)
        path = self.root / f"{name}-{'regression' if known else 'final'}-observations.json"
        h.immutable(path, observation_fixture(passed))
        value["observations"] = h.file_record(path, relative_to=self.root)
        return self.after(f"{name}-{'regression' if known else 'final'}", value)

    def validate_evaluation(self, value, *, known=False):
        self.validate_artifact(value["checkpoint"])
        self.validate_artifact(value["observations"])

    def outcomes(self, value):
        return h.read(self.root / value["observations"]["path"])

    def space(self, evaluations):
        self.calls.append(f"space-{evaluations}")


class CompletionBudgetTests(unittest.TestCase):
    def test_more_than_36_hours_allowed_but_original_update_cap_remains(self):
        with TemporaryDirectory() as temporary:
            clock, inherited = Clock(), old_account()
            original = deepcopy(inherited)
            budget = f.CompletionBudget(Path(temporary) / "budget.json", "completion", inherited, clock=clock)
            self.assertEqual(budget.account["record"]["charged_updates"], 792)
            self.assertEqual(budget.account["record"]["elapsed_seconds"], 140_000.0)
            self.assertEqual(budget.remaining_seconds, float("inf"))
            with budget.phase("long-evaluation") as deadline:
                self.assertEqual(deadline, float("inf"))
                clock.value += 10 * h.MAX_SECONDS
                budget.check()
                budget.claim_updates(h.MAX_UPDATES - 792)
                with self.assertRaises(h.BudgetLimit):
                    budget.claim_updates(1)
            self.assertEqual(budget.account["record"]["charged_updates"], h.MAX_UPDATES)
            self.assertEqual(budget.account["record"]["elapsed_seconds"], 140_000 + 10 * h.MAX_SECONDS)
            self.assertEqual(inherited, original, "Completion modified the inherited source object")

    def test_resume_keeps_charged_updates_and_elapsed_work(self):
        with TemporaryDirectory() as temporary:
            clock, inherited = Clock(), old_account()
            path = Path(temporary) / "budget.json"
            budget = f.CompletionBudget(path, "completion", inherited, clock=clock)
            with budget.phase("new-work"):
                budget.claim_updates(9)
                clock.value += 7
            before = path.read_bytes()
            resumed = f.CompletionBudget(path, "completion", inherited, clock=clock)
            self.assertEqual(resumed.account["record"]["charged_updates"], 801)
            self.assertEqual(resumed.account["record"]["elapsed_seconds"], 140_007)
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaises(ValueError):
                f.CompletionBudget(path, "another-completion", inherited, clock=clock)
            self.assertEqual(path.read_bytes(), before)

    def test_model_phases_are_sequential_and_interrupted_work_persists(self):
        with TemporaryDirectory() as temporary:
            clock, inherited = Clock(), old_account()
            path = Path(temporary) / "budget.json"
            budget = f.CompletionBudget(path, "completion", inherited, clock=clock)
            with self.assertRaisesRegex(OSError, "interrupted"):
                with budget.phase("first"):
                    budget.claim_updates(2)
                    clock.value += 11
                    with self.assertRaises(RuntimeError):
                        with budget.phase("parallel"):
                            self.fail("Concurrent model phase was admitted")
                    raise OSError("interrupted")
            resumed = f.CompletionBudget(path, "completion", inherited, clock=clock)
            self.assertEqual(resumed.account["record"]["charged_updates"], 794)
            self.assertEqual(resumed.account["record"]["elapsed_seconds"], 140_011)
            self.assertIsNone(resumed.account["record"]["inflight"])

    def test_old_inflight_is_marked_conservative_and_inherited_only_once(self):
        with TemporaryDirectory() as temporary:
            clock = Clock()
            inherited = old_account(inflight={"name": "old-control-DEV", "started_at": clock() - 100_000})
            preserved = deepcopy(inherited)
            path = Path(temporary) / "budget.json"
            budget = f.CompletionBudget(path, "completion", inherited, clock=clock)
            recovered = budget.account["record"]
            self.assertEqual(recovered["charged_updates"], inherited["charged_updates"])
            self.assertEqual(recovered["elapsed_seconds"], inherited["elapsed_seconds"] + 100_000)
            interval = recovered["stages"][-1]
            self.assertEqual(interval["name"], "old-control-DEV")
            self.assertEqual(interval["seconds"], 100_000)
            self.assertIs(interval["interrupted"], True)
            self.assertIs(interval["conservative_recovery_charge"], True)
            self.assertIn("not GPU compute time", f.POLICY["time_accounting"])
            clock.value += 1_000_000
            resumed = f.CompletionBudget(path, "completion", inherited, clock=clock)
            self.assertEqual(resumed.account["record"], recovered)
            self.assertEqual(inherited, preserved)

    def test_changed_inherited_account_and_rollback_are_rejected(self):
        with TemporaryDirectory() as temporary:
            path, inherited = Path(temporary) / "budget.json", old_account()
            f.CompletionBudget(path, "completion", inherited)
            modified = dict(inherited, charged_updates=791)
            with self.assertRaises(ValueError):
                f.CompletionBudget(path, "completion", modified)
            value = h.read(path)
            value["charged_updates"] = inherited["charged_updates"] - 1
            h.replace_json(path, value)
            with self.assertRaises(ValueError):
                f.CompletionBudget(path, "completion", inherited)

    def test_unchanged_training_worker_charges_same_mutable_completion_account(self):
        with TemporaryDirectory() as temporary:
            clock = Clock()
            path = Path(temporary) / "budget.json"
            budget = f.CompletionBudget(path, "completion", old_account(charged=h.MAX_UPDATES - 1), clock=clock)
            job = SimpleNamespace(optimizer_budget=budget.account)
            with budget.phase("unchanged-worker"):
                runner.claim_optimizer_budget(job, 1)
                self.assertEqual(budget.account["record"]["charged_updates"], h.MAX_UPDATES)
                with self.assertRaises(runner.BudgetExhausted):
                    runner.claim_optimizer_budget(job, 1)
                clock.value += 5
            self.assertEqual(h.read(path)["charged_updates"], h.MAX_UPDATES)
            self.assertEqual(h.read(path)["elapsed_seconds"], 140_005)


class CompletionStagesTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.root = Path(self.stack.enter_context(TemporaryDirectory()))
        self.source, self.output = self.root / "source", self.root / "completion-v1"
        (self.source / "stages").mkdir(parents=True)
        self.output.mkdir()
        self.budget = f.CompletionBudget(self.output / "budget.json", "completion", old_account())
        self.verifications = []
        self.stages = f.CompletionStages(self.source, self.output, "source-binding", "completion",
                                        self.budget, verify=lambda: self.verifications.append(True))

    def tearDown(self):
        self.stack.close()

    def test_authenticated_source_baseline_and_regression_are_reused_unchanged(self):
        for name in ("baseline", "control-1729-regression"):
            h.immutable(self.source / "stages" / f"{name}.json",
                        {"identity_sha256": "source-binding", "result": {"receipt": name}})
        (self.source / "budget.json").write_text(h.canonical(old_account()))
        (self.source / "result.json").write_text('{"status":"budget_stopped","pairs":[]}\n')
        before = {path: path.read_bytes() for path in self.source.rglob("*") if path.is_file()}
        validated = []
        for name in ("baseline", "control-1729-regression"):
            value = self.stages.run(name, lambda deadline: self.fail("Cached source stage was rerun"),
                                    lambda record: validated.append(record["receipt"]))
            self.assertEqual(value, {"receipt": name})
        self.assertEqual(validated, ["baseline", "control-1729-regression"])
        self.assertEqual(len(self.verifications), 2)
        self.assertEqual({path: path.read_bytes() for path in before}, before)

    def test_wrong_source_identity_is_rejected_before_action(self):
        path = self.source / "stages" / "baseline.json"
        h.immutable(path, {"identity_sha256": "foreign", "result": {"receipt": "bad"}})
        preserved = path.read_bytes()
        with self.assertRaises(ValueError):
            self.stages.run("baseline", lambda deadline: self.fail("Unauthenticated stage reran"))
        self.assertEqual(path.read_bytes(), preserved)

    def test_cached_source_artifact_must_still_validate(self):
        path = self.source / "stages" / "baseline.json"
        h.immutable(path, {"identity_sha256": "source-binding", "result": {"receipt": "tampered"}})
        def reject(value):
            raise ValueError("Saved artifact hash changed")
        with self.assertRaisesRegex(ValueError, "hash changed"):
            self.stages.run("baseline", lambda deadline: self.fail("Invalid artifact reran"), reject)

    def test_new_stages_are_published_to_completion_and_reused(self):
        calls = []
        self.assertEqual(self.stages.run("treatment-1729-final", lambda deadline:
                         calls.append(deadline) or {"receipt": "new"}), {"receipt": "new"})
        resumed = f.CompletionStages(self.source, self.output, "source-binding", "completion", self.budget)
        self.assertEqual(resumed.run("treatment-1729-final", lambda deadline:
                         self.fail("Completed new stage reran")), {"receipt": "new"})
        self.assertEqual(calls, [float("inf")])
        self.assertFalse((self.source / "stages" / "treatment-1729-final.json").exists())
        self.assertEqual(h.read(self.output / "stages" / "treatment-1729-final.json")["identity_sha256"],
                         "completion")

    def test_readonly_terminal_validation_does_not_create_missing_receipts(self):
        nonexistent = self.root / "absent-completion"
        readonly = f.CompletionStages(self.source, nonexistent, "source-binding", "completion", None)
        self.assertFalse(nonexistent.exists())
        with self.assertRaisesRegex(ValueError, "missing its stage receipt"):
            readonly.run("treatment-1729-final", lambda deadline: self.fail("Read-only loader ran a worker"))
        self.assertFalse(nonexistent.exists())


class FrozenIdentityTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.root = Path(self.stack.enter_context(TemporaryDirectory()))
        self.stack.enter_context(patch.object(f, "ROOT", self.root))

    def tearDown(self):
        self.stack.close()

    def test_code_git_metadata_may_change_but_frozen_bytes_cannot(self):
        source = self.root / "worker.py"
        source.write_text("unchanged implementation\n")
        entries = [h.file_record(source, relative_to=self.root)]
        original = {"files": entries, "git": {"head": "old", "dirty": True}}
        current = {"files": deepcopy(entries), "git": {"head": "published-completion", "dirty": False}}
        self.assertEqual(f.validate_code_identity(current, original), original)
        current["files"].append({"path": "unbound.py", "sha256": "new", "bytes": 1})
        with self.assertRaises(ValueError):
            f.validate_code_identity(current, original)
        current["files"] = deepcopy(entries)
        source.write_text("changed implementation\n")
        with self.assertRaises(ValueError):
            f.validate_code_identity(current, original)

    def checkpoint(self, name, step, lineage):
        path = self.root / name
        metadata = h.canonical({"__metadata__": {"step": str(step), "lineage": h.canonical(lineage)}}).encode()
        path.write_bytes(struct.pack("<Q", len(metadata)) + metadata)
        return h.file_record(path, relative_to=self.root)

    def test_checkpoint_requires_exact_step_source_hash_and_arm_identity(self):
        identity = {"name": "control-1729", "source": "unchanged"}
        source = {"manifest": {"origin": {"lineage": {"origin": "7205"}},
                               "assets": {"initial_checkpoint": {"sha256": "initial-hash"}}},
                  "identities": {"control-1729": identity}}
        lineage = {"identity_sha256": h.digest(identity), "initial_checkpoint_sha256": "initial-hash"}
        good = self.checkpoint("control.safetensors", 7991, lineage)
        f.validate_checkpoint(source, "control-1729", good, 7991)
        baseline = self.checkpoint("baseline.safetensors", 7205, {"origin": "7205"})
        f.validate_checkpoint(source, "baseline", baseline, 7205)
        for change in ({"identity_sha256": "different-arm"}, {"initial_checkpoint_sha256": "different-base"}):
            with self.subTest(change=change):
                wrong = self.checkpoint("wrong.safetensors", 7991, {**lineage, **change})
                with self.assertRaises(ValueError):
                    f.validate_checkpoint(source, "control-1729", wrong, 7991)
        with self.assertRaises(ValueError):
            f.validate_checkpoint(source, "control-1729", good, 7598)
        (self.root / "control.safetensors").write_bytes(b"changed checkpoint")
        with self.assertRaises(ValueError):
            f.validate_checkpoint(source, "control-1729", good, 7991)

    def test_admitted_prefix_accepts_append_and_tail_repair_but_rejects_mutation(self):
        path = self.root / "observations.jsonl"
        prefix = b'{"case_id":"first"}\n'
        path.write_bytes(prefix + b'{"torn":')
        entry = f.prefix_record(path)
        self.assertEqual(entry["committed_bytes"], len(prefix))
        f.verify_prefix(entry)
        path.write_bytes(prefix + b'{"case_id":"next"}\n')
        f.verify_prefix(entry)
        path.write_bytes(b'{"case_id":"other"}\n')
        with self.assertRaises(ValueError):
            f.verify_prefix(entry)
        path.write_bytes(prefix[:-1])
        with self.assertRaises(ValueError):
            f.verify_prefix(entry)

    def test_complete_unterminated_observation_is_pinned_before_newline_repair(self):
        path = self.root / "observations.jsonl"
        original = b'{"case_id":"saved"}\n{"case_id":"complete-tail"}'
        path.write_bytes(original)
        entry = f.prefix_record(path)
        self.assertEqual(entry["committed_bytes"], len(original))
        path.write_bytes(original + b'\n{"case_id":"next"}\n')
        f.verify_prefix(entry)
        path.write_bytes(b'{"case_id":"saved"}\n{"case_id":"modified-tail"}\n')
        with self.assertRaises(ValueError):
            f.verify_prefix(entry)

    def test_timing_and_event_unterminated_tails_follow_existing_repair_contract(self):
        for name in ("timings.jsonl", "events.jsonl"):
            with self.subTest(name=name):
                path = self.root / name
                committed = b'{"kind":"load","seconds":1.0}\n'
                path.write_bytes(committed + b'{"kind":"overhead","seconds":2.0}')
                entry = f.prefix_record(path)
                self.assertEqual(entry["committed_bytes"], len(committed))
                path.write_bytes(committed)
                f.verify_prefix(entry)
                path.write_bytes(b'{"kind":"load","seconds":9.0}\n')
                with self.assertRaises(ValueError):
                    f.verify_prefix(entry)


class PairCompletionTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.root = Path(self.stack.enter_context(TemporaryDirectory()))
        self.stack.enter_context(patch.object(f, "ROOT", self.root))
        self.stack.enter_context(patch.object(h, "ROOT", self.root))
        self.source, self.output = self.root / "source", self.root / "completion-v1"
        (self.source / "stages").mkdir(parents=True)
        self.output.mkdir()
        self.inherited, self.clock = old_account(), Clock()
        self.budget = f.CompletionBudget(self.output / "budget.json", "completion", self.inherited, clock=self.clock)
        backend = SyntheticBackend(self.root, self.budget)
        self.initial = self.root / backend.checkpoint("baseline", 7205)["path"]
        midpoint = {"checkpoint": backend.checkpoint("control-1729", 7598), "step": 7598}
        trained = {"checkpoint": backend.checkpoint("control-1729", 7991), "step": 7991}
        baseline = backend.evaluate("baseline", self.initial, float("inf"))
        regression = backend.evaluate("control-1729", self.root / midpoint["checkpoint"]["path"],
                                      float("inf"), known=True)
        for name, result in (("baseline", baseline), ("control-1729-midpoint", midpoint),
                             ("control-1729-regression", regression), ("control-1729-trained", trained)):
            h.immutable(self.source / "stages" / f"{name}.json",
                        {"identity_sha256": "source-binding", "result": result})
        h.immutable(self.source / "budget.json", self.inherited)
        h.immutable(self.source / "result.json", {"status": "budget_stopped", "baseline": baseline, "pairs": []})
        self.source_before = {path: path.read_bytes() for path in self.source.rglob("*") if path.is_file()}

    def tearDown(self):
        self.stack.close()

    def execute(self, *, fail_at=None, unresolved=0):
        self.budget = f.CompletionBudget(self.output / "budget.json", "completion", self.inherited, clock=self.clock)
        stages = f.CompletionStages(self.source, self.output, "source-binding", "completion", self.budget)
        backend = SyntheticBackend(self.root, self.budget, fail_at=fail_at, unresolved=unresolved)
        result = f.complete_pair(backend, stages, self.budget, self.initial)
        return result, backend

    def assert_source_preserved(self):
        self.assertEqual({path: path.read_bytes() for path in self.source.rglob("*") if path.is_file()},
                         self.source_before)

    def test_complete_saved_control_and_one_treatment_without_second_seed(self):
        result, backend = self.execute()
        self.assertEqual(backend.calls[0], "space-2")
        self.assertEqual([pair["seed"] for pair in result["pairs"]], [1729])
        self.assertEqual(sum(count for _, count in backend.optimizer_calls), h.UPDATES)
        self.assertTrue(all(name == "treatment-1729" for name, _ in backend.optimizer_calls))
        self.assertEqual(self.budget.account["record"]["charged_updates"], 792 + h.UPDATES)
        self.assertFalse(any("2718" in name for name in backend.calls))
        self.assertFalse(any(name in backend.calls for name in ("baseline-final", "control-1729-regression")))
        self.assertIs(result["product_acceptance"], False)
        self.assert_source_preserved()
        repeated, repeated_backend = self.execute()
        self.assertEqual(repeated, result)
        self.assertEqual(repeated_backend.optimizer_calls, [])
        self.assertNotIn("space-2", repeated_backend.calls)
        self.assertFalse(any(name.endswith(("-final", "-regression")) for name in repeated_backend.calls))
        self.assert_source_preserved()

    def test_incomplete_pair_keeps_original_disk_admission_before_model_work(self):
        with patch.object(SyntheticBackend, "space", side_effect=OSError("free-space reserve exhausted")) as space:
            with self.assertRaisesRegex(OSError, "free-space reserve"):
                self.execute()
            space.assert_called_once_with(evaluations=2)
        self.assertFalse((self.root / "treatment-1729-7598.json").exists())
        self.assertFalse((self.output / "stages" / "control-1729-final.json").exists())
        self.assertEqual(self.budget.account["record"]["charged_updates"], 792)
        self.assert_source_preserved()

    def test_physical_updates_exceeding_original_cap_are_rejected_before_work(self):
        self.inherited = old_account(charged=h.MAX_UPDATES - h.UPDATES + 1)
        (self.output / "budget.json").unlink()
        with self.assertRaises(h.BudgetLimit):
            self.execute()
        self.assertFalse((self.root / "treatment-1729-7598.json").exists())
        self.assertEqual(h.read(self.output / "budget.json")["charged_updates"], self.inherited["charged_updates"])
        self.assert_source_preserved()

    def test_resume_after_saved_training_output_does_not_recharge_updates(self):
        with self.assertRaises(OSError):
            self.execute(fail_at="treatment-1729-393")
        self.assertEqual(self.budget.account["record"]["charged_updates"], 792 + h.MIDPOINT)
        midpoint = (self.root / "treatment-1729-7598.json").read_bytes()
        result, backend = self.execute()
        self.assertEqual(len(result["pairs"]), 1)
        self.assertEqual(backend.optimizer_calls, [("treatment-1729", h.MIDPOINT)])
        self.assertEqual((self.root / "treatment-1729-7598.json").read_bytes(), midpoint)
        self.assertEqual(self.budget.account["record"]["charged_updates"], 792 + h.UPDATES)
        self.assert_source_preserved()

    def test_unresolved_treatment_preserves_quality_gate(self):
        result, backend = self.execute(unresolved=1)
        self.assertFalse(result["pairs"][0]["decision"]["passed"])
        self.assertEqual(result["pairs"][0]["decision"]["status"], "inconclusive")
        self.assertFalse(any("2718" in name for name in backend.calls))
        self.assertIs(result["product_acceptance"], False)

    def admitted_source(self):
        """Isolate coordinator admission; load_source's real authentication is a separate boundary."""
        hardware = {"devices": ["synthetic-original-allocation"]}
        manifest = {"runtime": hardware}
        h.immutable(self.source / "experiment.json", manifest)
        binding = h.digest(manifest)
        for path in (self.source / "stages").glob("*.json"):
            value = h.read(path)
            value["identity_sha256"] = binding
            h.replace_json(path, value)
        self.source_before = {path: path.read_bytes() for path in self.source.rglob("*") if path.is_file()}
        (self.root / "finish_history.py").write_text("synthetic completion code identity\n")
        runs = self.root / "runs"
        runs.mkdir()
        source = {"directory": self.source, "manifest": manifest, "budget": self.inherited,
                  "assets": {"initial_checkpoint": (self.initial, h.file_record(self.initial, relative_to=self.root))},
                  "static": [h.file_record(path, relative_to=self.root) for path in self.source_before],
                  "runs": runs, "worker_run_id": "original-worker", "result_was_present": True}
        return source, hardware

    def test_compare_preserves_closed_budget_stop_and_publishes_bound_child_result(self):
        source, hardware = self.admitted_source()
        args = SimpleNamespace(runs=source["runs"], source_run_id="original")
        def load_source(runs, source_id):
            self.assertEqual(source_id, "original")
            with self.assertRaisesRegex(RuntimeError, "execution lock"):
                with f.execution_lease(runs):
                    self.fail("Source admission ran outside the execution lease")
            return source
        with patch.object(f, "load_source", side_effect=load_source), \
                patch.object(runner, "four_gpu_identity", return_value=hardware), \
                patch.object(runner, "require_run_space") as space, \
                patch.object(f, "make_backend", side_effect=lambda source, budget, stages:
                             SyntheticBackend(self.root, budget)), redirect_stdout(io.StringIO()):
            result = f.compare(args)
        child = self.source / f.COMPLETION_NAME
        manifest = h.read(child / "experiment.json")
        self.assertEqual(result["identity_sha256"], h.digest(manifest))
        self.assertEqual(result["completion_policy"], f.POLICY)
        self.assertEqual(result["original_budget"], self.inherited)
        self.assertEqual(result["budget"]["charged_updates"], 792 + h.UPDATES)
        self.assertEqual(result["budget"]["maximum_seconds"], None)
        self.assertEqual(manifest["source_snapshot"]["original_result_present"], True)
        self.assertEqual(manifest["source_directory"], str(self.source.relative_to(self.root)))
        self.assertEqual(h.read(child / "manifest-binding.json"), {"sha256": h.digest(manifest)})
        self.assertEqual(h.read(child / "result.json"), result)
        self.assertEqual([pair["seed"] for pair in result["pairs"]], [1729])
        self.assertIs(result["TEST_opened"], False)
        self.assertIs(result["product_acceptance"], False)
        self.assertEqual({path: path.read_bytes() for path in self.source_before}, self.source_before)
        space.assert_called_once_with(source["runs"], 1024 * 1024)
        for artifact in result["artifacts"]:
            f.verify(artifact)

    def test_compare_rejects_changed_runtime_before_creating_completion(self):
        source, _ = self.admitted_source()
        with patch.object(f, "load_source", return_value=source), \
                patch.object(runner, "four_gpu_identity", return_value={"devices": ["other-allocation"]}), \
                patch.object(f, "make_backend") as backend:
            with self.assertRaisesRegex(ValueError, "original Python environment"):
                f.compare(SimpleNamespace(runs=source["runs"], source_run_id="original"))
            backend.assert_not_called()
        self.assertFalse((self.source / f.COMPLETION_NAME).exists())
        self.assert_source_preserved()


class UnchangedBackendTests(unittest.TestCase):
    def test_completion_uses_existing_kernel_lease_before_source_or_model_access(self):
        with TemporaryDirectory() as temporary, patch.object(f, "ROOT", Path(temporary)):
            runs = Path(temporary) / "runs"
            args = SimpleNamespace(runs=runs, source_run_id="original")
            with f.execution_lease(runs), patch.object(f, "load_source") as load_source:
                with self.assertRaisesRegex(RuntimeError, "execution lock"):
                    f.compare(args)
                load_source.assert_not_called()
            with f.execution_lease(runs):
                pass

    def test_existing_worker_identity_and_lease_contract_survive_completion(self):
        with TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            stack.enter_context(patch.object(f, "ROOT", root))
            stack.enter_context(patch.object(h, "ROOT", root))
            output, runs, source_dir = root / "completion-v1", root / "runs", root / "source"
            output.mkdir()
            runs.mkdir()
            (source_dir / "stages").mkdir(parents=True)
            budget = f.CompletionBudget(output / "budget.json", "completion", old_account())
            stages = f.CompletionStages(source_dir, output, "source", "completion", budget)
            identities = {f"{arm}-1729": {"identity": f"frozen-{arm}"} for arm in ("control", "treatment")}
            receipt = {"lineage": {"source": "original-7205"}}
            source = {"runs": runs, "worker_run_id": "original-rope-alignment-v1-checkpoint-io-v1",
                      "manifest": {"config": {"original": True}, "original": {}, "origin": receipt,
                                   "runtime": {"unchanged": True}}, "directory": source_dir,
                      "assets": {"initial_checkpoint": (root / "initial.safetensors", {"sha256": "initial"})},
                      "identities": identities, "experiments": {name: {} for name in identities},
                      "schedules": {name: object() for name in identities}}
            backend = f.make_backend(source, budget, stages)
            self.assertIsInstance(backend, h.ClusterBackend)
            self.assertIs(backend.identities, identities)
            self.assertIs(backend.schedules, source["schedules"])
            self.assertEqual(backend.directory, source_dir)
            job = backend.job("control-1729")
            self.assertEqual(job.run_id, source["worker_run_id"] + "-control-1729")
            self.assertIs(job.optimizer_budget, budget.account)
            self.assertEqual(job.fork_lineage, receipt["lineage"])

            def checkpoint(name, milestone):
                worker = runs / (source["worker_run_id"] + "-" + name)
                worker.mkdir(exist_ok=True)
                path = worker / f"checkpoint-step-{h.INITIAL_STEP + milestone}.safetensors"
                lineage = {"identity_sha256": h.digest(identities[name]), "initial_checkpoint_sha256": "initial"}
                header = h.canonical({"__metadata__": {"step": str(h.INITIAL_STEP + milestone),
                                                       "lineage": h.canonical(lineage)}}).encode()
                path.write_bytes(struct.pack("<Q", len(header)) + header)
                return path

            saved = checkpoint("control-1729", h.UPDATES)
            preserved = saved.read_bytes()
            with patch.object(runner, "train") as training:
                completed = backend.train("control-1729", h.UPDATES, float("inf"))
                training.assert_not_called()
            self.assertEqual(completed["step"], 7991)
            self.assertEqual(saved.read_bytes(), preserved)

            def train(job, identity, experiment, assets, schedule, *, until, lease_held, deadline):
                self.assertIs(identity, identities["treatment-1729"])
                self.assertIs(assets, source["assets"])
                self.assertIs(schedule, source["schedules"]["treatment-1729"])
                self.assertIs(job.optimizer_budget, budget.account)
                self.assertTrue(lease_held)
                self.assertEqual(deadline, float("inf"))
                checkpoint("treatment-1729", until)

            with patch.object(runner, "train", side_effect=train) as training:
                backend.train("treatment-1729", h.MIDPOINT, float("inf"))
                training.assert_called_once()


class SourceAdmissionTests(unittest.TestCase):
    """Run real admission authentication against tiny synthetic files, without model data."""

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(h.canonical(value) + "\n")
        return h.file_record(path, relative_to=self.root)

    def checkpoint(self, path, step, lineage):
        path.parent.mkdir(parents=True, exist_ok=True)
        header = h.canonical({"__metadata__": {"step": str(step), "lineage": h.canonical(lineage)}}).encode()
        path.write_bytes(struct.pack("<Q", len(header)) + header)
        return h.file_record(path, relative_to=self.root)

    def setUp(self):
        self.stack = ExitStack()
        self.root = Path(self.stack.enter_context(TemporaryDirectory()))
        self.stack.enter_context(patch.object(f, "ROOT", self.root))
        self.stack.enter_context(patch.object(h, "ROOT", self.root))
        self.runs, self.source_id = self.root / "runs", "source"
        self.directory = f.source_directory(self.runs, self.source_id)
        self.directory.mkdir(parents=True)
        self.worker_id = f"{self.source_id}-{h.REFERENCE_REVISION}-{h.CHECKPOINT_IO_RECOVERY}"
        code_files = []
        for name in ("worker.py", "history_parity.py", "model.py", "finish_history.py"):
            path = self.root / name
            path.write_text(f"synthetic {name} source\n")
            code_files.append(h.file_record(path, relative_to=self.root))
        self.code = {"files": code_files, "git_commit": "original", "dirty": False}
        self.identities = {name: {"name": name, "code": deepcopy(self.code)}
                           for name in ("pilot", "control-1729", "treatment-1729")}
        self.origin_lineage = {"source": "original-course"}
        initial = self.checkpoint(self.root / "initial.safetensors", 7205, self.origin_lineage)
        assets = {"initial_checkpoint": initial}
        for name in ("base", "tokenizer", "history_replay", "history_template", "registry", "history_train_specs"):
            assets[name] = self.write(self.root / "assets" / f"{name}.json", {"synthetic": name})
        fixed = [{"case_id": "f1", "panel": "fixed_prefix", "cohort": "supported", "family_id": "fixed",
                  "bootstrap_family_id": "fixed", "reference_eligible": True, "dialogue_id": None}]
        closed = [{"case_id": f"c{index}", "panel": "closed_loop", "cohort": cohort, "family_id": f"closed-{index}",
                   "bootstrap_family_id": f"closed-{index}", "reference_eligible": False, "dialogue_id": dialogue}
                  for index, (cohort, dialogue) in enumerate((("supported", None), ("supported", "d1"),
                                                            ("outside_scope", None)), 1)]
        assets["history_dev"] = self.write(self.root / "assets" / "history_dev.json",
            {"panels": {"fixed_prefix": fixed, "closed_loop": closed, "challenge": []}})
        self.config = {"assets": {"history_dev": {"path": assets["history_dev"]["path"]}},
                       "evaluation": {"fixed_prefix": 1, "closed_loop": 3, "challenge": 0, "supported": 2,
                                      "outside_scope": 1, "dialogues": 1, "standalone": 1, "exact_reference": 1}}
        self.original = {"synthetic_original": True}
        self.account = old_account()
        self.budget_path = self.runs / self.source_id / "budget.json"
        budget_record = self.write(self.budget_path, self.account)
        parent = self.write(self.root / "parent.json", {"original": "preserved"})
        revision = {"id": h.REFERENCE_REVISION, "parent_identity_sha256": self.account["identity_sha256"],
                    "parent_manifest": parent, "failures": [], "diagnostic_evidence": {},
                    "io_recovery": {"id": h.CHECKPOINT_IO_RECOVERY, "parent_manifest": parent, "failures": []},
                    "shared_budget": {"path": budget_record["path"],
                                      "identity_sha256": self.account["identity_sha256"], "maximum_seconds": h.MAX_SECONDS},
                    "budget_at_revision": deepcopy(self.account)}
        self.manifest = {"config": self.config, "original": self.original, "assets": assets,
                         "code": self.code, "origin": {"checkpoint": initial, "lineage": self.origin_lineage},
                         "runtime": {"synthetic": "four-gpu-allocation"}, "revision": revision}
        self.write(self.directory / "experiment.json", self.manifest)
        self.binding = h.digest(self.manifest)
        self.write(self.directory / "identities.json", self.identities)
        schedules, totals = schedule_fixture()
        self.write(self.directory / "token-budget.json", totals)
        train_path = self.root / "train.jsonl"
        train = (train_path, self.write(train_path, {"synthetic": "TRAIN fixture"}))
        prepared = {"collection_receipt": {"fingerprint": {
            key: assets[name]["sha256"] for key, name in (("checkpoint_sha256", "initial_checkpoint"),
                ("registry_sha256", "registry"), ("tokenizer_sha256", "tokenizer"),
                ("specs_sha256", "history_train_specs"))}}}
        self.parity = {"passed": True, "binding": {
            key: h.file_record(self.root / assets[name]["path"]) for key, name in
            (("checkpoint", "initial_checkpoint"), ("base", "base"), ("tokenizer", "tokenizer"),
             ("probes", "history_replay"), ("native_template", "history_template"))}}
        self.parity["binding"].update(implementation=h.file_record(self.root / "history_parity.py"),
            runtime=h.file_record(self.root / "model.py"), reference_version=f.history_parity.REFERENCE_VERSION,
            reference_policy=f.history_parity.reference_policy_record(h.REFERENCE_POLICY),
            tolerances=f.history_parity.TOLERANCES)
        self.write(self.directory / "parity" / "parity.json", self.parity)
        self.arm_lineages = {name: {"identity_sha256": h.digest(identity),
                            "initial_checkpoint_sha256": initial["sha256"]} for name, identity in self.identities.items()}
        measurements = [{"seconds": 1.0}, {"seconds": 1.0}]
        pilot = {"checkpoint": self.checkpoint(self.root / "pilot.safetensors", 7207, self.arm_lineages["pilot"]),
                 "measurements": measurements}
        observations = self.directory / "evaluations" / "baseline" / "observations.jsonl"
        observations.parent.mkdir(parents=True)
        rows = [dict(row, success=True, critical_status="none_observed", reference_success=row["reference_eligible"])
                for row in fixed + closed]
        observations.write_text("".join(h.canonical(row) + "\n" for row in rows))
        def summary(supported, outside, exact):
            metric = lambda value: {"passed": value, "planned": value}
            return {"supported": metric(supported), "outside_scope": metric(outside), "exact_reference": metric(exact),
                    "whole_dialogues": metric(1), "standalone": metric(1), "critical_types": {},
                    "critical_review_required": 0, "critical_detected": 0, "missing": 0}
        score = {"observed_records": 4, "infrastructure_failures": 0,
                 "panels": {"fixed_prefix": summary(1, 0, 1), "closed_loop": summary(2, 1, 0)}}
        score["summary"] = score["panels"]["closed_loop"]
        timings = observations.parent / "timings.jsonl"
        timings.write_text('{"kind":"load","seconds":1.0}\n')
        baseline = {"checkpoint": initial, "score": score, "seconds": 1.0,
                    "observations": h.file_record(observations, relative_to=self.root),
                    "timings": h.file_record(timings, relative_to=self.root)}
        for name, result in (("official-parity", self.parity), ("prepare", prepared), ("pilot", pilot), ("baseline", baseline)):
            self.write(self.directory / "stages" / f"{name}.json", {"identity_sha256": self.binding, "result": result})
        self.write(self.directory / "result.json", {"identity_sha256": self.binding, "status": "budget_stopped",
                                                    "baseline": baseline, "pairs": []})
        for name, step in (("control-1729", 7991), ("treatment-1729", 7598)):
            worker = self.runs / (self.worker_id + "-" + name)
            self.checkpoint(worker / "checkpoint.safetensors", step, self.arm_lineages[name])
            saved = self.checkpoint(worker / f"checkpoint-step-{step}.safetensors", step, self.arm_lineages[name])
            (worker / "events.jsonl").write_text('{"event":"saved"}\n')
            if name == "control-1729":
                self.partial_input = self.directory / "evaluations" / name / "input.json"
                self.write(self.partial_input, {"run": self.identities[name], "checkpoint": saved, "known": False})
                (self.partial_input.parent / "observations.jsonl").write_text('{"case_id":"saved"}\n')
        declared = {name: (self.root / value["path"], value) for name, value in assets.items()}
        self.stack.enter_context(patch.object(h, "load", return_value=(self.config, self.original, declared)))
        self.stack.enter_context(patch.object(h, "validate_prepared", return_value=(train, schedules)))
        def current_identities(*unused):
            identities = deepcopy(self.identities)
            for identity in identities.values():
                identity["code"]["git_commit"] = "new-completion-publication"
                identity["code"]["dirty"] = True
            return identities, {name: {} for name in identities}
        self.stack.enter_context(patch.object(h, "make_identities", side_effect=current_identities))
        self.stack.enter_context(patch.object(runner, "require_pilot", return_value=measurements))

    def tearDown(self):
        self.stack.close()

    def load(self):
        return f.load_source(self.runs, self.source_id)

    def test_real_admission_accepts_exact_source_with_new_git_metadata_without_writes(self):
        before = {path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}
        source = self.load()
        self.assertEqual(source["identities"], self.identities)
        self.assertEqual(source["budget"], self.account)
        self.assertEqual(source["worker_run_id"], self.worker_id)
        self.assertTrue(source["result_was_present"])
        self.assertEqual({path: path.read_bytes() for path in self.root.rglob("*") if path.is_file()}, before)

    def test_real_admission_rejects_frozen_byte_parity_budget_and_partial_input_changes(self):
        mutations = [
            (self.root / "worker.py", b"changed frozen worker"),
            (self.directory / "parity" / "parity.json", h.canonical({**self.parity, "passed": False}).encode()),
            (self.budget_path, h.canonical({**self.account, "charged_updates": 791}).encode()),
            (self.partial_input, h.canonical({**h.read(self.partial_input), "known": True}).encode()),
        ]
        for path, changed in mutations:
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.write_bytes(changed)
                with self.assertRaises(ValueError):
                    self.load()
                path.write_bytes(original)
        self.load()

    def snapshot(self):
        source = self.load()
        output = self.directory / f.COMPLETION_NAME
        output.mkdir()
        manifest = {"policy": f.POLICY, "source_manifest": f.record(self.directory / "experiment.json"),
                    "source_directory": str(self.directory.relative_to(self.root)), "inherited_budget": source["budget"],
                    "source_snapshot": f.snapshot_source(source), "code": f.code_record(self.root, ["finish_history.py"])}
        self.write(output / "manifest-binding.json", {"sha256": h.digest(manifest)})
        return source, manifest, output

    def test_snapshot_allows_append_and_advanced_checkpoint_but_pins_immutable_source(self):
        source, manifest, output = self.snapshot()
        worker = self.runs / (self.worker_id + "-treatment-1729")
        with (worker / "events.jsonl").open("a") as stream:
            stream.write('{"event":"advanced"}\n')
        self.checkpoint(worker / "checkpoint.safetensors", 7599, self.arm_lineages["treatment-1729"])
        f.validate_completion(manifest, source, output)
        (worker / "checkpoint-step-7598.safetensors").write_bytes(b"changed admitted checkpoint")
        with self.assertRaises(ValueError):
            f.validate_completion(manifest, source, output)

    def test_snapshot_rejects_same_cursor_byte_change_and_rollback(self):
        source, manifest, output = self.snapshot()
        latest = self.runs / (self.worker_id + "-treatment-1729") / "checkpoint.safetensors"
        original = latest.read_bytes()
        latest.write_bytes(original + b"different tensors at same cursor")
        with self.assertRaisesRegex(ValueError, "without advancing"):
            f.validate_completion(manifest, source, output)
        self.checkpoint(latest, 7597, self.arm_lineages["treatment-1729"])
        with self.assertRaisesRegex(ValueError, "rolled back"):
            f.validate_completion(manifest, source, output)
        self.checkpoint(latest, 7599, {"identity_sha256": "foreign", "initial_checkpoint_sha256": "foreign"})
        with self.assertRaisesRegex(ValueError, "lineage"):
            f.validate_completion(manifest, source, output)


class ActualEvaluatorReplayTests(unittest.TestCase):
    """Exercise the unchanged evaluator's real on-disk prefix recovery without a model."""

    def interrupted_prefix(self, output, replies):
        config = h.read(REPO / "experiments-history.json")
        original = h.read(REPO / "experiments.json")
        command = ["node", str(REPO / "evaluate-actual.mjs"),
                   str(REPO / original["assets"]["registry"]["path"]),
                   str(REPO / config["assets"]["history_dev"]["path"]),
                   str(output), h.canonical(config["evaluation"]), "0"]
        process = subprocess.Popen(command, cwd=REPO, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, encoding="utf-8")
        requests = []
        try:
            for index in range(replies + 1):
                line = process.stdout.readline()
                if not line:
                    self.fail(f"Evaluator exited before request: {process.stderr.read()}")
                request = json.loads(line)
                self.assertEqual(request["kind"], "generate")
                requests.append(request)
                if index < replies:
                    # A synthetic provider failure avoids tokenization and model execution.
                    process.stdin.write(json.dumps({"error": "inference_failed"}) + "\n")
                    process.stdin.flush()
            process.stdin.close()
            process.stdin = None
            _, error = process.communicate(timeout=30)
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("saved observations remain resumable", error)
            return requests
        finally:
            if process.poll() is None:
                process.kill()
            process.wait(timeout=30)
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    stream.close()

    def test_partial_dev_replays_saved_prefix_and_requests_only_next_case(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary) / "observations.jsonl"
            before = self.interrupted_prefix(output, 3)
            preserved = output.read_bytes()
            self.assertEqual(len(preserved.splitlines()), 3)
            with output.open("ab") as stream:
                stream.write(b'{"torn_tail":')
            resumed = self.interrupted_prefix(output, 1)
            self.assertEqual(resumed[0], before[-1])
            self.assertTrue(output.read_bytes().startswith(preserved))
            rows = h.read_observations(output)
            self.assertEqual(len(rows), 4)
            self.assertEqual(len({row["case_id"] for row in rows}), 4)
            self.assertEqual([row["case_id"] for row in rows],
                             [request["case_id"] for request in before])
            self.assertTrue(all(row["generation"]["error"] == "inference_failed" for row in rows))


if __name__ == "__main__":
    unittest.main()
