"""Read-only reporter tests using 32 synthetic observations; no model is loaded."""

from contextlib import ExitStack, redirect_stdout
from copy import deepcopy
from hashlib import sha256
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import inspect_reference as reporter


class ReportFixture(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.outer = Path(self.stack.enter_context(TemporaryDirectory()))
        self.root = self.outer / "checkout"
        self.root.mkdir()
        self.stack.enter_context(patch.object(reporter, "ROOT", self.root))
        self.directory = self.root / "runs" / reporter.RUN_ID
        self.directory.mkdir(parents=True)
        self.identity = {"policy": {"qualification_cases": 32, "transport_minimum": 30},
                         "code": {"git_commit": "historical-training-commit", "dirty": False}}
        self.case_ids = [f"qualification-{index:02d}" for index in range(32)]
        self.manifest = {"identity_sha256": reporter.digest(self.identity),
                         "train_panel": {"qualification_ids": self.case_ids},
                         "reference": {"fingerprint": {"sha256": "synthetic-reference-weights"}}}
        self.write(self.directory / "identity.json", self.identity)
        self.write(self.directory / "experiment.json", self.manifest)
        self.write(self.directory / "manifest-binding.json", {"sha256": reporter.digest(self.manifest)})
        self.build()

    def tearDown(self):
        self.stack.close()

    @staticmethod
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")

    def build(self, transport=30, supported=1):
        self.rows = []
        for index, case_id in enumerate(self.case_ids):
            gpu = int(sha256(case_id.encode()).hexdigest()[:8], 16) % 4
            worker = reporter.digest({"manifest": reporter.digest(self.manifest), "model": "reference",
                                      "phase": "qualification", "gpu": gpu,
                                      "weights": self.manifest["reference"]["fingerprint"]})
            self.rows.append({"case_id": case_id, "cohort": "supported", "semantic_eligible": True,
                "transport_valid": index < transport, "success": index < supported,
                "critical_status": "none_observed", "error_kind": None if index < supported else "wrong_answer",
                "generation": {"diagnostic_worker_sha256": worker, "finish_reason": "stop",
                               "raw_text": "synthetic model output", "raw_ids": [10, 50], "input_tokens": 12}})
        self.publish()

    def publish(self):
        paths = [self.directory / "reference" / "qualification" / f"train-{gpu}.jsonl" for gpu in range(4)]
        for gpu, path in enumerate(paths):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in self.rows
                if int(sha256(row["case_id"].encode()).hexdigest()[:8], 16) % 4 == gpu))
        supported = [row for row in self.rows if row["semantic_eligible"] and row["cohort"] == "supported"]
        transport_passed = sum(row["transport_valid"] is True for row in self.rows)
        supported_passed = sum(row["success"] is True and row["critical_status"] == "none_observed" for row in supported)
        counts = {"transport_valid": {"passed": transport_passed, "planned": len(self.rows),
                                      "accuracy": transport_passed / len(self.rows)},
                  "supported": {"passed": supported_passed, "planned": len(supported),
                                "accuracy": supported_passed / len(supported) if supported else None}}
        passed = transport_passed >= 30 and supported_passed >= 1
        self.qualification = {"identity_sha256": self.manifest["identity_sha256"],
                              "observations": [reporter.record(path) for path in paths],
                              "all_fixed_train_panel": counts, "passed": passed}
        receipt = self.directory / "stages" / "reference-transport.json"
        self.write(receipt, self.qualification)
        self.result = {"identity_sha256": self.manifest["identity_sha256"],
                       "status": "COMPLETE_DIAGNOSTIC" if passed else "TRAIN_FIT_COMPLETE_REFERENCE_NOT_QUALIFIED",
                       "reference_transport": self.qualification,
                       "artifacts": [reporter.record(path) for path in
                                     (self.directory / "experiment.json", self.directory / "manifest-binding.json", receipt)]}
        self.write(self.directory / "result.json", self.result)

    def bytes(self):
        return {path.relative_to(self.root): path.read_bytes() for path in self.root.rglob("*") if path.is_file()}


class AdmissionCountsTests(ReportFixture):
    def test_all_four_gate_outcomes_reproduce_exact_counts(self):
        for transport, supported in ((30, 1), (29, 1), (32, 0), (29, 0)):
            with self.subTest(transport=transport, supported=supported):
                self.build(transport, supported)
                report = reporter.load_report()
                self.assertEqual(report["passed"], transport >= 30 and supported >= 1)
                self.assertEqual(report["gates"], {"native_format": transport >= 30,
                                                 "complete_supported_answer": supported >= 1})
                self.assertEqual(report["counts"]["transport_valid"],
                                 {"passed": transport, "planned": 32, "accuracy": transport / 32})
                self.assertEqual(report["counts"]["supported"],
                                 {"passed": supported, "planned": 32, "accuracy": supported / 32})

    def test_critical_outside_scope_and_ineligible_successes_cannot_qualify_reference(self):
        self.build(32, 3)
        self.rows[0]["critical_status"] = "detected"
        self.rows[1]["cohort"] = "outside_scope"
        self.rows[2]["semantic_eligible"] = False
        self.publish()
        report = reporter.load_report()
        self.assertEqual(report["counts"]["supported"], {"passed": 0, "planned": 30, "accuracy": 0.0})
        self.assertFalse(report["gates"]["complete_supported_answer"])

    def test_saved_count_or_decision_cannot_override_observations(self):
        for changed in ("count", "decision"):
            with self.subTest(changed=changed):
                self.build(29, 0)
                if changed == "count":
                    self.qualification["all_fixed_train_panel"]["transport_valid"]["passed"] = 30
                else:
                    self.qualification["passed"] = True
                    self.result["status"] = "COMPLETE_DIAGNOSTIC"
                receipt = self.directory / "stages" / "reference-transport.json"
                self.write(receipt, self.qualification)
                self.result["reference_transport"] = self.qualification
                self.result["artifacts"][-1] = reporter.record(receipt)
                self.write(self.directory / "result.json", self.result)
                with self.assertRaises(ValueError):
                    reporter.load_report()


class SavedBindingTests(ReportFixture):
    def test_shards_missing_duplicate_or_unplanned_cases_are_rejected(self):
        for change in ("missing", "duplicate", "unplanned"):
            with self.subTest(change=change):
                self.build()
                if change == "missing":
                    self.rows.pop()
                elif change == "duplicate":
                    self.rows[-1] = deepcopy(self.rows[0])
                else:
                    row = self.rows[-1]
                    row["case_id"] = "not-planned"
                    gpu = int(sha256(row["case_id"].encode()).hexdigest()[:8], 16) % 4
                    row["generation"]["diagnostic_worker_sha256"] = reporter.digest({
                        "manifest": reporter.digest(self.manifest), "model": "reference", "phase": "qualification",
                        "gpu": gpu, "weights": self.manifest["reference"]["fingerprint"]})
                self.publish()
                with self.assertRaisesRegex(ValueError, "missing, duplicated or unplanned"):
                    reporter.load_report()

    def test_altered_shard_bytes_and_wrong_worker_binding_are_rejected(self):
        first = self.root / self.qualification["observations"][0]["path"]
        original = first.read_bytes()
        first.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "hash changed"):
            reporter.load_report()
        first.write_bytes(original)
        self.rows[0]["generation"]["diagnostic_worker_sha256"] = "another-worker"
        self.publish()
        with self.assertRaisesRegex(ValueError, "another reference worker"):
            reporter.load_report()

    def test_missing_shard_and_duplicate_shard_entry_are_rejected(self):
        path = self.root / self.qualification["observations"][0]["path"]
        path.unlink()
        with self.assertRaises(OSError):
            reporter.load_report()
        self.build()
        self.qualification["observations"][1] = self.qualification["observations"][0]
        receipt = self.directory / "stages" / "reference-transport.json"
        self.write(receipt, self.qualification)
        self.result["reference_transport"] = self.qualification
        self.result["artifacts"][-1] = reporter.record(receipt)
        self.write(self.directory / "result.json", self.result)
        with self.assertRaisesRegex(ValueError, "exactly the four"):
            reporter.load_report()

    def test_manifest_and_binding_must_be_pinned_by_terminal_artifacts(self):
        for name in ("experiment.json", "manifest-binding.json", "reference-transport.json"):
            with self.subTest(name=name):
                self.build()
                self.result["artifacts"] = [entry for entry in self.result["artifacts"]
                                             if Path(entry["path"]).name != name]
                self.write(self.directory / "result.json", self.result)
                with self.assertRaises(ValueError):
                    reporter.load_report()

    def test_altered_identity_or_manifest_binding_is_rejected(self):
        identity_path = self.directory / "identity.json"
        before = identity_path.read_bytes()
        self.write(identity_path, {**self.identity, "changed": True})
        with self.assertRaisesRegex(ValueError, "identities"):
            reporter.load_report()
        identity_path.write_bytes(before)
        self.write(self.directory / "manifest-binding.json", {"sha256": "different-manifest"})
        with self.assertRaisesRegex(ValueError, "identities"):
            reporter.load_report()

    def test_run_traversal_and_symlinked_shard_escape_are_rejected(self):
        for run_id in ("../outside", "/absolute", "a/b", "", "x" * 101):
            with self.subTest(run_id=run_id), self.assertRaises(ValueError):
                reporter.load_report(run_id)
        shard = self.root / self.qualification["observations"][0]["path"]
        outside = self.outer / "outside.jsonl"
        outside.write_bytes(shard.read_bytes())
        shard.unlink()
        shard.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "inside this checkout"):
            reporter.load_report()


class DisplayTests(ReportFixture):
    def test_first_two_examples_explain_both_failed_gates_despite_case_sort_order(self):
        self.build(29, 0)
        selected = reporter.select_examples(reporter.load_report(), 2)
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[0]["case_id"], "qualification-29")
        self.assertFalse(selected[0]["transport_valid"])
        self.assertEqual(selected[1]["case_id"], "qualification-00")
        self.assertTrue(selected[1]["transport_valid"])
        self.assertFalse(selected[1]["success"])

    def test_output_examples_are_bounded_json_escaped_and_saved_files_stay_unchanged(self):
        self.build(0, 0)
        raw = "\x1b[31m\nquoted: \"\\" + "A" * 1500 + "OMITTED-MIDDLE" + "B" * 3000
        for row in self.rows:
            row["generation"]["raw_text"] = raw
        self.rows[1]["error_kind"] = "parse\nerror\x1b"
        self.publish()
        before = self.bytes()
        stream = io.StringIO()
        with redirect_stdout(stream):
            reporter.show(reporter.load_report(), examples=2)
        output = stream.getvalue()
        self.assertEqual(output.count("Raw response (first "), 2)
        self.assertEqual(output.count("Raw response (last 200 characters):"), 2)
        self.assertIn("\\u001b", output)
        self.assertIn("\\n", output)
        self.assertNotIn("\x1b", output)
        self.assertNotIn("OMITTED-MIDDLE", output)
        self.assertLess(len(output), 5000)
        self.assertIn("Failed gates: native_format, complete_supported_answer", output)
        self.assertEqual(self.bytes(), before)

    def test_zero_examples_omits_raw_responses(self):
        self.build(29, 0)
        stream = io.StringIO()
        with redirect_stdout(stream):
            reporter.show(reporter.load_report(), examples=0)
        self.assertNotIn("Example ", stream.getvalue())
        self.assertNotIn("Raw response", stream.getvalue())

    def test_saved_report_does_not_depend_on_current_git_metadata_or_execute_commands(self):
        before = self.bytes()
        with patch("subprocess.run", side_effect=AssertionError("No Git or model subprocess is allowed")):
            report = reporter.load_report()
        self.assertTrue(report["passed"])
        self.assertEqual(self.identity["code"]["git_commit"], "historical-training-commit")
        self.assertEqual(self.bytes(), before)


if __name__ == "__main__":
    unittest.main()
