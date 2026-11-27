"""Compare saved verdicts across legacy observation formats."""

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from diagnostics import inspect_week_failures as failures


class SavedFailuresTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()

    def write(self, relative, data):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = data if isinstance(data, bytes) else json.dumps(data).encode()
        path.write_bytes(raw)
        return {"path": relative, "bytes": len(raw), "sha256": sha256(raw).hexdigest()}

    def journal(self, relative, *records):
        return self.write(relative, b"".join(json.dumps(row).encode() + b"\n" for row in records))

    def test_saved_stage_receipt_checks_the_original_bytes(self):
        directory = self.root / "runs" / failures.RUN_ID
        folder = f"runs/{failures.RUN_ID}/full/"
        model = {"checkpoint": {"sha256": "ckpt"}, "scores": {"legacy155": {"observed_records": 155}}}
        binding = failures.digest({"manifest": "manifest", "model": "full", "checkpoint": model["checkpoint"]})
        self.write(folder + "input.json", {"binding": binding, "checkpoint": model["checkpoint"]})
        evidence = self.journal(folder + "legacy155.jsonl", {"case_id": "known"})
        stage = {"binding": binding, "observations": [evidence], "score": model["scores"]["legacy155"]}
        self.write(folder + "legacy155.json", dict(stage, receipt_sha256=failures.digest(stage)))
        args = (self.root, directory, "full", "legacy155", model, "manifest")
        self.assertEqual(failures.stage_files(*args), [self.root / evidence["path"]])
        self.journal(folder + "legacy155.jsonl", {"case_id": "other"})
        with self.assertRaisesRegex(ValueError, "mismatch"):
            failures.stage_files(*args)

    def test_known_dev_uses_the_saved_judgment_including_blocked_turns(self):
        case = {"case_id": "k", "cohort": "supported", "dialogue_id": None}
        grade = {
            "case_id": "k",
            "success": False,
            "critical_status": "none_observed",
            "reasons": ["blocked_continuation"],
        }
        record = {"case_id": "k", "result": grade, "status": "blocked", "after": {"exchanges": []}}
        self.journal("known.jsonl", record)
        score = failures.summarizeEvaluation([case], [grade])
        rows = failures.observations([self.root / "known.jsonl"], {"k": case}, "legacy155", score)
        self.assertEqual(rows["k"]["reason"], "blocked_continuation")
        score["supported"]["passed"] = 1
        with self.assertRaisesRegex(ValueError, "supported does not reproduce"):
            failures.observations([self.root / "known.jsonl"], {"k": case}, "legacy155", score)

    def test_dev1250_counts_closed_loop_from_full_and_compact_observations(self):
        case = {"case_id": "a", "panel": "closed_loop", "cohort": "supported", "family_id": "f", "turn_index": 0}
        other = dict(case, case_id="b", panel="fixed_prefix")
        grade = {"success": False, "critical_status": "detected", "critical_category": "wrong_reference"}
        plan = {"b": other, "a": case}
        summary = failures.summarizeEvaluation([case], [dict(case, **grade)])
        for schema in ({"after": {"exchanges": [{}]}}, {"delivered": {}}):
            self.journal("actual.jsonl", dict(case, **grade, **schema), dict(other, **grade, **schema))
            values = failures.observations([self.root / "actual.jsonl"], plan, "legacy1250", summary)
            self.assertEqual(list(values), ["a"])
            self.assertEqual(values["a"]["reason"], "wrong_reference")

    def test_render_counts_gains_losses_and_flags_relative_to_baseline(self):
        case = {
            "case_id": "a",
            "cohort": "supported",
            "question": "Which reply?",
            "policy": {"kind": "quote"},
            "authoring_family_id": "quote",
        }
        good = {
            "success": True,
            "critical_status": "none_observed",
            "reason": "none",
            "history": "same",
            "question": case["question"],
            "expected": [],
            "selected": [],
        }
        bad = dict(good, success=False, critical_status="detected", reason="wrong_reference", selected=["t2.c1"])
        plans = {split: {"a": case} for split in failures.SPLITS}
        saved = {
            split: {"baseline": {"a": good}, "matched": {"a": bad}, "full": {"a": good}}
            for split in failures.SPLITS
        }
        result = failures.render(plans, saved)
        self.assertIn("DEV155 matched | gained 0, lost 1", result)
        self.assertIn("DEV1250 closed loop matched | 1 newly flagged, 0 no longer flagged", result)
        self.assertIn("selected=['t2.c1']", result)
        self.assertIn("shared by both arms: 0", result)
