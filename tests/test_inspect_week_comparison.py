"""Check the terminal summary against the saved weekly report format."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from diagnostics import inspect_week_comparison as report


class WeeklyReportTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.directory = self.root / "runs" / report.RUN_ID
        self.directory.mkdir(parents=True)
        self.manifest = {"protocol": report.RUN_ID, "seed": 1729}
        self.write("experiment.json", self.manifest)
        self.result = {
            "identity_sha256": report.digest(self.manifest),
            "status": "complete",
            "TEST_opened": False,
            "comparison": {
                "full_minus_matched": -1 / 143,
                "paired_95pct": {"lower": -0.085, "upper": 0.08},
                "infrastructure_clean": True,
            },
            "models": {
                name: {"scores": self.scores(passed)}
                for name, passed in zip(report.MODELS, (90, 101, 100), strict=True)
            },
        }

    def write(self, filename, value):
        (self.directory / filename).write_text(json.dumps(value), encoding="utf-8")

    def scores(self, passed):
        def summary(successes, planned):
            return {
                "supported": {"passed": successes, "planned": planned},
                "whole_dialogues": {"passed": 0, "planned": 24},
                "outside_scope": {"passed": 17, "planned": 18},
                "critical_detected": 2,
                "critical_review_required": 13,
                "missing": 0,
            }

        closed = summary(passed, 143)
        return {
            "fresh": {
                "observed_records": 233,
                "summary": closed,
                "panels": {"closed_loop": closed, "challenge": summary(32, 72)},
            },
            "legacy155": {"observed_records": 155, "summary": summary(72, 131)},
            "legacy1250": {"observed_records": 1250, "summary": summary(507, 834)},
        }

    def test_renders_all_panels_using_only_the_two_saved_json_files(self):
        self.write("result.json", self.result)
        rendered = report.render(report.load_report(self.root))
        self.assertEqual(len(rendered.splitlines()), 15)
        for value in ("90/143", "101/143", "100/143", "32/72", "72/131", "507/834", "-0.7 pp"):
            self.assertIn(value, rendered)
        self.assertEqual(sorted(p.name for p in self.directory.iterdir()), ["experiment.json", "result.json"])

    def test_rejects_a_report_from_another_experiment_or_with_an_incomplete_panel(self):
        self.write("result.json", dict(self.result, identity_sha256="different"))
        with self.assertRaisesRegex(ValueError, "original manifest"):
            report.load_report(self.root)
        self.result["models"]["full"]["scores"]["legacy1250"]["observed_records"] = 1249
        self.write("result.json", self.result)
        with self.assertRaisesRegex(ValueError, "full/legacy1250"):
            report.load_report(self.root)
