"""Exercise the compact transfer report with planned cases and synthetic saved verdicts."""
from pathlib import Path
import json
import re
import unittest

from inspect_transfer import report

ROOT = Path(__file__).resolve().parent
RAW_MARKER = "UNEXPURGATED_DELIVERED_PAYLOAD"
RETRY_GAIN = "sxdev-live-080"
RETRY_LOSS = "sxdev-live-084"
AMBIGUOUS_GAIN = "sxdev-live-108"
QUOTE_LOSS = "sxdev-live-068"
SHARED_GAIN = "sxdev-live-001"
SHARED_LOSS = "sxdev-live-002"
OUTSIDE = "sxdev-live-051"
CHALLENGE = "sxdev-fixed-001"


def metric(passed, planned):
    return {"passed": passed, "planned": planned, "accuracy": passed / planned if planned else None}


def observation(row, failures, different_history):
    case_id = row["case_id"]
    reason = failures.get(case_id)
    status = {"wrong_reference": "detected", "semantic_unresolved": "review_required"}.get(reason, "none_observed")
    name = "clarify_reference" if row["policy"]["kind"] == "ambiguous" else "get_default_value"
    return {**row, "success": reason is None, "critical_status": status,
            "critical_category": reason if status == "detected" else None,
            "failure_category": reason, "before_visible_history_sha256": "b"*64 if different_history else "a"*64,
            "policy_branch": "present", "expected_model_call_batches": [[{"name": name, "arguments": {}}]],
            "generation": {"raw_text": "<start_function_call>call:get_default_value{}<end_function_call>",
                           "raw_ids": [50]}, "selected_reference_targets": [],
            "delivered": {"assistant": RAW_MARKER * 25, "outcome": "committed", "error_code": None}}


def summary(rows):
    def count(items):
        return metric(sum(row["success"] for row in items), len(items))
    dialogues = {r["dialogue_id"] for r in rows if r["dialogue_id"]}
    passed = sum(all(r["success"] for r in rows if r["dialogue_id"] == d) for d in dialogues)
    return {"supported": count([r for r in rows if r["cohort"] == "supported"]),
            "outside_scope": count([r for r in rows if r["cohort"] == "outside_scope"]),
            "standalone": count([r for r in rows if r["cohort"] == "supported" and not r["dialogue_id"]]),
            "whole_dialogues": metric(passed, len(dialogues)),
            "critical_detected": sum(r["critical_status"] == "detected" for r in rows),
            "critical_review_required": sum(r["critical_status"] == "review_required" for r in rows),
            "missing": 0}


def fixture():
    spec = json.loads((ROOT / "assets/transfer-dev-v1.json").read_text())
    plan = {r["case_id"]: r for values in spec["panels"].values() for r in values}
    losses = {
        "baseline": {AMBIGUOUS_GAIN, QUOTE_LOSS, RETRY_LOSS, SHARED_GAIN, CHALLENGE},
        "ordinary-1729": {RETRY_GAIN, AMBIGUOUS_GAIN, SHARED_LOSS, OUTSIDE},
        "contrast-1729": {QUOTE_LOSS, RETRY_LOSS, SHARED_LOSS, CHALLENGE},
    }
    rows, models = {}, {}
    for index, (arm, failed) in enumerate(losses.items()):
        failures = {k: "execution_error" for k in failed}
        if arm == "contrast-1729":
            failures.update({QUOTE_LOSS: "wrong_reference", RETRY_LOSS: "semantic_unresolved"})
        rows[arm] = {k: observation(r, failures, arm == "contrast-1729" and k in (AMBIGUOUS_GAIN, RETRY_GAIN))
                     for k, r in plan.items()}
        panels = {panel: summary([r for r in rows[arm].values() if r["panel"] == panel]) for panel in spec["panels"]}
        scores = {"fresh": {"summary": panels["closed_loop"], "panels": panels, "observed_records": len(plan)}}
        for split, denom in (("legacy155", 131), ("legacy1250", 834)):
            scores[split] = {"summary": {**panels["closed_loop"], "supported": metric(42+index, denom)}}
        models[arm] = {"scores": scores, "checkpoint": {"path": str(Path("runs") / arm / "checkpoint.safetensors")}}
    pair = {"seed": 1729, "arms": {"ordinary": models["ordinary-1729"], "contrast": models["contrast-1729"]},
            "decision": {"seed": 1729, "supported_gain": 0,
                         "paired_95pct": {"lower": -.06, "upper": .07, "families": 61}, "positive_signal": False}}
    return {"status": "not_promoted", "pairs": [pair], "baseline": models["baseline"], "release_ready": False}, plan, rows


def signed_pairs(line):
    return [(int(gained), int(lost)) for gained, lost in re.findall(r"\+(\d+)\s*/\s*-(\d+)", line)]


class TransferReportTests(unittest.TestCase):
    def setUp(self):
        self.result, self.plan, self.rows = fixture()

    def render(self):
        return report(self.result, self.plan, self.rows)

    def test_directional_pairs_join_by_case_and_separate_panels(self):
        text = self.render()
        self.rows["ordinary-1729"] = dict(reversed(self.rows["ordinary-1729"].items()))
        self.assertEqual(self.render(), text)
        for label, expected in {
            "contrast vs ordinary": [(2, 2), (2, 2), (1, 0), (0, 1)],
            "ordinary vs parent": [(3, 2), (2, 1), (0, 1), (1, 0)],
            "contrast vs parent": [(2, 1), (1, 0), (0, 0), (0, 0)],
        }.items():
            with self.subTest(label=label):
                line = next(line for line in text.splitlines() if label in line and signed_pairs(line))
                self.assertEqual(signed_pairs(line), expected)

    def test_policy_counts_show_cancelling_wins_losses(self):
        text = self.render()
        for policy, expected in (("primary/retry", [(1, 1), (1, 1), (0, 0)]),
                                 ("primary/ambiguous", [(1, 0), (0, 0), (1, 0)])):
            line = next(line for line in text.splitlines() if policy in line and signed_pairs(line))
            self.assertEqual(signed_pairs(line), expected)

    def test_report_includes_parent_quality_and_safe_examples(self):
        text = self.render()
        for token in ("parent 7991", "139/143", "140/143", "42/131", "44/834", "not_promoted",
                      "dialogues", "crit/review", "history=different", "history=same"):
            with self.subTest(token=token):
                self.assertIn(token, text)
        contrast_row = next(line.split() for line in text.splitlines() if line.startswith("contrast "))
        self.assertEqual(contrast_row[1:5], ["140/143", "22/24", "18/18", "1/1"])
        self.assertIn("different visible histories: +2/-0 of +2/-2", text)
        self.assertIn("review(semantic_unresolved)", text)
        self.assertIn(RETRY_GAIN, text)
        self.assertIn(RETRY_LOSS, text)
        self.assertNotIn("Seed 2718", text)
        self.assertNotIn(RAW_MARKER, text)
        self.assertLess(len(text.splitlines()), 65)
        self.assertLess(len(text), 9000)


if __name__ == "__main__":
    unittest.main()
