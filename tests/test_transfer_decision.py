import copy
import unittest
from unittest.mock import patch

from training import transfer_decision as policy


def candidate(passed):
    metric = lambda yes, total: {"passed": yes, "planned": total}
    summary = {"supported": metric(passed, 20), "whole_dialogues": metric(passed // 4, 5),
               "standalone": metric(3, 4), "outside_scope": metric(3, 4), "critical_types": {},
               "critical_detected": 0, "critical_review_required": 2, "missing": 0}
    rows = {str(i): {"case_id": str(i), "family_id": str(i // 2), "bootstrap_family_id": str(i // 2),
                      "cohort": "supported", "panel": "closed_loop", "success": i < passed,
                      "critical_status": "none_observed"} for i in range(20)}
    score = {"summary": summary, "infrastructure_failures": 0}
    return {"scores": {k: copy.deepcopy(score) for k in ("fresh", "legacy155", "legacy1250")}}, rows


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.object(policy, "BOOTSTRAP_REPEATS", 200))
        self.parent, _ = candidate(9)
        self.a, self.a_rows = candidate(10)
        self.b, self.b_rows = candidate(18)

    def compare(self):
        return policy.compare_pair(self.parent, self.a, self.b, self.a_rows, self.b_rows, 1729)

    def test_paired_gain_keeps_unresolved_and_denominators(self):
        result = self.compare()
        self.assertTrue(result["positive_signal"])
        self.assertAlmostEqual(result["supported_gain"], .4)
        self.assertEqual(result["paired_95pct"]["families"], 10)
        self.assertEqual(result["unresolved"]["contrast"]["fresh"], 2)
        self.b_rows["0"]["success"] = False
        with self.assertRaisesRegex(ValueError, "reproduce"):
            self.compare()

    def test_guards_catch_regressions_and_infrastructure(self):
        self.b["scores"]["legacy1250"]["summary"]["outside_scope"]["passed"] = 2
        result = self.compare()
        self.assertFalse(result["positive_signal"])
        self.assertFalse(result["guards"]["legacy1250/all/outside_scope/ordinary"])
        self.b["scores"]["legacy155"]["infrastructure_failures"] = 1
        self.assertFalse(self.compare()["infrastructure_clean"])

    def test_unpaired_families_rejected(self):
        self.b_rows["0"]["bootstrap_family_id"] = "different"
        with self.assertRaisesRegex(ValueError, "blocks differ"):
            self.compare()
