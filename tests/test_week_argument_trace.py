"""Use native saved DEV155 errors to exercise schema and state checks."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from diagnostics.week_argument_trace import render_arguments, trace_argument
from eval_tools.io import registry_at
from evaluation.functiongemma import encodeCalls
from evaluation.session import applyModelOutput, createSession


ROOT = Path(__file__).resolve().parents[1]


def call(name, **arguments):
    return {"name": name, "arguments": arguments}


class WeeklyArgumentTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry, _ = registry_at(ROOT / "assets/registry.json")

    def setUp(self):
        self.before = createSession(self.registry, {"dialogueVersion": 2, "sessionId": "argument-fixture"})

    def save_exchange(self, calls, cid="fixture"):
        after = applyModelOutput(self.registry, self.before, "Fixture question.", encodeCalls(calls))
        return {"case_id": cid, "status": "observed", "before": self.before, "after": after}

    def trace(self, calls):
        record = self.save_exchange(calls)
        return trace_argument(self.registry, record, record["after"]["exchanges"][-1])

    def test_missing_argument_in_domain_contract(self):
        result = self.trace([call("get_default_value")])
        self.assertIn("compile | call 1 get_default_value", result)
        self.assertIn("get_default_value function contract | $.subject=<missing>", result)
        self.assertIn('rule {"required":true}', result)

    def test_bad_reference_in_dialogue_tool(self):
        self.before = self.save_exchange([call("get_default_value", subject="TR")])["after"]
        result = self.trace([call("get_default_value", subject="TR"), call("repeat_answer", target="t99.c1")])
        self.assertIn("compile | call 2 repeat_answer", result)
        self.assertIn('dialogue tool schema | $.target="t99.c1"', result)
        self.assertIn('"enum":["t1.c1"]', result)

    def test_runtime_rejects_noop_replacement(self):
        self.before = self.save_exchange([call("get_default_value", subject="TR")])["after"]
        result = self.trace([call("get_default_value", subject="TR", _replace="t1.c1")])
        self.assertIn("execute | call 1", result)
        self.assertIn("Replacement does not change the call.", result)
        self.assertIn('call arguments {"_replace":"t1.c1","subject":"TR"}', result)

    def test_domain_contract_is_checked_after_revision(self):
        self.before = self.save_exchange([call("get_default_value", subject="TR")])["after"]
        result = self.trace([call("change_get_default_value", target="t1.c1", changes={}, remove=["subject"])])
        self.assertIn("execute | call 1 change_get_default_value", result)
        self.assertIn("get_default_value function contract | $.subject=<missing>", result)

    def test_only_observed_supported_regressions_are_counted(self):
        bad = self.save_exchange([call("get_default_value")], "new")
        blocked = {**bad, "case_id": "blocked", "status": "blocked"}
        passed = {"success": True, "critical_status": "none_observed"}
        review = {"success": False, "critical_status": "review_required"}
        cids = ("new", "blocked")
        saved = {"baseline": dict.fromkeys(cids, passed), "matched": dict.fromkeys(cids, review),
                 "full": dict.fromkeys(cids, passed)}
        plan = {cid: {"cohort": "supported"} for cid in cids}
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy155.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in (bad, blocked)) + "\n")
            before = path.read_bytes()
            output = render_arguments(self.registry, plan, saved, {"matched": [path], "full": []})
            self.assertEqual(path.read_bytes(), before)
        self.assertIn("matched: 1 of 2 losses recorded invalid_arguments", output)
        self.assertTrue(any("new | grade=review_required" in line for line in output))
        self.assertFalse(any("  blocked |" in line for line in output))
