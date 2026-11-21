"""Saved-Qwen error inspection at the native compilation, execution and review boundaries."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
import unittest

from diagnostics.qwen_case_details import createCaseInspector
from eval_tools.diagnose_data import compactObservation, createTrainDiagnostics
from eval_tools.io import registry_at
from evaluation import api

ROOT = Path(__file__).resolve().parents[1]


def call(name, **arguments):
    return dict(name=name, arguments=arguments)


DEFAULT = call("get_default_value", subject="TR")


class CaseInspectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry, bundle = registry_at(ROOT / "assets/registry.json")
        cls.api = api()
        cls.diagnostic = createTrainDiagnostics(cls.registry)
        cls.inspect = staticmethod(createCaseInspector(cls.registry, bundle["contracts"]))

    def setUp(self):
        self.before = self.api.createSession(self.registry, dict(dialogueVersion=2, sessionId="fixture"))

    def ask(self, calls, question="Default TR?"):
        self.before = self.api.applyModelOutput(self.registry, self.before, question, self.api.encodeCalls(calls))

    def fixture(self, expected, actual, question="Fixture request for the declared operation."):
        row = dict(
            case_id="inspection-fixture",
            family_id="fixture-family",
            group="fixture",
            question=question,
            before=self.before,
            calls=expected,
            after=self.api.applyModelOutput(self.registry, self.before, question, self.api.encodeCalls(expected)),
            model_input=self.api.renderPrompt(self.registry, self.api, self.before, question),
        )
        item = self.diagnostic.prepare(dict(source="history", row=row))
        full = self.diagnostic.score(
            item, dict(provider_kind="reference", complete=True, raw_text=self.api.encodeCalls(actual))
        )
        return item, compactObservation(full), full

    def report(self, expected, actual):
        case, receipt, _ = self.fixture(expected, actual)
        return self.inspect(case, receipt)

    def test_dynamic_reference(self):
        self.ask([DEFAULT])
        result = self.report([DEFAULT], [DEFAULT, call("get_default_value", subject="TR", _replace="t99.c1")])
        trace = result["trace"]
        self.assertEqual(
            (trace["stage"], trace["call_index"], trace["error"]["message"]),
            ("compile", 1, "$._replace: value is not in the declared domain"),
        )
        item = next(s for s in result["schemas"] if s["source"] == "dynamic_model_tool")
        self.assertFalse(item["valid"])
        self.assertEqual((item["path"], item["fragment"]["enum"]), ("$._replace", ["t1.c1"]))
        self.assertIs(result["scores_changed"], False)

    def test_required_only_in_domain(self):
        r = self.report([DEFAULT], [call("get_default_value")])
        self.assertEqual(r["trace"]["error"]["message"], "$: missing subject")
        exposed, domain = r["schemas"]
        self.assertTrue(exposed["valid"])
        self.assertFalse(domain["valid"])
        self.assertEqual((domain["fragment"]["required"], domain["fragment"]["field"]), (["subject"], "subject"))

    def test_clarification_metadata(self):
        r = self.report([DEFAULT], [call("get_default_value", subject="TR", _missing=["subject"])])
        self.assertEqual(
            r["trace"]["error"]["message"], "Missing-argument metadata requires explicit clarification."
        )
        self.assertTrue(r["schemas"][0]["valid"])

    def test_noop_replacement(self):
        self.ask([DEFAULT])
        r = self.report([DEFAULT], [call("get_default_value", subject="TR", _replace="t1.c1")])
        self.assertEqual(
            (r["trace"]["stage"], r["trace"]["call_index"], r["trace"]["error"]["message"]),
            ("execute", 0, "Replacement does not change the call."),
        )
        self.assertTrue(all(v["valid"] for v in r["schemas"]))

    def test_failed_retry_binding(self):
        self.before = self.api.recordProviderFailure(
            self.registry, self.before, "Original request.", "inference_failed"
        )
        retry = call("retry_exchange", target="e1")
        missing = call("get_value_guidance", subject="TR", preprocessing="fMRIPrep")
        r = self.report([retry, DEFAULT], [retry, missing])
        self.assertEqual(
            (r["trace"]["stage"], r["trace"]["call_index"], r["error_kind"]),
            ("execute", 1, "contract_not_implemented"),
        )
        contract = r["contract"]
        self.assertEqual(contract["domain_call"], missing)
        self.assertTrue(contract["schema_valid"])
        self.assertFalse(contract["exact_binding_present"])
        self.assertGreater(contract["same_function_count"], contract["same_function_subject_count"])
        self.assertGreater(contract["same_function_subject_count"], 0)
        self.assertLessEqual(len(contract["same_function_examples"]), 3)

    def test_change_retains_original_arguments(self):
        self.ask([call("get_value_guidance", subject="TR")], "Choose TR.")
        r = self.report(
            [DEFAULT], [call("change_get_value_guidance", target="t1.c1", changes={"preprocessing": "fMRIPrep"})]
        )
        self.assertEqual(r["trace"]["stage"], "execute")
        self.assertEqual(
            r["contract"]["domain_call"], call("get_value_guidance", subject="TR", preprocessing="fMRIPrep")
        )
        self.assertEqual([s["name"] for s in r["schemas"]], ["change_get_value_guidance", "get_value_guidance"])

    def test_group_attribution(self):
        self.ask(
            [call("get_value_guidance", _clarify=True, _each="subject")],
            "Choose values for unspecified parameters.",
        )
        self.assertEqual(self.before["exchanges"][-1]["outcome"], "committed")
        r = self.report(
            [DEFAULT], [call("get_value_guidance", _resolve="t1.p1", subject="TR", preprocessing="fMRIPrep")]
        )
        self.assertEqual(
            (r["trace"]["stage"], r["trace"]["call_index"], r["trace"]["error"]["code"]),
            ("execute", None, "invalid_arguments"),
        )
        self.assertIn("group_finalization", r["trace"]["attribution"])

    def test_pending_arguments(self):
        self.ask(
            [call("get_value_guidance", subject="TR", _clarify=True, _missing=["preprocessing"])],
            "Choose TR after unspecified preprocessing.",
        )
        r = self.report([DEFAULT], [call("get_value_guidance", _resolve="t1.p1", refinement_enabled=True)])
        self.assertEqual(
            r["trace"]["error"]["message"],
            "Completion may supply only missing arguments; changing known arguments requires an explicit replacement.",
        )
        self.assertEqual([(s["source"], s["valid"]) for s in r["schemas"]], [("dynamic_model_tool", True)])

    def test_full_review_alternatives(self):
        expected = [call("explain_item_count_limits", subject="concatenate_item", bound="maximum")]
        sample = self.fixture(expected, [call("get_default_value", subject="concatenate_item")])
        case, receipt, full = sample
        saved = deepcopy(receipt)
        r = self.inspect(case, receipt)
        rev = r["review"]
        self.assertEqual(r["critical_status"], "review_required")
        self.assertIs(rev["adjudicated"], False)
        self.assertEqual(r["trace"]["stage"], "committed")
        alternative = rev["alternatives"][0]
        self.assertFalse(alternative["delivered"]["equal"])
        self.assertFalse(alternative["semantic"]["equal"])
        self.assertTrue(alternative["semantic"]["entries"])
        self.assertEqual(alternative["calls"], expected)
        official = self.api.applyModelOutput(
            self.registry, self.before, case["question"], self.api.encodeCalls(expected)
        )
        self.assertGreater(len(official["exchanges"][-1]["assistant"]), 240)
        for obj, state in [(rev["actual"], full["after"]), (alternative["expected"], official)]:
            self.assertEqual(obj["delivered"], state["exchanges"][-1]["assistant"])
            self.assertEqual(obj["semantic"], state["semantic"])
        self.assertEqual(receipt, saved)

    def test_identical_text_different_action(self):
        self.ask([DEFAULT])
        case, receipt, full = self.fixture([DEFAULT], [call("repeat_answer", target="t1.c1")], "Answer afresh.")
        self.assertEqual(full["critical_status"], "review_required")
        review = self.inspect(case, receipt)["review"]
        self.assertTrue(review["alternatives"][0]["delivered"]["equal"])
        self.assertFalse(review["alternatives"][0]["semantic"]["equal"])
        self.assertFalse(review["adjudicated"])

    def test_tamper(self):
        case, receipt, _ = self.fixture([DEFAULT], [call("get_default_value")])
        with self.assertRaisesRegex(ValueError, "receipt does not reproduce"):
            self.inspect(case, dict(receipt, error_kind="contract_not_implemented"))
        with self.assertRaisesRegex(ValueError, "changed"):
            self.inspect(dict(case, question=case["question"] + "changed"), receipt)

    def test_jsonl_interface(self):
        case, receipt, _ = self.fixture([DEFAULT], [call("get_default_value")])
        proc = subprocess.run(
            [sys.executable, "-m", "diagnostics.qwen_case_details", str(ROOT / "assets/registry.json")],
            cwd=ROOT,
            input=json.dumps(dict(action="inspect", case=case, observation=receipt)) + "\n",
            text=True,
            capture_output=True,
            check=True,
        )
        result = json.loads(proc.stdout)
        self.assertEqual(
            (result["kind"], result["inspection"]["trace"]["error"]["code"]), ("inspection", "invalid_arguments")
        )


if __name__ == "__main__":
    unittest.main()
