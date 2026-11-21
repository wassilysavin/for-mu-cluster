"""Exercise native TRAIN evidence, semantic alternatives and immutable replay."""

import json
import unittest

from evaluation import api
from evaluation.functiongemma import encodeCalls
from evaluation.history_train import materializeScenario, renderPrompt
from evaluation.session import applyModelOutput, createSession
from eval_tools.diagnose_data import compactObservation, createTrainDiagnostics, materializeGeneration
from eval_tools.io import ROOT, records, registry_at


def call(name, **args):
    return {"name": name, "arguments": args}


def complete(calls):
    return {"provider_kind": "reference", "complete": True, "raw_text": encodeCalls(calls)}


class TrainDiagnosticsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry, _ = registry_at(ROOT / "assets/registry.json")
        cls.core = api()
        cls.diag = createTrainDiagnostics(cls.registry)

    def session(self, *batches):
        state = createSession(self.registry, {"dialogueVersion": 2, "sessionId": "diagnostic-unit"})
        for i, batch in enumerate(batches):
            state = applyModelOutput(self.registry, state, f"Request {i+1}", encodeCalls(batch))
        return state

    def prepare(self, before, calls, source="history", supply=True):
        question = "Respond to the specified request."
        row = {"case_id": "test-history", "question": question, "calls": calls}
        prompt = renderPrompt(self.registry, self.core, before, question)
        return self.diag.prepare({"source": source, "row": row, "prompt": prompt, **({"before": before} if supply else {})})

    def test_frozen_common_prompt_distinguishes_native_from_canonical(self):
        calls = [call("get_default_value", subject="TR")]
        item = self.prepare(self.session(), calls, source="common", supply=False)
        result = self.diag.score(item, {**complete(calls), "raw_text": "\n" + encodeCalls(calls) + "\n"})
        self.assertTrue(item["semantic_eligible"])
        self.assertFalse(result["native_literal_exact"])
        self.assertTrue(result["canonical_call_exact"] and result["semantic_success"])

    def test_unrestorable_common_history_has_no_semantic_claim(self):
        state = self.session([call("explain_purpose", subject="TR")])
        calls = [call("repeat_answer", target="t1.c1")]
        item = self.prepare(state, calls, "common", False)
        result = self.diag.score(item, complete(calls))
        self.assertFalse(item["semantic_eligible"])
        self.assertTrue(result["native_literal_exact"])
        self.assertIsNone(result["semantic_success"])
        self.assertIsNone(result["semantic_result"])
        self.assertTrue(self.prepare(state, calls, "common", True)["semantic_eligible"])

    def test_wrong_occurrence_rejected_despite_identical_text(self):
        batch = [call("explain_purpose", subject="TR")]
        item = self.prepare(self.session(batch, batch), [call("repeat_exchange", target="e1")])
        good = self.diag.score(item, complete([call("repeat_answer", target="t1.c1")]))
        bad = self.diag.score(item, complete([call("repeat_exchange", target="e2")]))
        self.assertTrue(good["semantic_success"])
        self.assertFalse(good["native_literal_exact"] or bad["semantic_success"])
        self.assertEqual(good["after"]["exchanges"][-1]["assistant"], bad["after"]["exchanges"][-1]["assistant"])
        self.assertEqual(bad["critical_status"], "detected")
        self.assertEqual(bad["semantic_result"]["critical_category"], "wrong_reference")

    def test_whole_reply_needs_all_parts(self):
        state = self.session([call("explain_purpose", subject="TR"), call("get_default_value", subject="TR")])
        item = self.prepare(state, [call("repeat_exchange", target="e1")])
        batch = [call("repeat_answer", target="t1.c1"), call("repeat_answer", target="t1.c2")]
        self.assertFalse(self.diag.score(item, complete(batch[:1]))["semantic_success"])
        self.assertTrue(self.diag.score(item, complete(batch))["semantic_success"])

    def test_explicit_replacement_matches_change(self):
        before = self.session([call("explain_purpose", subject="TR")])
        item = self.prepare(before, [call("change_explain_purpose", target="t1.c1", changes={"subject": "BIDS.basepath"})])
        good = self.diag.score(item, complete([call("explain_purpose", subject="BIDS.basepath", _replace="t1.c1")]))
        self.assertTrue(good["semantic_success"])
        self.assertFalse(good["canonical_call_exact"])
        wrong = self.diag.score(item, complete([call("explain_purpose", subject="BIDS.basepath")]))
        self.assertFalse(wrong["semantic_success"])
        self.assertEqual(wrong["semantic_result"]["critical_category"], "omitted_reference")

    def test_tampered_state_question_and_gold_rejected(self):
        before = self.session([call("explain_purpose", subject="TR")])
        calls = [call("repeat_exchange", target="e1")]
        item = self.prepare(before, calls)
        bad = json.loads(json.dumps(before))
        bad["exchanges"][0]["assistant"] += " altered"
        with self.assertRaisesRegex(ValueError, "does not reproduce"):
            self.diag.prepare({"source": "history", "before": bad, "prompt": item["prompt"],
                               "row": {"case_id": "bad", "calls": calls, "question": item["question"]}})
        with self.assertRaisesRegex(ValueError, "changed"):
            self.diag.score({**item, "gold_calls": []}, complete(calls))
        with self.assertRaisesRegex(ValueError, "question differs"):
            self.diag.prepare({"source": "common", "prompt": item["prompt"],
                               "row": {"case_id": "bad", "calls": calls, "question": "Different?"}})

    def test_incomplete_other_provider_keeps_foreign_ids(self):
        before = self.session()
        calls = [call("explain_purpose", subject="TR")]
        item = self.prepare(before, calls)
        incomplete = {**complete(calls), "complete": False, "raw_ids": [999999, 1000000], "finish_reason": "length"}
        result = self.diag.score(item, incomplete)
        after = result["after"]
        self.assertFalse(result["semantic_success"] or result["native_literal_exact"])
        ex = after["exchanges"][-1]
        self.assertEqual(ex["generation"]["token_ids"], incomplete["raw_ids"])
        self.assertEqual(ex["raw_output"], incomplete["raw_text"])
        self.assertEqual(ex["error_code"], "incomplete_model_output")
        self.assertEqual(after["semantic"], before["semantic"])
        failed = materializeGeneration(self.registry, before, item["question"], {"error": "inference_failed"})
        self.assertEqual(failed["exchanges"][-1]["outcome"], "provider_error")

    def test_every_preauthored_group_has_reproducible_gold(self):
        groups = set()
        for spec in records(ROOT / "assets/history-train-specs.jsonl"):
            if spec["group"] in groups:
                continue
            row = materializeScenario(self.registry, self.core, spec)
            prompt = renderPrompt(self.registry, self.core, row["before"], row["question"])
            item = self.diag.prepare({"row": row, "source": "history", "prompt": prompt})
            self.assertTrue(self.diag.score(item, complete(row["calls"]))["semantic_success"], spec["case_id"])
            groups.add(spec["group"])
        self.assertTrue(groups)

    def test_compact_train_reconstructs_all_evidence(self):
        before = self.session([call("explain_purpose", subject="TR")], [call("explain_purpose", subject="BIDS.basepath")])
        item = self.prepare(before, [call("repeat_exchange", target="e1")])
        gen = {**complete(item["gold_calls"]), "raw_ids": [900000], "diagnostic_worker_sha256": "a" * 64,
               "finish_reason": "complete_batch", "seconds": 1.25}
        full = self.diag.score(item, gen)
        receipt = compactObservation(full)
        self.assertFalse({"before", "after"} & receipt.keys())
        for key in ("before_sha256", "after_sha256", "after_semantic_sha256"):
            self.assertRegex(receipt[key], r"^[a-f0-9]{64}$")
        self.assertEqual(receipt["delivered"]["assistant"], full["after"]["exchanges"][-1]["assistant"])
        self.assertEqual(receipt["delivered"]["outcome"], "committed")
        self.assertEqual(receipt["generation"], gen)
        self.assertEqual(receipt["semantic_result"], full["semantic_result"])
        self.assertLess(len(json.dumps(receipt)), len(json.dumps(full)))
        self.assertEqual(self.diag.expand(item, receipt), full)
        for change in ({"after_sha256": "0" * 64}, {"delivered": {**receipt["delivered"], "assistant": "altered"}},
                       {"generation": {**gen, "raw_text": ""}}):
            with self.assertRaisesRegex(ValueError, "does not reproduce"):
                self.diag.expand(item, {**receipt, **change})
