"""Scorer regression tests for authored, observed and migrated TRAIN histories."""

import json
import unittest

from evaluation import api
from evaluation.functiongemma import encodeCalls
from evaluation.history_train import continuation, materializeScenario, renderPrompt
from evaluation.session import createSession, applyModelOutput
from eval_tools.io import ROOT, records
from eval_tools.new_train_fit import createScorer
from tests.test_eval_diagnose_data import call


class FitScorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = json.loads((ROOT / "assets/registry.json").read_bytes())
        cls.scorer = createScorer(cls.bundle)
        cls.core = api()

    def bind(self, before, row, **options):
        revised = {**before, "registry_id": self.scorer.registry.fingerprint()}
        prompt = renderPrompt(self.scorer.registry, self.core, revised, row["question"])
        req = {"case_id": row["case_id"], "target": encodeCalls(row["calls"]), "prompt": prompt}
        config = {"section": "variable", "branch": "frozen_train", **options}
        return self.scorer.prepare(req, row, before, config), req, config

    def generation(self, meta, calls, complete=True):
        return {"raw_text": encodeCalls(calls), "raw_ids": [1, 50] if complete else [1] * 512, "complete": complete,
                "provider_kind": "functiongemma", "input_sha256": meta["prompt_sha256"], "max_new_tokens": 512,
                "finish_reason": "complete_batch" if complete else "length"}

    def score(self, meta, calls):
        return self.scorer.score(meta["case_id"], self.generation(meta, calls))["observation"]

    def test_equivalent_program_requires_correct_address(self):
        state = createSession(self.scorer.registry, {"sessionId": "equal-text", "dialogueVersion": 2})
        domain = [call("explain_purpose", subject="TR")]
        for q in ("What does TR mean?", "What does TR mean here?"):
            state = applyModelOutput(self.scorer.registry, state, q, encodeCalls(domain))
        row = {"case_id": "addresses", "group": "references", "question": "Quote my first reply.",
               "calls": [call("repeat_exchange", target="e1")]}
        meta, req, config = self.bind(state, row)
        self.assertTrue(self.score(meta, row["calls"])["canonical_call_exact"])
        equivalent = self.score(meta, [call("repeat_answer", target="t1.c1")])
        self.assertTrue(equivalent["success"])
        self.assertFalse(equivalent["canonical_call_exact"])
        wrong = self.score(meta, [call("repeat_answer", target="t2.c1")])
        self.assertFalse(wrong["success"])
        self.assertEqual(wrong["critical_category"], "wrong_reference")
        for field, value in (("target", encodeCalls(domain)), ("prompt", req["prompt"].replace(row["question"], "Other question"))):
            with self.assertRaisesRegex(ValueError, field + " differs"):
                self.scorer.prepare({**req, field: value}, row, state, config)
        with self.assertRaisesRegex(ValueError, "different TRAIN input"):
            self.scorer.score(row["case_id"], {**self.generation(meta, domain), "input_sha256": "other"})
        truncated = self.scorer.score(row["case_id"], self.generation(meta, row["calls"], False))["observation"]
        self.assertFalse(truncated["success"] or truncated["transport_valid"])

    def test_migrated_shared_preserves_actual_failed_history(self):
        fn = "explain_parameter_data_matching"
        args = [b["call"]["arguments"] for b in self.bundle["contracts"]["bindings"] if b["call"]["name"] == fn]
        mismatch = next({"subject": a["subject"], "reference": b["reference"]} for a in args for b in args
                        if {"subject": a["subject"], "reference": b["reference"]} not in args)
        raw = encodeCalls([call(fn, **mismatch)])
        old = createSession(self.scorer.original, {"dialogueVersion": 2, "sessionId": "migrated"})
        before = applyModelOutput(self.scorer.original, old, "Explain the matching.", raw)
        self.assertEqual(before["exchanges"][0]["error_code"], "contract_not_implemented")
        revised = applyModelOutput(self.scorer.registry, {**old, "registry_id": self.scorer.registry.fingerprint()},
                                   "Explain the matching.", raw)
        self.assertEqual(revised["exchanges"][0]["error_code"], "invalid_arguments")
        row = {"case_id": "shared:migrated", "group": "shared_replay", "question": "Quote your first reply.",
               "calls": [call("repeat_exchange", target="e1")]}
        meta, _, _ = self.bind(before, row, migrated=True, section="shared")
        self.assertTrue(self.score(meta, row["calls"])["success"])

    def test_every_authored_group_binds_both_arms(self):
        roots = {r["root_id"]: r for r in records(ROOT / "assets/transfer-train-roots.jsonl")}
        for arm in ("ordinary", "contrast"):
            seen = set()
            for row in records(ROOT / f"assets/transfer-train-{arm}.jsonl"):
                root = roots[row["root_id"]]
                if root.get("observed_prefix_id") or root["group"] in seen:
                    continue
                before = materializeScenario(self.scorer.registry, self.core, {**root, "session_id": root["root_id"],
                                 "case_id": row["case_id"], "continuation": row["continuation"]})["before"]
                policy = continuation(before, row["continuation"])
                meta, _, _ = self.bind(before, {**row, **policy, "group": root["group"]}, branch=policy["branch"])
                self.assertFalse({"target", "calls"} & meta.keys())
                self.assertTrue(self.score(meta, policy["calls"])["success"])
                seen.add(root["group"])
            self.assertEqual(seen, {r["group"] for r in roots.values() if not r.get("observed_prefix_id")})

    def test_missing_answer_vs_ambiguous_reference(self):
        registry = self.scorer.registry
        before = createSession(registry, {"dialogueVersion": 2, "sessionId": "absent"})
        before = applyModelOutput(registry, before, "A prior error.", "not a function batch")
        before = applyModelOutput(registry, before, "A completed reply.", encodeCalls([call("explain_purpose", subject="TR")]))
        row = {"group": "absent", "question": "Repeat the scientific answer from my first request.",
               "calls": [call("clarify_reference", kind="answer")]}
        wrong = [call("repeat_answer", target="t1.c1")]
        meta, _, _ = self.bind(before, {**row, "case_id": "missing"}, branch="requested_answer_missing")
        self.assertEqual(self.score(meta, wrong)["critical_category"], "wrong_reference")
        meta, _, _ = self.bind(before, {**row, "case_id": "ambiguous", "question": "Repeat that answer."},
                               branch="preauthored_ambiguous_reference")
        self.assertEqual(self.score(meta, wrong)["critical_status"], "review_required")
