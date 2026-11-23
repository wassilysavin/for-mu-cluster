"""Regression checks for prompt-only inference, arm membership and interpretation."""
from array import array
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import check_new_train_fit as fit


class Codec:
    def decode(self, ids, *, skip_special_tokens):
        assert not skip_special_tokens
        return bytes(ids).decode()

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return type("Encoding", (), {"ids": list(text.encode())})


def make_fixture():
    arms = {}
    source = {}
    for name in ("ordinary", "contrast"):
        cases = []
        for cid, section, answer in (("replay", "shared", "r"), (name + "-same", "variable", "a"),
                                     (name + "-distinct", "variable", name)):
            prompt = "seen?" if section == "variable" else "again?"
            group = "ambiguous_reference" if section == "variable" else "shared_replay"
            meta = {"case_id": cid, "section": section, "group": group, "position": len(cases),
                    "row_sha256": fit.digest([cid, prompt, answer])}
            source[cid] = dict(meta, input_ids=list((prompt + answer).encode()), prompt_tokens=len(prompt))
            cases.append(meta)
        arms[name + "-1729"] = {"cases": cases, "checkpoint": name,
                                "fresh_dev": {"supported": {"passed": 5, "planned": 10}}}
    inputs = {"arms": arms, "base": "base", "tokenizer": "tokenizer"}

    def rows(inputs, arm):
        return (source[c["case_id"]] for c in inputs["arms"][arm]["cases"])
    return inputs, rows, source


class FixtureScorer:
    def __init__(self, source):
        self.source = source
        self.prepared = {}
        self.seen = []

    def request(self, req, expected):
        cid = req["case_id"]
        if expected == "prepared":
            self.seen.append(cid)
            row = self.source[cid]
            self.prepared[cid] = {"case_id": cid, "section": row["section"], "group": row["group"],
                                  "cohort": "supported", "prompt_sha256": fit.text_sha(req["prompt"]),
                                  "target_sha256": fit.text_sha(req["target"])}
            return self.prepared[cid]
        assert expected == "scored" and req["generation"]["input_sha256"] == self.prepared[cid]["prompt_sha256"]
        ok = req["generation"]["raw_text"] == "ok"
        return {"observation": {**self.prepared[cid], "success": ok, "canonical_call_exact": False,
                                "native_literal_exact": False, "transport_valid": True,
                                "critical_status": "none_observed" if ok else "review_required",
                                "critical_category": None, "failure_category": None if ok else "semantic_unresolved"}}


class TrainFitTests(unittest.TestCase):
    def test_complete_arms_with_shared_and_unique_new_inputs(self):
        inputs, rows, source = make_fixture()
        rpc = FixtureScorer(source)
        launched = []

        def generate(out, name, binding, checkpoint, base, tok, prompts, lease, budget):
            launched.append(name)
            self.assertEqual(checkpoint, name.split("-")[0])
            self.assertEqual(set(prompts), {r["case_id"] for r in rows(inputs, name)})
            for cid, ids in prompts.items():
                self.assertIsInstance(ids, array)
                self.assertEqual(list(ids), source[cid]["input_ids"][:source[cid]["prompt_tokens"]])
            return {cid: {"input_tokens": len(ids), "raw_text": "miss" if cid == "contrast-distinct" else "ok",
                          "finish_reason": "complete_batch"} for cid, ids in prompts.items()}

        with patch.object(fit, "iter_cases", rows), patch.object(fit, "generate_arm", generate):
            prepared = fit.prepare(inputs, Codec(), rpc)
            result = fit.evaluate(inputs, prepared, rpc, "digest", None, 99, None)
        self.assertEqual(len(rpc.seen), 5)
        self.assertEqual(launched, list(inputs["arms"]))
        c = result["arms"]["contrast-1729"]
        self.assertEqual([c["subsets"][k]["total"] for k in
                          ("variable", "shared_replay", "variable_unique", "variable_common")], [2, 1, 1, 1])
        self.assertEqual(c["subsets"]["all"]["passed"], 2)
        self.assertEqual(c["subsets"]["all"]["canonical_calls"], 0)
        self.assertEqual(c["subsets"]["variable_unique"]["unresolved"], 1)
        self.assertEqual(c["seen_unique_fit"], "resolve_reviews_first")
        self.assertEqual(result["arms"]["ordinary-1729"]["seen_variable_fit"], "high_seen_fit")

    def test_score_refuses_missing_or_mismatched_generation(self):
        inputs, rows, source = make_fixture()
        rpc = FixtureScorer(source)
        with patch.object(fit, "iter_cases", rows):
            prepared = fit.prepare(inputs, Codec(), rpc)
        name = "ordinary-1729"
        arm = inputs["arms"][name]
        with self.assertRaisesRegex(ValueError, "Incomplete TRAIN"):
            fit.score_arm(name, arm, set(), prepared, {}, rpc)
        with self.assertRaisesRegex(ValueError, "input length"):
            fit.score_arm(name, arm, set(), prepared, {c["case_id"]: {"input_tokens": 0} for c in arm["cases"]}, rpc)

    def test_review_bounds_and_threshold(self):
        for passed, review, expected in [(90, 0, "high_seen_fit"), (88, 2, "resolve_reviews_first"),
                                         (87, 2, "inspect_misses_then_test_longer_training")]:
            self.assertEqual(fit.assess({"passed": passed, "total": 100, "unresolved": review}), expected)


class OutputRevisionTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        self.output = root / "runs/new-train-fit-v1"
        self.output.mkdir(parents=True)
        self.receipt = self.output / "experiment.json"
        self.manifest = {
            "protocol": "new-train-fit-v1", "inputs_sha256": "frozen", "arms": {"ordinary-1729": {}},
            "code": [{"path": "eval_tools/new_train_fit.py"}],
        }
        self.legacy = dict(self.manifest, code=[{"path": "new_train_fit.mjs"}])
        self.enterContext(patch.object(fit, "ROOT", root))
        self.enterContext(patch.object(fit, "OUTPUT", self.output))

    def save(self, receipt):
        self.receipt.write_text(json.dumps(receipt), encoding="utf-8")

    def test_new_and_matching_runs_keep_their_path(self):
        self.assertEqual(fit.choose_output(self.manifest), self.output)
        self.save(self.manifest)
        self.assertEqual(fit.choose_output(self.manifest), self.output)

    def test_legacy_receipt_and_empty_arm_survive_python_restart(self):
        self.save(self.legacy)
        original = self.receipt.read_bytes()
        (self.output / "ordinary-1729").mkdir()
        revision = self.output / fit.PYTHON_REVISION
        self.assertEqual(fit.choose_output(self.manifest), revision)
        revision.mkdir(parents=True)
        (revision / "experiment.json").write_text(json.dumps(self.manifest))
        self.assertEqual(fit.choose_output(self.manifest), revision)
        self.assertEqual(self.receipt.read_bytes(), original)

    def test_changing_frozen_inputs_or_leaving_generations_prevents_revision(self):
        self.save(dict(self.legacy, inputs_sha256="different"))
        with self.assertRaisesRegex(ValueError, "Frozen TRAIN-fit record differs"):
            fit.choose_output(self.manifest)
        self.save(self.legacy)
        arm = self.output / "ordinary-1729"
        arm.mkdir()
        recorded = arm / "gpu-0.jsonl"
        recorded.write_text("saved generation\n")
        with self.assertRaisesRegex(ValueError, "contains results"):
            fit.choose_output(self.manifest)
        self.assertEqual(recorded.read_text(), "saved generation\n")
        self.assertFalse((self.output / fit.PYTHON_REVISION).exists())


if __name__ == "__main__":
    unittest.main()
