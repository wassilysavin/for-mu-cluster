"""Offline checks for the paired TRAIN intervention and resumable materialization."""

from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main, mock

import transfer_train as train


class TrainPairTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.common_source = {r["case_id"]: (r["question"],r["calls"]) for r in train.rows(train.ASSETS / "train-full-v2.jsonl")}
        cls.roots = {r["root_id"]: r for r in train.declarations("roots")}
        cls.arms = {arm: train.declarations(arm) for arm in train.ARMS}

    def test_question_and_policy_marginals_match_but_joint_distribution_differs(self):
        self.assertEqual(train.check()["variable_cases_per_arm"], 320)
        for arm in train.ARMS:
            meanings = defaultdict(set)
            for row in self.arms[arm]:
                meanings[row["root_id"]].add(row["meaning"])
            self.assertEqual({len(v) for v in meanings.values()}, {1 if arm == "ordinary" else 2})

    def test_anchors_and_corrected_subjects_match_original_scientific_labels(self):
        for r in self.roots.values():
            for step in r["prefix"]:
                self.assertTrue(all(cid in self.common_source for cid in step["ancestor_case_ids"]))
                if step.get("kind") == "gold" and "calls" in step:
                    self.assertEqual(self.common_source[step["ancestor_case_ids"][0]], (step["question"],step["calls"]))
            if r["group"] == "ambiguous_address":
                topics = [step["calls"][0]["arguments"].get("subject") for step in r["prefix"]]
                self.assertEqual(topics[0], topics[2])
                self.assertNotEqual(topics[0],topics[1])
                self.assertNotEqual(r["prefix"][0]["calls"],r["prefix"][2]["calls"])
        for row in self.arms["contrast"]:
            policy = row["continuation"]
            sources = [self.common_source[case][1][0] for case in policy.get("source_case_ids", [])]
            if self.roots[row["root_id"]]["group"] == "ambiguous_address":
                utterance = policy["question"]
                if row["meaning"] == 0:
                    self.assertNotRegex(utterance, r"\b(earlier|later|first|last)\b")
                    self.assertRegex(utterance, r"does not say which|have not specified one")
                    self.assertEqual(policy["kind"], "clarify_reference")
                else:
                    self.assertRegex(utterance, r"\b(earlier|later|first|last)\b")
                    self.assertEqual(policy["kind"], "quote_answer")
            if policy["kind"] == "fresh":
                if row["root_id"].startswith("st-tr-condition_omission") and row["meaning"] == 0:
                    original = sources[0]
                    self.assertEqual(policy["calls"],[{"name":original["name"],"arguments":{
                        "subject":original["arguments"]["subject"],"_clarify":True,"_missing":["value"]}}])
                else:
                    self.assertEqual(sources,policy["calls"])
            if policy["kind"] == "revise_answer":
                self.assertEqual(sources, [policy["expected_original"],policy["replacement"]])
                expected = {**sources[0]["arguments"], **policy["changes"]}
                self.assertEqual(expected, sources[1]["arguments"])
            if policy["kind"] == "answer_again":
                step = self.roots[row["root_id"]]["prefix"][policy["exchange_index"]-1]
                source = self.common_source[step["ancestor_case_ids"][0]]
                self.assertEqual(source, (step["question"],policy["calls"]))

    def test_explicit_hrf_endpoint_does_not_supervise_generic_change(self):
        coverage = train.read(train.ASSETS / "history-common-coverage.json")["functions"]
        shared = train.common_ids(self.common_source, coverage)
        self.assertFalse(train.EXCLUDED_SHARED.intersection(shared))
        for case_id in shared:
            question, calls = self.common_source[case_id]
            self.assertFalse(train.generic_interval_conflict({"question": question, "calls": calls}), case_id)
        for records in self.arms.values():
            for row in records:
                policy = row["continuation"]
                self.assertFalse(train.generic_interval_conflict(policy), row["case_id"])
                if policy["kind"] == "revise_answer":
                    self.assertFalse(train.generic_interval_conflict({"question": policy["question"], "calls": [
                        policy["expected_original"], policy["replacement"]]}))
        for root in self.roots.values():
            self.assertFalse(any(train.generic_interval_conflict(step) for step in root["prefix"]))

    def test_observed_roots_are_bound_to_real_source_prefixes(self):
        specs = {r["prefix_id"]:r for r in train.rows(train.ASSETS / "history-train-specs.jsonl")}
        observed = [r for r in self.roots.values() if "observed_prefix_id" in r]
        self.assertEqual(len({r["observed_prefix_id"] for r in observed}),16)
        for r in observed:
            template = specs[r["observed_prefix_id"]]
            self.assertEqual(r["source_session_id"], template["session_id"])
            self.assertEqual([s["question"] for s in r["prefix"]], [s["question"] for s in template["prefix"]])

    def test_each_batch_has_same_replay_and_pair_slots(self):
        coverage = train.read(train.ASSETS / "history-common-coverage.json")["functions"]
        ids = train.common_ids(self.common_source, coverage)
        plan = train.arrangements(ids, self.arms)
        for seed in train.SEEDS:
            arms = {arm:{position:case_id for case_id, places in plan.items()
                         if (position:=places.get(f"{arm}-{seed}")) is not None} for arm in train.ARMS}
            self.assertEqual(set(arms["ordinary"]), set(range(640)))
            for i in range(640):
                a,b=(arms[arm][i] for arm in train.ARMS)
                if i % 2:
                    self.assertEqual(a.replace("ordinary","contrast"),b)
                else:
                    self.assertEqual(a,b)
                    self.assertTrue(a.startswith("shared:"))
        self.assertEqual(len(coverage),97)

    def test_no_unbound_overwrite_and_atomic_resume(self):
        with TemporaryDirectory(dir=train.ROOT) as where, mock.patch.object(train, "ROOT", Path(where)):
            root = Path(where)
            output = root / "runs/train"
            for name in ("transfer_train.py","data_tools/transfer_train.py","data_tools/transfer_contract.py",
                         "transfer_contract.py", "private_runtime.py", "source_provenance.py"):
                (root/name).parent.mkdir(parents=True, exist_ok=True)
                (root/name).write_text(name)
            def write_tokens(path, receipts, budget):
                path.write_text("complete\n")
                return {"example":True}
            with mock.patch.object(train,"receipt_source",return_value=(None,"verified")), mock.patch.object(train,"check",return_value={}), mock.patch.object(train,"build_file",side_effect=write_tokens) as build:
                result=train.prepare(output, available_bytes=10_000_000)
                self.assertEqual(train.prepare(output)["sha256"],result["sha256"])
                (output/"train-receipt.json").rename(output/".receipt.pending")
                (output/"train.jsonl").rename(output/".train.partial")
                self.assertEqual(train.prepare(output)["sha256"],result["sha256"])
                self.assertEqual(build.call_count,1)
                (output/"train-receipt.json").unlink()
                with self.assertRaisesRegex(ValueError,"Unbound"):
                    train.prepare(output)

if __name__ == "__main__":
    main()
