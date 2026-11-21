"""Verify fixed coverage, receipt replay, shard boundaries and relay persistence."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from evaluation import api
from evaluation.actual_history import createActualHistoryHarness
from evaluation.functiongemma import encodeCalls
from evaluation.history_train import renderPrompt
from eval_tools.diagnose_data import compactObservation
from eval_tools.diagnose_dev import PANELS, expandRecords, full_observation, loadPanel, shardOf, summarize
from eval_tools.io import ROOT, line_bytes


class DiagnosticDevTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        registry, panel = ROOT / "assets/registry.json", ROOT / "assets/actual-history-v3-dev.json"
        cls.cmd = [sys.executable, "-m", "eval_tools.diagnose_dev", str(registry), str(panel)]
        cls.context = loadPanel(registry, panel)
        cls.core = api()
        cls.harness = createActualHistoryHarness(cls.core)

    def temporary(self):
        directory = ROOT / ".runtime/tmp"
        directory.mkdir(parents=True, exist_ok=True)
        return Path(self.enterContext(tempfile.TemporaryDirectory(dir=directory)))

    def test_complete_disjoint_dialogue_shards(self):
        rows = self.context.plan
        parts = [[r for r in rows if shardOf(r) == i] for i in range(4)]
        ids = [r["case_id"] for part in parts for r in part]
        self.assertEqual(len(ids), 1250)
        self.assertEqual(len(set(ids)), len(ids))
        self.assertEqual(set(ids), {r["case_id"] for r in rows})
        self.assertTrue(all(parts))
        dialogs = {}
        for row in rows:
            if row.get("dialogue_id"):
                dialogs.setdefault((row["panel"], row["dialogue_id"]), []).append(row)
        self.assertEqual(len(dialogs), 136)
        for (panel, did), group in dialogs.items():
            self.assertEqual([r["turn_index"] for r in group], list(range(1, 9 if panel == "challenge" else 6)))
            self.assertEqual([r for r in parts[shardOf(group[0])] if r.get("dialogue_id") == did], group)

    def test_summary_keeps_denominators_and_reviews(self):
        context = self.context
        results = [{**r, "success": i % 17 != 0, "reference_success": i % 17 != 0 and r["reference_eligible"],
                    "critical_status": "none_observed" if i % 17 else "review_required" if i % 34 else "detected",
                    "critical_category": "wrong_reference" if i % 34 == 0 else None} for i, r in enumerate(context.plan)]
        out = summarize(context.data, results)
        for panel in PANELS:
            self.assertEqual(out["panels"][panel], self.harness.summarize(context.data["panels"][panel],
                                                                          [r for r in results if r["panel"] == panel]))
        self.assertEqual([out["summary"][k]["planned"] for k in ("supported", "outside_scope", "whole_dialogues")], [834, 64, 128])
        self.assertEqual(out["panels"]["fixed_prefix"]["supported"]["planned"], 288)
        self.assertEqual(out["panels"]["challenge"]["whole_dialogues"]["planned"], 8)
        self.assertEqual(out["observed_records"], 1250)
        self.assertFalse(out["release_ready"])
        reduced = summarize(context.data, results[1:])["panels"]["fixed_prefix"]
        self.assertEqual(reduced["missing"], 1)
        self.assertEqual(reduced["supported"]["planned"], 288)

    def dialogue(self):
        context, harness = self.context, self.harness
        rows = [r for r in context.plan if r.get("dialogue_id") == "dev-ah3-loop-quote_displayed-01"]
        before = harness.initial(context.registry, rows[0])
        full = []
        for i, row in enumerate(rows):
            resolved = harness.resolve(row, before)
            prompt = renderPrompt(context.registry, self.core, before, resolved["question"])
            gen = {"provider_kind": "reference", "complete": i != 1, "diagnostic_worker_sha256": "b" * 64,
                   "raw_text": encodeCalls(resolved["allowed_batches"][0]) if i != 1 else "unfinished",
                   "raw_ids": [420001] if i != 1 else [420000], "finish_reason": "complete_batch" if i != 1 else "length"}
            full.append(full_observation(context, harness, row, before, resolved, prompt, gen))
            if i == 2:
                self.assertIn("reliable answer to that request", prompt)
                self.assertEqual(resolved["reference_targets"], ["e2"])
            before = full[-1]["after"]
        return rows, full

    def test_actual_incomplete_reply_is_quoted(self):
        rows, full = self.dialogue()
        self.assertEqual([r["success"] for r in full], [True, False, True, True, True])
        self.assertEqual(full[1]["after"]["semantic"], full[1]["before"]["semantic"])
        self.assertEqual(full[2]["after"]["exchanges"][-1]["assistant"], full[1]["after"]["exchanges"][-1]["assistant"])
        last = full[1]["after"]["exchanges"][-1]
        self.assertEqual((last["outcome"], last["raw_output"], last["generation"]["token_ids"]), ("error", "unfinished", [420000]))
        scores = self.harness.summarize(rows, full)
        self.assertEqual((scores["supported"]["passed"], scores["whole_dialogues"]["passed"], scores["missing"]), (4, 0, 0))

    def test_cli_plan_and_reordered_or_incomplete_summary(self):
        context = self.context
        output = subprocess.check_output([*self.cmd, "--plan"], cwd=ROOT)
        plan = json.loads(output)["rows"]
        self.assertEqual([r["case_id"] for r in plan], [r["case_id"] for r in context.plan])
        self.assertEqual([r["shard"] for r in plan], list(map(shardOf, context.plan)))
        path = self.temporary() / "rows.jsonl"
        results = [{k: r[k] for k in ("case_id", "panel")} | {"bundle_sha256": context.bundleHash, "success": True,
                   "reference_success": bool(r["reference_eligible"]), "critical_status": "none_observed"} for r in context.plan]
        argv = [*self.cmd, "--summarize", path]
        path.write_bytes(b"".join(map(line_bytes, results)))
        scores = json.loads(subprocess.check_output(argv, cwd=ROOT))["score"]["summary"]
        self.assertEqual((scores["supported"]["passed"], scores["whole_dialogues"]["passed"]), (834, 128))
        for bad in ([results[1], results[0], *results[2:]], results[:-1]):
            path.write_bytes(b"".join(map(line_bytes, bad)))
            self.assertNotEqual(subprocess.run(argv, cwd=ROOT, capture_output=True).returncode, 0)
            self.assertTrue(path.read_bytes().endswith(b"\n"))

    def test_compact_replay_and_read_only_audit(self):
        _, full = self.dialogue()
        receipts = list(map(compactObservation, full))
        self.assertEqual(list(expandRecords(self.context, receipts, {"binding": "b" * 64})), full)
        self.assertEqual(summarize(self.context.data, receipts), summarize(self.context.data, full))
        self.assertLess(len(json.dumps(receipts)), len(json.dumps(full)))
        self.assertEqual(receipts[1]["delivered"]["outcome"], "error")
        for bad in (receipts[1:], [receipts[1], receipts[0], *receipts[2:]],
                    [receipts[0], {**receipts[1], "after_semantic_sha256": "0"*64}, *receipts[2:]],
                    [*receipts[:2], {**receipts[2], "generation": {**receipts[2]["generation"], "raw_text": ""}}, *receipts[3:]]):
            with self.assertRaises(ValueError):
                list(expandRecords(self.context, bad))
        with self.assertRaisesRegex(ValueError, "another provider"):
            list(expandRecords(self.context, receipts, {"binding": "c"*64}))
        path = self.temporary() / "compact.jsonl"
        evidence = b"".join(map(line_bytes, receipts))
        path.write_bytes(evidence)
        audit = json.loads(subprocess.check_output([*self.cmd, "--audit", path], cwd=ROOT))
        self.assertEqual(audit["observations"], 5)
        self.assertFalse(audit["model_inference_performed"])
        out = subprocess.check_output([*self.cmd, "--expand", path, receipts[2]["case_id"]], cwd=ROOT)
        item, conclusion = map(json.loads, out.splitlines())
        self.assertEqual(item["observation"], full[2])
        self.assertEqual(conclusion["kind"], "audit")
        self.assertEqual(path.read_bytes(), evidence)

    def test_relay_persist_and_resume_verified_prefix(self):
        path = self.temporary() / "relay.jsonl"
        binding = "d" * 64
        selected = [r for r in self.context.plan if shardOf(r) == 0]
        cmd = [*self.cmd, path, "0", "4", "536870912", binding, "--relay-persist"]

        def start():
            p = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            def close():
                if p.poll() is None:
                    p.terminate()
                p.communicate(timeout=15)
            self.addCleanup(close)
            return p

        def send(p, message):
            p.stdin.write(line_bytes(message))
            p.stdin.flush()

        p = start()
        req = json.loads(p.stdout.readline())
        self.assertEqual((req["kind"], req["case_id"]), ("generate", selected[0]["case_id"]))
        self.assertFalse(path.exists())
        before = self.harness.initial(self.context.registry, selected[0])
        resolved = self.harness.resolve(selected[0], before)
        send(p, {"provider_kind": "reference", "complete": True, "raw_text": encodeCalls(resolved["allowed_batches"][0]),
                 "raw_ids": [420001], "diagnostic_worker_sha256": binding})
        persist = json.loads(p.stdout.readline())
        self.assertFalse(path.exists())
        self.assertEqual(persist["kind"], "persist")
        receipt = persist["observation"]
        self.assertEqual(receipt["case_id"], selected[0]["case_id"])
        self.assertNotIn("after", receipt)
        self.assertRegex(receipt["delivered_sha256"], "^[a-f0-9]{64}$")
        evidence = line_bytes(receipt)
        path.write_bytes(evidence)
        send(p, {"kind": "persisted", "case_id": req["case_id"]})
        self.assertEqual(json.loads(p.stdout.readline())["case_id"], selected[1]["case_id"])
        p.terminate()
        p.wait(timeout=15)
        resumed = start()
        req = json.loads(resumed.stdout.readline())
        self.assertEqual((req["kind"], req["case_id"]), ("generate", selected[1]["case_id"]))
        self.assertEqual(path.read_bytes(), evidence)
