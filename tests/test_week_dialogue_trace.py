"""Replay full and compact saved evidence on an actual DEV1250 dialogue."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from diagnostics.week_dialogue_trace import render_dialogues
from eval_tools.diagnose_data import compactObservation
from eval_tools.diagnose_dev import full_observation, loadPanel
from eval_tools.evaluate_actual import _materialize
from eval_tools.io import ROOT, line_bytes
from evaluation import api
from evaluation.actual_history import createActualHistoryHarness
from evaluation.functiongemma import encodeCalls
from evaluation.history_train import renderPrompt


REGISTRY = ROOT / "assets/registry.json"
BUNDLE = ROOT / "assets/actual-history-v3-dev.json"
DIALOGUE = "dev-ah3-loop-original_version-02"


class WeekDialogueTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.context = loadPanel(REGISTRY, BUNDLE)
        cls.rows = [r for r in cls.context.plan if r.get("dialogue_id") == DIALOGUE and r["turn_index"] <= 3]

    def observations(self, name):
        core = api()
        harness = createActualHistoryHarness(core)
        before = harness.initial(self.context.registry, self.rows[0])
        records = []
        for row in self.rows:
            resolved = harness.resolve(row, before)
            question = resolved["question"]
            prompt = renderPrompt(self.context.registry, core, before, question)
            batch = resolved["allowed_batches"][0]
            if name == "matched" and row["turn_index"] == 1:
                batch = [{"name": "explain_purpose", "arguments": {"subject": "TR"}}]
            if name != "baseline" and row["turn_index"] == 3:
                call = batch[0]
                batch = [
                    {
                        "name": call["name"],
                        "arguments": {k: v for k, v in call["arguments"].items() if k != "_replace"},
                    }
                ]
            generation = {
                "raw_text": encodeCalls(batch),
                "raw_ids": [50],
                "complete": True,
                "diagnostic_worker_sha256": "a" * 64,
            }
            args = self.context, harness, row, before, resolved, prompt, generation
            observation = full_observation(*args, _materialize) if name == "baseline" else full_observation(*args)
            records.append(observation)
            before = observation["after"]
        return records

    def fixture(self):
        path = Path(self.enterContext(TemporaryDirectory(dir=ROOT / ".runtime/tmp")))
        paths, saved = {}, {}
        for name in ("baseline", "matched", "full"):
            journal = path / f"{name}.jsonl"
            evidence = self.observations(name)
            saved[name] = {r["case_id"]: {k: r[k] for k in ("success", "critical_status")} for r in evidence}
            if name != "baseline":
                evidence = list(map(compactObservation, evidence))
            journal.write_bytes(b"".join(map(line_bytes, evidence)))
            paths[name] = [journal]
        plan = {r["case_id"]: r for r in self.rows}
        return plan, saved, paths

    def test_distinguishes_earlier_state_difference_from_the_target_flag(self):
        plan, saved, paths = self.fixture()
        before = {path: path.read_bytes() for group in paths.values() for path in group}
        text = "\n".join(render_dialogues(REGISTRY, BUNDLE, plan, saved, paths))
        self.assertIn("shared new critical flags: 1", text)
        self.assertIn(f"{DIALOGUE}-t3 | replace", text)
        self.assertIn("matched | first state/answer difference: t1", text)
        self.assertIn("full | first state/answer difference: t3", text)
        self.assertIn("first non-pass: t3 omitted_reference", text)
        self.assertIn("expected=['t1.c1'] selected=-", text)
        self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_rejects_missing_turn_and_already_critical_baseline(self):
        plan, saved, paths = self.fixture()
        matched = paths["matched"][0]
        matched.write_bytes(b"\n".join(matched.read_bytes().splitlines()[1:]) + b"\n")
        with self.assertRaisesRegex(ValueError, "prefixes are incomplete"):
            render_dialogues(REGISTRY, BUNDLE, plan, saved, paths)
        saved["baseline"][f"{DIALOGUE}-t3"]["critical_status"] = "detected"
        self.assertEqual(
            render_dialogues(REGISTRY, BUNDLE, plan, saved, paths),
            ["DEV1250 closed loop | shared new critical flags: 0"],
        )
