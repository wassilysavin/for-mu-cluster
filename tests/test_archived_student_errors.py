"""Exercise saved FunctionGemma call comparisons and complete-state replay."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from diagnostics.student_errors import comparePrograms, inspectArm
from evaluation import api
from evaluation.actual_history import createActualHistoryHarness
from evaluation.functiongemma import encodeCalls
from evaluation.history_train import renderPrompt
from evaluation.session import applyModelOutput, recordIncompleteModelOutput
from eval_tools.diagnose_dev import loadPanel
from eval_tools.io import ROOT, digest, line_bytes


def call(name, **args):
    return {"name": name, "arguments": args}


class StudentReplayTests(unittest.TestCase):
    def test_equivalent_alternatives_duplicate_calls_and_order(self):
        a, b = call("get_units", subject="width"), call("explain_purpose", subject="width")
        self.assertEqual(comparePrograms([a], [[a, b], [a]])["comparison"], "exact_allowed")
        self.assertIn("fewer_calls", comparePrograms([a], [[a, a]])["signals"])
        self.assertIn("extra_calls", comparePrograms([a, a], [[a]])["signals"])
        self.assertEqual(comparePrograms([b, a], [[a, b]])["signals"], ["call_order"])
        first, second = call("repeat_answer", target="t1.c1"), call("repeat_answer", target="t1.c2")
        self.assertEqual(comparePrograms([second, first], [[first, second]])["signals"], ["call_order"])

    def test_only_common_signals_are_definitive(self):
        a, b, c = call("get_units", subject="width"), call("get_default_value", subject="width"), call("get_units", subject="height")
        result = comparePrograms([a], [[c], [b]])
        self.assertEqual(result["comparison"], "alternative_dependent")
        self.assertFalse({"argument_value", "operation_substitution"} & set(result["signals"]))

    def test_boolean_number_null_and_missing_remain_distinct(self):
        result = comparePrograms([call("op", condition=0, invented=None)], [[call("op", condition=False, retained=None)]])
        self.assertEqual(set(result["signals"]), {"missing_argument", "extra_argument", "argument_value"})

    def test_full_dialogue_replay_rejects_tampering_and_preserves_evidence(self):
        source = loadPanel(ROOT / "assets/registry.json", ROOT / "assets/actual-history-v3-dev.json")
        plan = [r for r in source.plan if r.get("dialogue_id") == "dev-ah3-loop-quote_displayed-01"]
        self.assertEqual(len(plan), 5)
        ctx = SimpleNamespace(**{**vars(source), "plan": plan,
                                "data": {**source.data, "panels": {"fixed_prefix": [], "closed_loop": plan, "challenge": []}}})
        core = api()
        harness = createActualHistoryHarness(core)
        before = harness.initial(ctx.registry, plan[0])
        full = []
        for index, row in enumerate(plan):
            resolved = harness.resolve(row, before)
            question = resolved["question"]
            prompt = renderPrompt(ctx.registry, core, before, question)
            raw = "synthetic incomplete native call" if index == 1 else encodeCalls(resolved["allowed_batches"][0])
            ids = [1] * 512 if index == 1 else [50]
            after = (recordIncompleteModelOutput(ctx.registry, before, question, raw, ids, "token_limit") if index == 1
                     else applyModelOutput(ctx.registry, before, question, raw))
            grade = harness.grade(ctx.registry, row, before, after, resolved)
            full.append({**grade, "bundle_sha256": ctx.bundleHash, "prompt_sha256": digest(prompt),
                         "before_visible_history_sha256": resolved["before_visible_history_sha256"], "question": question,
                         "policy_branch": resolved["branch"], "expected_model_call_batches": resolved["allowed_batches"],
                         "before": before, "after": after, "generation": {"raw_text": raw, "raw_ids": ids}})
            before = after
        self.assertEqual(full[1]["failure_category"], "execution_error")
        self.assertEqual(full[2]["before"]["exchanges"][1]["error_code"], "incomplete_model_output")
        self.assertEqual(full[2]["expected_reference_targets"], ["e2"])
        private = ROOT / ".runtime/tmp"
        private.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=private) as temp:
            path = Path(temp) / "observations.jsonl"
            def write(rows):
                data = b"".join(map(line_bytes, rows))
                path.write_bytes(data)
                return data
            initial = write(full)
            report = inspectArm(ctx, path)
            closed = report["score"]["panels"]["closed_loop"]
            self.assertEqual((closed["supported"]["passed"], closed["whole_dialogues"]["passed"]), (4, 0))
            self.assertEqual(len(report["rows"]), len(plan))
            self.assertEqual(path.read_bytes(), initial)
            for invalid in ([full[0], {**full[1], "success": True}, *full[2:]],
                            [*full[:2], {**full[2], "expected_model_call_batches": [[]]}, *full[3:]],
                            [*full[:2], {**full[2], "before": full[1]["before"]}, *full[3:]],
                            [full[0], {**full[1], "generation": {**full[1]["generation"], "raw_ids": [50]}}, *full[2:]],
                            full[:-1], [*full[:-1], full[3]]):
                evidence = write(invalid)
                with self.assertRaises(ValueError):
                    inspectArm(ctx, path)
                self.assertEqual(path.read_bytes(), evidence)
