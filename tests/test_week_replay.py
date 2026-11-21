"""Check TRAIN reference binding, conservative duplicate handling and target classification."""

from private_runtime import python_tool
import json
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from data_tools.week_replay import SOURCE, curate, group_for, iter_replay


def record(cid, key, target, offset):
    return {"source_case_id": cid, "visible_sha256": key, "target_sha256": target, "offset": offset}


class ReplayTests(TestCase):
    def test_one_valid_row_per_input_and_conflicting_targets_are_quarantined(self):
        a = record("a", "shared", "same", 7)
        b = record("b", "shared", "same", 2)
        x = record("x", "conflict", "one", 3)
        y = record("y", "conflict", "two", 4)
        z = record("z", "other", "unchanged", 1)
        result, removed = curate([(b, None), (y, None), (a, None), (z, "legacy_hrf_interval_target"), (x, None)])
        self.assertEqual(result, [a])
        self.assertEqual(removed, {"b": "duplicate_of:a", "x": "conflicting_targets",
                                   "y": "conflicting_targets", "z": "legacy_hrf_interval_target"})

    def test_reference_scope_is_distinct_from_argument_clarification(self):
        h = {"exchanges": []}
        self.assertEqual(group_for([{"name": "clarify_reference", "arguments": {"kind": "answer"}}], h), "clarify")
        self.assertEqual(group_for([{"name": "get_units", "arguments": {"_clarify": "subject"}}], h), "clarify_argument")
        self.assertEqual(group_for([{"name": "repeat_exchange", "arguments": {"target": "e1"}}], h), "quote_whole")
        group = group_for([{"name": "decline_request", "arguments": {}}, {"name": "get_units", "arguments": {}}], h)
        self.assertEqual(group, "mixed_scope")

    def test_random_access_keeps_original_tokens_and_detects_stale_bindings(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp).resolve()
            assets = root / "assets"
            assets.mkdir()
            row = {"case_id": "r", "input_ids": [1, 2, 3, 50], "prompt_tokens": 2}
            raw = (json.dumps(row) + "\n").encode()
            (assets / SOURCE).write_bytes(b'{}\n' + raw)
            ref = {"case_id": "week-replay:r", "source_case_id": "r", "source": SOURCE, "offset": 3,
                   "bytes": len(raw), "row_sha256": sha256(raw).hexdigest()}
            self.assertEqual(list(iter_replay([ref], root)), [(ref, row)])
            for invalid in (dict(ref, source_case_id="other"), dict(ref, offset=0), dict(ref, row_sha256="0" * 64)):
                with self.assertRaises(ValueError):
                    list(iter_replay([invalid], root))


class NativeReplayTests(TestCase):
    def test_native_validator_accepts_frozen_gold_and_rejects_wrong_state(self):
        import subprocess
        from tokenizers import Tokenizer
        from transfer_contract import MODEL_TURN, USER_TURN, upgrade_replay
        from transfer_train import ROOT, rows

        tokenizer = Tokenizer.from_file(str(ROOT / "assets/tokenizer.json"))
        row = next(r for r in rows(ROOT / "assets" / SOURCE) if r["case_id"] == "train-dialogue-0001-turn-2")
        upgraded = upgrade_replay(row, tokenizer)
        ids, cut = upgraded["input_ids"], upgraded["prompt_tokens"]
        prompt = tokenizer.decode(ids[:cut], skip_special_tokens=False)
        req = {"case_id": "week-replay:" + row["case_id"], "source_case_id": row["case_id"],
               "question": row["question"], "calls": row["calls"], "model_input": prompt,
               "target": tokenizer.decode(ids[cut:], skip_special_tokens=False)}
        cmd = [*python_tool("data_tools/week_replay_check.py")]
        good = subprocess.run(cmd, input=json.dumps(req) + "\n", text=True, capture_output=True,
                              cwd=ROOT, check=True)
        got, summary = map(json.loads, good.stdout.splitlines())
        self.assertEqual(got["kind"], "verified")
        self.assertEqual(summary, {"kind": "complete", "cases": 1})
        developer, user = prompt.split(USER_TURN)
        data = json.loads(user.removesuffix(MODEL_TURN))
        data["history"]["exchanges"].clear()
        wrong_prompt = developer + USER_TURN + json.dumps(data) + MODEL_TURN
        bad = subprocess.run(cmd, input=json.dumps(dict(req, model_input=wrong_prompt)) + "\n",
                             text=True, capture_output=True, cwd=ROOT)
        self.assertNotEqual(bad.returncode, 0)
        self.assertNotIn('"kind":"verified"', bad.stdout)
