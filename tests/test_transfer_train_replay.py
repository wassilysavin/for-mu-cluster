"""Use the real frozen tokenizer and replay inputs to check the input-only migration."""

import json
import subprocess
from unittest import TestCase, main, mock

from transfer_contract import MODEL_TURN, USER_TURN, upgrade_replay
from transfer_train import ASSETS, ROOT, rows


class PackedReplayTests(TestCase):
    @classmethod
    def setUpClass(cls):
        from tokenizers import Tokenizer
        cls.tokenizer = Tokenizer.from_file(str(ASSETS / "tokenizer.json"))
        cls.fresh = next(rows(ASSETS / "train-full-v2.jsonl"))

    def test_user_turn_and_supervision_are_bit_exact(self):
        t = self.tokenizer
        row = self.fresh
        new = upgrade_replay(row, t)
        a,b = [t.decode(v["input_ids"][:v["prompt_tokens"]], skip_special_tokens=False) for v in (row,new)]
        self.assertTrue(a.endswith(MODEL_TURN))
        self.assertEqual(a.partition(USER_TURN)[2], b.partition(USER_TURN)[2])
        self.assertEqual(row["input_ids"][row["prompt_tokens"]:], new["input_ids"][new["prompt_tokens"]:])
        self.assertNotEqual(a,b)
        self.assertEqual(a.split("\n\n",1)[0], b.split("\n\n",1)[0])

    def test_packed_migration_equals_live_native_prompt(self):
        migrated = upgrade_replay(self.fresh, self.tokenizer)
        text = self.tokenizer.decode(migrated["input_ids"][:migrated["prompt_tokens"]], skip_special_tokens=False)
        user = json.loads(text.split(USER_TURN)[1].removesuffix(MODEL_TURN))
        from data_tools.transfer_contract import transferRegistry
        from evaluation import api
        from evaluation.history_train import renderPrompt
        runtime = api()
        registry = transferRegistry(json.loads((ASSETS / "registry.json").read_bytes()))
        before = runtime.createSession(registry, dict(dialogueVersion=2, sessionId=user["history"]["session_id"]))
        self.assertEqual(text, renderPrompt(registry, runtime, before, user["question"]))

    def test_unchanged_contract_and_misaligned_target_are_rejected(self):
        with mock.patch("transfer_contract.upgrade_packed", side_effect=lambda text: text):
            with self.assertRaisesRegex(ValueError,"missing the required contract"):
                upgrade_replay(self.fresh,self.tokenizer)
        changed = {**self.fresh,"prompt_tokens":self.fresh["prompt_tokens"]-1}
        with self.assertRaisesRegex(ValueError,"boundaries"):
            upgrade_replay(changed,self.tokenizer)

if __name__ == "__main__":
    main()
