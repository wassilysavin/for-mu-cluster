"""Check loss boundaries, frozen corpus validation and reproducible epoch reading."""
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import week_train as week


class Codec:
    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return type("Ids", (), {"ids": list(text.encode())})


def row(cid):
    data = json.dumps({"format": "test", "history": {"session_id": cid}, "question": "Repeat it."})
    return {"case_id": cid, "family_id": cid, "group": "quote", "question": "Repeat it.", "calls": ["mock"],
            "history_origin": "fixture", "ancestor_case_ids": ["train-fixture"],
            "model_input": "dev},{eloper},{dev" + week.USER_TURN + data + week.MODEL_TURN, "target": "2"}


class WeeklyCorpusTests(unittest.TestCase):
    def sample(self, path):
        h = sha256()
        with path.open("wb") as f:
            week.emit(f, h, {"type": "manifest", "format": week.FORMAT, "split": "train",
                             "tokenizer_sha256": week.FROZEN["tokenizer.json"]})
            keys = set()
            cells, original = {}, {}
            for i, stream in enumerate(week.STREAMS):
                item = row(str(i))
                seq = week.tokenized(item, Codec())
                cell, pre, key, _ = week.pack(stream, item, seq)
                keys.add(key)
                cells[item["case_id"]] = cell
                original[item["case_id"]] = seq
                if not i:
                    week.write_prefix(f, h, cell["prefix"], pre, {})
                week.emit(f, h, cell)
            f.write((json.dumps({"type": "summary", "cases": 3, "prefixes": 1, "streams": dict.fromkeys(week.STREAMS, 1),
                                 "body_sha256": h.hexdigest()}) + "\n").encode())
        self.assertEqual(len(keys), 1)  # Only opaque session identifiers differ.
        return cells, original

    def test_loss_boundary_and_deterministic_epoch(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            _, original = self.sample(path)
            prefixes, index, summary = week.scan(path, Codec())
            self.assertEqual((len(prefixes), len(index), summary["cases"]), (1, 3, 3))
            with patch.object(week, "load_tokenizer", Codec):
                first = list(week.iter_epoch(path, epoch=0))
                self.assertEqual(first, list(week.iter_epoch(path, epoch=0)))
                self.assertEqual(set(index), {r["case_id"] for r in week.iter_epoch(path, epoch=1)})
            for entry in first:
                cid = entry.pop("case_id")
                self.assertEqual(entry, original[cid])
                self.assertEqual(entry["input_ids"][entry["prompt_tokens"]:], [50])

    def test_four_example_updates_carry_every_case_across_epochs(self):
        epochs = [["a", "b", "c"], ["c", "a", "b"], ["b", "c", "a"]]
        with patch.object(week, "iter_epoch", side_effect=lambda path, epoch, seed: iter(epochs[epoch])):
            self.assertEqual(list(week.iter_batches(updates=2)), [["a", "b", "c", "c"], ["a", "b", "b", "c"]])
            self.assertEqual(list(week.iter_batches(updates=0)), [])

    def test_detects_changed_content_truncation_and_append(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "train.jsonl"
            self.sample(path)
            data = path.read_bytes()
            for mutation in (data[:data.rfind(b"\n", 0, -1) + 1], data + b"{}\n",
                             data.replace(b'"group":"quote"', b'"group":"other"')):
                path.write_bytes(mutation)
                with self.assertRaises(ValueError):
                    week.scan(path)

    def test_reconstructed_target_must_match_token_receipt(self):
        value = row("x")
        case, prefix, _, _ = week.pack("dialogue", value, week.tokenized(value, Codec()))
        case["target"] = "22"
        with self.assertRaisesRegex(ValueError, "Native tokens differ"):
            week.unpack(case, {case["prefix"]: prefix}, Codec())


if __name__ == "__main__":
    unittest.main()
