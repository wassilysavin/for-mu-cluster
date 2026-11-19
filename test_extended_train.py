"""Regression checks for nested data, matched update counts and verified reconstruction."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from audit_week_train import wording_probe
import extended_train as ext
from test_week_train import Codec, row


class MatchedCourseTests(unittest.TestCase):
    def test_wording_probe_ignores_self_and_sibling_rows(self):
        records = [("Repeat?", "one", "quote"), ("Repeat?", "one", "quote"),
                   ("Repeat?", "two", "clarify"), ("Retry?", "one", "retry")]
        self.assertEqual(wording_probe(records), {"cases": 4, "seen_wording_elsewhere": 3,
                                                 "correct_among_seen": 0})

    def test_proportional_quotas_keep_rare_groups(self):
        groups = {"a": list(range(100)), "b": list(range(100, 149)), "c": [149]}
        chosen = ext.choose_control(groups, size=20)
        self.assertEqual(sum(map(len, chosen.values())), 20)
        self.assertEqual(chosen["c"], [149])
        self.assertIn(len(chosen["a"]), (12, 13))
        self.assertEqual(chosen, ext.choose_control(dict(reversed(groups.items())), size=20))
        with self.assertRaises(ValueError):
            ext.choose_control(groups, size=2)

    def test_both_arms_visit_same_stratum_at_each_position(self):
        groups = {i: [f"{i}/{j}" for j in range(size)] for i, size in enumerate((3000, 2600, 250, 1))}
        updates = 3200
        full, matched = [list(ext.draws(groups, arm=a, updates=updates)) for a in ("full", "matched")]
        self.assertEqual(len(full), 4 * updates)
        self.assertEqual([c.split("/")[0] for c in full], [c.split("/")[0] for c in matched])
        self.assertEqual(len(set(full)), 5851)
        self.assertEqual(len(set(matched)), ext.CONTROL_SIZE)
        self.assertEqual(len(set(full[:5851])), 5851)
        self.assertEqual(full[:640], list(ext.draws(groups, arm="full", updates=160)))
        self.assertEqual(list(ext.draws(groups, arm="full", updates=0)), [])

    def test_case_key_ignores_session_id_but_preserves_history(self):
        a = {"question": "Again?", "history": {"session_id": "abc", "exchanges": []}}
        b = {"question": "Again?", "history": {"session_id": "def", "exchanges": []}}
        self.assertEqual(ext.visible_key(a), ext.visible_key(b))
        b["history"]["exchanges"].append({"outcome": "error"})
        self.assertNotEqual(ext.visible_key(a), ext.visible_key(b))

    def test_verified_parent_and_streaming_build(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "parent.jsonl"
            dest = root / "v2.jsonl"
            codec = Codec()
            from hashlib import sha256
            import json
            h = sha256()
            block_ids = {}
            with old.open("wb") as f:
                ext.base.emit(f, h, {"format": ext.base.FORMAT, "split": "train"})
                for i, stream in enumerate(ext.base.STREAMS):
                    r = row(f"old-{i}")
                    r["calls"] = [{"name":"mock"}]
                    r["question"] += str(i)
                    r["model_input"] = r["model_input"].replace("Repeat it.", r["question"])
                    r["group"] = "turn"
                    seq = ext.base.tokenized(r, codec)
                    c, pre, _, _ = ext.base.pack(stream, r, seq)
                    if not i:
                        ext.base.write_prefix(f, h, c["prefix"], pre, block_ids)
                    ext.base.emit(f, h, c)
                f.write((json.dumps({"type":"summary", "cases":3}) + "\n").encode())
            def generated(script):
                r = row("new")
                r["calls"] = [{"name":"mock"}]
                yield r
            with (patch.object(ext.base, "DATA", old), patch.object(ext, "PARENT_SHA", ext.base.digest(old)),
                  patch.object(ext.base, "rendered", generated), patch.object(ext.base, "load_tokenizer", lambda: codec),
                  patch.object(ext.base, "public_development", lambda: (set(),set())),
                  patch.object(ext.base, "items", lambda path: [{"case_id":"train-fixture"}]),
                  patch.object(ext, "CONTROL_SIZE", 3), patch.object(ext, "AUTHORS", {"extra":"ignored.mjs"}),
                  patch.object(ext, "file_bindings", lambda: {}),
                  patch.object(ext, "choose_control", lambda groups, **kwargs: {g:ids[:1] for g, ids in groups.items()})):
                stats = ext.build(dest)
                self.assertEqual(stats["cases"], 4)
                _, offsets, _, check = ext.inspect(dest, codec)
                self.assertEqual(set(offsets), {"old-0","old-1","old-2","new"})
                self.assertEqual(check, stats)
                self.assertEqual(ext.build(dest), stats)
                modified = dest.read_bytes().replace(b'"group":"turn"', b'"group":"fail"',1)
                dest.write_bytes(modified)
                with self.assertRaisesRegex(ValueError, "parent records"):
                    ext.inspect(dest, codec)


if __name__ == "__main__":
    unittest.main()
