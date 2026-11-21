"""Independently replay every authored weekly row against the immutable corpus."""

import hashlib
import unittest
from collections import Counter

from evaluation.functiongemma import decodeCalls
from data_tools import extended_conditions as conditions
from data_tools import extended_histories, extended_scope, week_compositions, week_dialogues
from data_tools.common import ASSETS, compact, rows, sha
from data_tools.week_replay_check import USER_TURN, verifier

WEEK_SHA = "0d0c5e28ef54f598b3f72ebd09de6dd79a0f299d46b09a5d726536fa582cd198"
AUTHOR_OUTPUTS = {
    "composition": "94a04615be621f6b7c7aad54483b6e8167547e89bd47d741cfa371afcd656916",
    "dialogue": "582cc08df6def7a27c9f2bd60a139b5a980db748b6e45ad3a0d51c22c45bbc4a",
    "scientific_conditions": "1cefb0996e93994218f728e83b4f6591a0c26921668158a7c93024e14ba75a37",
    "history_followup": "997d8b950504fb50a44dc151eeb5a55a0a50fb0aec5bf24bdd194054e4904e5e",
    "scope_change": "79717449189f8705a996f419a279294dae00a094fcff7d7e449478903c9ada70",
}


class FrozenAuthorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = ASSETS / "week-train-v2.jsonl"
        with cls.corpus.open("rb") as source:
            digest = hashlib.sha256()
            for chunk in iter(lambda: source.read(1 << 20), b""):
                digest.update(chunk)
            cls().assertEqual(digest.hexdigest(), WEEK_SHA)

    def replay(self, stream, generate, planned):
        expected = (r for r in rows(self.corpus) if r["type"] == "example" and r["stream"] == stream)
        seen, groups = set(), Counter()
        output = hashlib.sha256()

        def emit(row):
            output.update((compact(row) + "\n").encode("utf-8"))
            frozen = next(expected)
            self.assertEqual(row["split"], "train")
            for key in (
                "case_id",
                "family_id",
                "group",
                "history_origin",
                "ancestor_case_ids",
                "question",
                "calls",
                "target",
            ):
                self.assertEqual(row[key], frozen[key], (row["case_id"], key))
            prefix, suffix = row["model_input"].split(USER_TURN)
            self.assertEqual(sha(prefix), frozen["prefix"], row["case_id"])
            self.assertEqual(USER_TURN + suffix, frozen["input_suffix"], row["case_id"])
            self.assertEqual(decodeCalls(row["target"]), row["calls"])
            self.assertNotIn(row["case_id"], seen)
            seen.add(row["case_id"])
            groups[row["group"]] += 1

        generate(emit)
        self.assertIsNone(next(expected, None))
        self.assertEqual(len(seen), planned)
        self.assertEqual(output.hexdigest(), AUTHOR_OUTPUTS[stream])
        return groups

    def test_compositions(self):
        def generate(emit):
            for row in week_compositions.buildCompositions(True):
                emit(row)

        groups = self.replay("composition", generate, 740)
        self.assertEqual(
            groups,
            {
                "science_argument_contrast": 100,
                "science_same_subject": 224,
                "science_same_operation": 224,
                "science_three_obligations": 192,
            },
        )

    def test_dialogues(self):
        groups = self.replay(
            "dialogue", lambda emit: week_dialogues.buildWeekDialogues({"render": True, "emit": emit}), 1472
        )
        self.assertEqual(len(groups), 14)

    def test_conditions(self):
        def generate(emit):
            for row in conditions.expandedScience(True):
                emit(row)

        self.assertEqual(
            self.replay("scientific_conditions", generate, 3601),
            {"scientific_direct": 793, "correct_condition": 236, "select_operation": 1342, "correct_subject": 1230},
        )

    def test_histories(self):
        groups = self.replay(
            "history_followup",
            lambda emit: extended_histories.buildExtendedHistories({"render": True, "emit": emit}),
            5472,
        )
        self.assertEqual(len(groups), 11)

    def test_mixed_scope(self):
        self.replay(
            "scope_change", lambda emit: extended_scope.buildExtendedScope({"render": True, "emit": emit}), 1932
        )

    def test_frozen_replays(self):
        blocks, prefixes = [], {}
        verify = verifier()
        count = 0
        for row in rows(self.corpus):
            if row["type"] == "developer_blocks":
                self.assertEqual(row["start"], len(blocks))
                blocks.extend(row["fragments"])
            elif row["type"] == "developer_prefix":
                text = "},{".join(blocks[i] for i in row["blocks"])
                self.assertEqual(sha(text), row["sha256"])
                prefixes[row["sha256"]] = text
            elif row.get("stream") == "replay":
                (source_id,) = row["ancestor_case_ids"]
                prompt = prefixes[row["prefix"]] + row["input_suffix"]
                proof = verify({**row, "source_case_id": source_id, "model_input": prompt})
                self.assertEqual(proof["kind"], "verified")
                self.assertEqual(proof["prompt_sha256"], sha(prompt))
                self.assertEqual(proof["target_sha256"], sha(row["target"]))
                count += 1
        self.assertEqual(count, 2957)


class AuthorBoundaryTests(unittest.TestCase):
    def test_independent_anchors(self):
        def row(text, origin="fresh", **params):
            return {
                "question": text,
                "history_origin": origin,
                "calls": [{"name": "get_default_value", "arguments": {"subject": "TR", **params}}],
            }

        self.assertTrue(week_compositions.usableAnchor(row("What is the default TR value?")))
        for bad in (
            row("What is TR?", origin="derived_history_v2"),
            row("What is its documented value?"),
            row("What is STRay?"),
            row("Give only the TR default."),
            row("What is TR?", _clarify=True),
        ):
            self.assertFalse(week_compositions.usableAnchor(bad))

    def test_boolean_literal_and_bound_wording(self):
        def questions(name, **args):
            return conditions.directQuestions({"name": name, "arguments": args})

        for flag, description in [(True, "enabled"), (False, "disabled")]:
            qs = questions("get_value_guidance", subject="num_sigmas", refinement_enabled=flag)
            self.assertEqual(len(qs), 2)
            self.assertTrue(all("refinement is " + description in q for q in qs))
        self.assertTrue(all("refinement" not in q for q in questions("get_value_guidance", subject="num_sigmas")))
        self.assertEqual(questions("get_value_guidance", subject="TR", invented=True), [])
        for literal in ("False", "true"):
            qs = questions("explain_flag_literal_effect", subject="BIDS.enable", value=literal)
            self.assertTrue(qs and all(compact(literal) in q and "XML" in q for q in qs))
        for bound, direction in (("minimum", "lower"), ("maximum", "upper")):
            self.assertIn(
                direction, questions("explain_item_count_limits", subject="concatenate_item", bound=bound)[0]
            )

    def test_revision_default_and_scope_selection(self):
        qs = conditions.directQuestions({"name": "get_default_value", "arguments": {"subject": "version"}})
        self.assertEqual(len(qs), 2)
        for q in qs:
            self.assertRegex(q, r"this (?:GEM-pRF )?revision")
            self.assertIn("default", q)
            self.assertNotRegex(q, "fallback|example configuration|omitted")
        for style in range(6):
            first, second = [conditions.selectQuestion("Option A?", "Option B?", pick, style) for pick in (0, 1)]
            self.assertNotEqual(first, second)
            self.assertTrue(all(q.index("Option A?") < q.index("Option B?") for q in (first, second)))


if __name__ == "__main__":
    unittest.main()
