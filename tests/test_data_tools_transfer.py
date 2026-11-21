"""Transferred native schema pairing, observed-history migration and CLI controls."""

import hashlib
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from pathlib import Path

from data_tools import history_data, transfer_contract, transfer_train
from data_tools.common import compact, read_json, render, runtime, session, sha
from evaluation.functiongemma import encodeCalls
from evaluation.history_train import applyObservation
from evaluation.model_tools import modelTools
from evaluation.prompt import renderToolContract
from evaluation.registry import createRegistry
from evaluation.schema import ContractError
from evaluation.session import applyModelOutput


class TransferContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = read_json("registry.json")
        cls.original = createRegistry(cls.source["knowledge"], cls.source["contracts"])
        cls.revised = transfer_contract.transferRegistry(cls.source)
        cls.name = transfer_contract.NAME
        cls.pairs = [
            r["call"]["arguments"] for r in cls.source["contracts"]["bindings"] if r["call"]["name"] == cls.name
        ]

    def test_admitted_complete_pairs_have_source_bindings(self):
        for first in self.pairs:
            for second in self.pairs:
                args = {"subject": first["subject"], "reference": second["reference"]}
                call = {"name": self.name, "arguments": args}
                if args in self.pairs:
                    self.assertEqual(self.revised.resolve(call), self.original.resolve(call))
                else:
                    with self.assertRaises(ContractError) as error:
                        self.revised.validateCall(call)
                    self.assertEqual(error.exception.code, "invalid_arguments")

    def test_complete_paired_fixture_is_byte_stable(self):
        digest = hashlib.sha256()

        def emit(row):
            digest.update((compact(row) + "\n").encode("utf-8"))

        transfer_train.materialize({"fixture": True, "emit": emit})
        self.assertEqual(digest.hexdigest(), "04ba930431b4aefcc70a4f4c6f03b262f4b7f53d97cdb39f34eba08f60bf7449")

    def test_recover_interrupted_prefix_receipt(self):
        prefix = {"session_id": "synthetic-collection", "prefix": [{"question": "What is TR?"}]}
        immutable = {"specs_sha256": "fixture", "registry_sha256": self.original.fingerprint()}
        question = prefix["prefix"][0]["question"]
        before = session(self.original, prefix["session_id"])
        item = {
            "kind": "native",
            "raw_output": "malformed",
            "raw_ids": [50],
            "question": question,
            "model_input_sha256": sha(render(self.original, before, question)),
        }
        after = applyObservation(self.original, runtime(), before, question, item)
        record = {**immutable, "prefix_id": "p", "observations": [item], "actual_session": after}
        raw = history_data.line(record)
        broken = b'{"prefix_id":'
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "receipts.jsonl"
            path.write_bytes(raw + broken)
            with redirect_stdout(io.StringIO()) as log:
                saved = history_data.read_receipts(path, immutable, {"p": prefix}, self.original, runtime())
            summary = json.loads(log.getvalue())
            self.assertEqual(summary["kind"], "receipt_tail_recovered")
            self.assertEqual(Path(summary["preserved"]).read_bytes(), broken)
            self.assertEqual(path.read_bytes(), raw)
            self.assertEqual(saved["p"], record)
            changed = {**record, "actual_session": before}
            path.write_bytes(history_data.line(changed))
            with self.assertRaisesRegex(ValueError, "differs from replayed observations"):
                history_data.read_receipts(path, immutable, {"p": prefix}, self.original, runtime())

    def test_packed_contract_before_and_after_pending(self):
        old = session(self.original, "schema-test")
        new = session(self.revised, "schema-test")

        def same_prompt():
            before = renderToolContract(modelTools(self.original, old["semantic"], old["exchanges"]))
            after = renderToolContract(modelTools(self.revised, new["semantic"], new["exchanges"]))
            self.assertEqual(transfer_contract.upgradePacked(json.loads(before), self.source), after)

        def send(reg, state, call):
            return applyModelOutput(reg, state, "Check the requested match.", encodeCalls([call]))

        same_prompt()
        for args in (
            {"subject": self.pairs[0]["subject"], "_clarify": True},
            {"reference": self.pairs[0]["reference"], "_resolve": "p1"},
        ):
            if "_resolve" in args:
                args["_resolve"] = new["semantic"]["pending"][0]["id"]
            call = {"name": self.name, "arguments": args}
            old, new = send(self.original, old, call), send(self.revised, new, call)
            self.assertEqual(new["exchanges"][-1]["outcome"], "committed")
            same_prompt()
        target = new["semantic"]["turns"][-1]["answers"][0]["id"]

        def change(args):
            return send(
                self.revised, new, {"name": "change_" + self.name, "arguments": {"target": target, "changes": args}}
            )["exchanges"][-1]

        self.assertEqual(change(self.pairs[1])["outcome"], "committed")
        self.assertEqual(change({"subject": self.pairs[1]["subject"]})["error_code"], "invalid_arguments")

    def test_observed_receipt_failure_and_tampering(self):
        spec = {
            "session_id": "transfer-train-receipt-test",
            "prefix": [{"question": "What does TR specify?"}, {"question": "What does BIDS.basepath specify?"}],
        }
        record = {
            "specs_sha256": "fixture",
            "registry_sha256": self.original.fingerprint(),
            "model_fingerprint": {"checkpoint_step": 7205},
            "observations": [],
        }
        before = session(self.original, spec["session_id"])
        for i, step in enumerate(spec["prefix"]):
            question = step["question"]
            raw = (
                encodeCalls([{"name": "explain_purpose", "arguments": {"subject": "BIDS.basepath"}}])
                if i
                else "malformed"
            )
            obs = {
                "question": question,
                "kind": "native",
                "raw_output": raw,
                "model_input_sha256": sha(render(self.original, before, question)),
            }
            record["observations"].append(obs)
            before = applyObservation(self.original, runtime(), before, question, obs)
        record["actual_session"] = before
        migrated = transfer_train.observedSession(self.revised, self.original, record, spec, spec, "fixture")
        self.assertEqual(migrated["registry_id"], self.revised.fingerprint())
        for k in ("semantic", "exchanges"):
            self.assertEqual(migrated[k], before[k])
        self.assertEqual(migrated["exchanges"][0]["outcome"], "error")
        self.assertEqual(migrated["exchanges"][1]["semantic_turn_id"], "t1")
        tampered = deepcopy(record)
        tampered["actual_session"]["exchanges"][0]["assistant"] = "a different reply"
        with self.assertRaisesRegex(ValueError, "Saved TRAIN dialogue differs"):
            transfer_train.observedSession(self.revised, self.original, tampered, spec, spec, "fixture")


if __name__ == "__main__":
    unittest.main()
