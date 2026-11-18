"""CPU regressions for exact transfer TRAIN membership, checkpoint binding and rereading."""

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import struct
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import new_fit_inputs as fit


def write(root, rel, value):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = value if isinstance(value, bytes) else json.dumps(value).encode()
    path.write_bytes(raw)
    return {"path": str(rel), "sha256": sha256(raw).hexdigest(), "bytes": len(raw)}


def lines(rows):
    return b"".join(json.dumps(r).encode() + b"\n" for r in rows)


def checkpoint(step, identity, parent):
    header = json.dumps({"__metadata__": {"format": "gemprf-torch-volta-fp32-checkpoint-v2", "step": str(step),
                          "lineage": json.dumps({"identity_sha256": identity, "initial_checkpoint_sha256": parent})}}).encode()
    return struct.pack("<Q", len(header)) + header


class InputTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory(dir=fit.ROOT)
        self.addCleanup(tmp.cleanup)
        self.root = root = Path(tmp.name)
        roots = [{"root_id": str(i), "lineage": f"pair-{i // 2}", "group": "retry"} for i in range(160)]
        declarations = {arm: [{"root_id": str(i // 2), "slot": i % 2, "meaning": 0 if arm == "ordinary" else i % 2,
                               "case_id": f"{arm}-{i}", "continuation": {"kind": "answer_again"}}
                              for i in range(320)] for arm in fit.ARMS}
        frozen = {f"transfer-train-{k}.jsonl": write(root, f"assets/transfer-train-{k}.jsonl", lines(v))["sha256"]
                  for k, v in {"roots": roots, **declarations}.items()}
        assets = {k: write(root, f"assets/{k}", k.encode()) for k in ("base", "registry.json", "tokenizer.json", "train-full-v2.jsonl")}
        frozen.update({k: assets[k]["sha256"] for k in assets if k != "base"})
        write(root, "experiments.json", {"assets": {"tokenizer": assets["tokenizer.json"], "base": assets["base"]}})
        contexts = write(root, "assets/diagnostic-train-contexts.jsonl", b"{}\n")
        cm = write(root, "assets/diagnostic-train-contexts.json", {"contexts": contexts})
        prefix = write(root, fit.PREFIX_RECEIPTS, b"prefix")
        shared = [f"old-{i}" for i in range(320)]
        pos = fit.arrangements(shared, declarations)
        tr_data = lines({"case_id": c, "positions": p, "input_ids": [7, 8, 9, 50], "prompt_tokens": 3} for c, p in pos.items())
        tr = write(root, fit.RUN / "train/train.jsonl", tr_data)
        self.positions = pos
        summary = {"variable_cases_per_arm": 320, "frozen_sha256": frozen}
        rec = {"schema": "semantic-transfer-train-v1", "count": 640, "updates": 160, "common_cases": shared,
               "identity": dict(summary, observed_receipts_sha256=prefix["sha256"], code={}),
               "token_balance": {arm: {"prompt": 1920, "target": 640} for arm in fit.ARMS},
               "schedule_keys": {arm: {str(s): f"{arm}-{s}" for s in fit.SEEDS} for arm in fit.ARMS},
               "presented": {arm: {"groups": {"shared_replay": 320, "retry": 320}} for arm in fit.ARMS},
               "bytes": len(tr_data), "sha256": tr["sha256"]}
        tr_receipt = write(root, fit.RUN / "train/train-receipt.json", rec)
        old = {"sha256": "a" * 64}
        common = {"train": tr, "code": {"files": []}, "base": assets["base"], "tokenizer": assets["tokenizer.json"],
                  "initial_checkpoint": old}
        manifest = {"common": common, "training": summary, "train_receipt": tr_receipt,
                    "token_balance": rec["token_balance"], "identities": {}, "fresh": {"bundle_sha256": "e" * 64}}
        arms = {}
        for arm in fit.ARMS:
            n = arm + "-1729"
            directory = Path("runs") / ("semantic-transfer-v1-" + n)
            ids = sorted((c for c in pos if n in pos[c]), key=lambda c: pos[c][n])
            identity = {"train": dict(tr, rows=640, arm=n, schedule_sha256=fit.digest(ids)),
                        "recipe": {"examples_per_update": 4, "initial_step": 7991,
                                   "training": {"arm": n, "updates": 160}, "milestones": [160]}}
            stamp = fit.digest(identity)
            write(root, directory / "run.json", {"identity": identity, "identity_sha256": stamp})
            manifest["identities"][n] = stamp
            model = write(root, directory / "checkpoint-step-8151.safetensors", checkpoint(8151, stamp, old["sha256"]))
            arms[arm] = {"checkpoint": model, "scores": {"fresh": {"summary": {"supported": {"passed": 0, "planned": 1}}}}}
        self.result = {"pairs": [{"seed": 1729, "arms": arms}]}
        write(root, fit.RUN / "result.json", self.result)
        for flag in [patch.object(fit, "FROZEN", frozen), patch.object(fit, "CONTEXT_MANIFEST_SHA", cm["sha256"]),
                     patch.object(fit, "load_saved_transfer", return_value=(manifest, self.result, {}, {}))]:
            flag.start()
            self.addCleanup(flag.stop)

    def test_exact_schedule_and_only_completed_pair(self):
        inputs = fit.load_inputs(self.root)
        self.assertEqual(set(inputs["arms"]), {"ordinary-1729", "contrast-1729"})
        for name in inputs["arms"]:
            records = list(fit.iter_cases(inputs, name, self.root))
            self.assertEqual(len(records), 640)
            self.assertEqual(Counter(r["section"] for r in records), {"variable": 320, "shared": 320})
            self.assertEqual([r["position"] for r in records], list(range(640)))
            self.assertTrue(all(self.positions[r["case_id"]][name] == r["position"] for r in records))
            self.assertTrue(all(r["input_ids"][:r["prompt_tokens"]] == [7, 8, 9] for r in records))
            self.assertTrue(all(r["case_id"].startswith(("shared:", name.split("-")[0])) for r in records))

    def test_rejects_same_size_token_replacement_after_verification(self):
        inputs = fit.load_inputs(self.root)
        path = self.root / inputs["train"]["path"]
        path.write_bytes(path.read_bytes().replace(b"[7, 8, 9, 50]", b"[7, 8, 1, 50]"))
        with self.assertRaisesRegex(ValueError, "changed at"):
            next(fit.iter_cases(inputs, "ordinary-1729", self.root))

    def test_rejects_other_seed_checkpoint_even_if_file_record_matches(self):
        model = self.result["pairs"][0]["arms"]["contrast"]
        model["checkpoint"] = write(self.root, model["checkpoint"]["path"], checkpoint(8151, "f" * 64, "a" * 64))
        with self.assertRaisesRegex(ValueError, "lineage"):
            fit.load_inputs(self.root)

    def test_rejects_changed_row_order_metadata_before_inference(self):
        path = self.root / fit.RUN / "train/train.jsonl"
        data = path.read_bytes()
        start = data.index(b'"ordinary-1729": ')
        at = start + len(b'"ordinary-1729": ')
        path.write_bytes(data[:at] + (b"1" if data[at:at+1] == b"0" else b"0") + data[at+1:])
        with self.assertRaisesRegex(ValueError, "TRAIN position"):
            fit.load_inputs(self.root)


if __name__ == "__main__":
    unittest.main()
