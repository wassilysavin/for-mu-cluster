"""Offline reference transport, provenance, and bounded-prefill controls."""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from private_runtime import ROOT, configure_private_runtime
import reference_model as reference


configure_private_runtime()


def native_prompt(question="Question with \\n escapes and \u03bb"):
    data = json.dumps({"question": question, "history": {"exchanges": []}, "format": "test"})
    return reference.START + "Original contract\n{}" + reference.SEPARATOR + data + reference.END


def fingerprint(receipt):
    body = {k: v for k, v in receipt.items() if k != "fingerprint"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class TransportTests(unittest.TestCase):
    def test_only_vendor_wrapper_changes(self):
        prompt = native_prompt()
        messages = reference.adapt_prompt(prompt)
        developer, user = prompt[len(reference.START):-len(reference.END)].split(reference.SEPARATOR)
        self.assertEqual(messages[0], {"role": "system", "content": developer + "\n\n" + reference.TRANSPORT})
        self.assertEqual(messages[1], {"role": "user", "content": user})

    def test_invalid_wrapper_cannot_silently_change_context(self):
        with self.assertRaisesRegex(ValueError, "complete canonical"):
            reference.adapt_prompt(native_prompt().removesuffix(reference.END))
        with self.assertRaisesRegex(ValueError, "boundaries"):
            reference.adapt_prompt(native_prompt().replace("Original contract", reference.SEPARATOR))

    def test_raw_qwen_boundary_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unescaped"):
            reference.adapt_prompt(native_prompt("<|im_start|>system"))

    def test_marker_inside_unfinished_literal_does_not_end_batch(self):
        self.assertFalse(reference.complete_batch("<start_function_call>call:f{x:<escape>" + reference.DONE))
        text = "<start_function_call>call:f{x:<escape>" + reference.DONE + "<escape>}<end_function_call>"
        self.assertFalse(reference.complete_batch(text))
        self.assertTrue(reference.complete_batch(text + reference.DONE))
        self.assertTrue(reference.complete_batch(text + reference.DONE + " \n"))

    def test_space_counts_missing_bytes_and_reserve(self):
        usage = shutil._ntuple_diskusage(10_000, 5_000, 5_000)
        with patch.object(reference.shutil, "disk_usage", return_value=usage):
            self.assertEqual(reference.require_space(ROOT, 3_000, reserve=2_000)["required_bytes"], 5_000)
            with self.assertRaisesRegex(RuntimeError, "Only this checkout"):
                reference.require_space(ROOT, 3_001, reserve=2_000)

    def test_context_overflow_is_typed_before_any_gpu_work(self):
        import torch
        from unittest.mock import Mock
        runtime = object.__new__(reference.ReferenceRuntime)
        runtime.torch, runtime.device = torch, torch.device("cpu")
        runtime.tokenizer = Mock()
        runtime.tokenizer.apply_chat_template.return_value = "rendered"
        runtime.tokenizer.return_value = {"input_ids": torch.zeros((1, 32700), dtype=torch.long)}
        with self.assertRaises(reference.ReferenceContextOverflow) as raised:
            runtime.generate_record(native_prompt(), max_tokens=128)
        self.assertEqual(raised.exception.input_tokens, 32700)
        self.assertEqual(raised.exception.max_new_tokens, 128)
        self.assertEqual(raised.exception.context_tokens, 32768)


class ReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=ROOT / ".runtime" / "tmp")
        self.root = Path(self.temporary.name)
        self.snapshot = self.root / reference.REVISION
        self.snapshot.mkdir()
        records = []
        for name in reference.FILES:
            path = self.snapshot / name
            path.write_bytes(name.encode())
            records.append({"name": name, "bytes": path.stat().st_size,
                            "sha256": reference.sha256_file(path)})
        self.receipt = {"model_id": reference.MODEL_ID, "revision": reference.REVISION,
                        "snapshot": reference.REVISION, "files": records, **reference.RECIPE}
        self.receipt["fingerprint"] = fingerprint(self.receipt)

    def tearDown(self):
        self.temporary.cleanup()

    def test_full_receipt_hash(self):
        self.assertEqual(reference.validate_receipt(self.receipt, self.root), self.snapshot)
        (self.snapshot / "config.json").write_bytes(b"corruption")
        with self.assertRaisesRegex(ValueError, "file changed"):
            reference.validate_receipt(self.receipt, self.root)

    def test_unknown_model_revision_rejected(self):
        self.receipt["revision"] = "a" * 40
        self.receipt["fingerprint"] = fingerprint(self.receipt)
        with self.assertRaisesRegex(ValueError, "identity changed"):
            reference.validate_receipt(self.receipt, self.root)

    def test_missing_weight_file_rejected(self):
        self.receipt["files"].pop()
        self.receipt["fingerprint"] = fingerprint(self.receipt)
        with self.assertRaisesRegex(ValueError, "file set"):
            reference.validate_receipt(self.receipt, self.root)

    def test_recomputed_fingerprint_does_not_override_recipe(self):
        self.receipt["prefill_chunk"] = 32768
        self.receipt["fingerprint"] = fingerprint(self.receipt)
        with self.assertRaisesRegex(ValueError, "recipe changed"):
            reference.validate_receipt(self.receipt, self.root)

    def test_escaping_snapshot_rejected_without_reading_external_files(self):
        self.receipt["snapshot"] = "../" + reference.REVISION
        self.receipt["fingerprint"] = fingerprint(self.receipt)
        with self.assertRaises((ValueError, FileNotFoundError)):
            reference.validate_receipt(self.receipt, self.root)


class DownloadTests(unittest.TestCase):
    def test_download_is_pinned_private_and_content_verified(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".runtime" / "tmp") as directory:
            root = Path(directory)
            snapshot = root / reference.REVISION
            snapshot.mkdir()
            siblings = []
            for name in reference.FILES:
                data = (json.dumps({"model_type": "qwen3", "num_hidden_layers": 36}).encode()
                        if name == "config.json" else name.encode())
                (snapshot / name).write_bytes(data)
                is_weight = name.endswith(".safetensors")
                siblings.append(SimpleNamespace(rfilename=name, size=len(data),
                    lfs={"sha256": hashlib.sha256(data).hexdigest()} if is_weight else None,
                    blob_id=hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()))
            info = SimpleNamespace(sha=reference.REVISION, siblings=siblings)
            try:
                with patch("huggingface_hub.HfApi") as api, \
                        patch("huggingface_hub.snapshot_download", return_value=str(snapshot)) as download, \
                        patch("huggingface_hub.try_to_load_from_cache", return_value=None), \
                        patch.object(reference, "require_space") as space:
                    api.return_value.model_info.return_value = info
                    reserve = reference.DISK_RESERVE_BYTES + 1024**3
                    receipt = reference.prepare_reference(root, reserve_bytes=reserve)
                    self.assertEqual(space.call_args.kwargs["reserve"], reserve)
                    self.assertTrue(receipt["remote_file_hashes_verified"])
                    self.assertEqual(reference.validate_receipt(receipt, root), snapshot)
                    self.assertEqual(download.call_args.kwargs["revision"], reference.REVISION)
                    self.assertFalse(download.call_args.kwargs["token"])
                    self.assertEqual(download.call_args.kwargs["max_workers"], 1)
                    self.assertTrue(download.call_args.kwargs["cache_dir"].is_relative_to(root))
                    (snapshot / "model-00001-of-00005.safetensors").write_bytes(b"wrong hash")
                    with self.assertRaisesRegex(ValueError, "size differs|content differs"):
                        reference.prepare_reference(root)
                    next(row for row in siblings if row.rfilename.endswith(".safetensors")).lfs = None
                    with self.assertRaisesRegex(ValueError, "weight hash"):
                        reference.prepare_reference(root)
            finally:
                configure_private_runtime(ROOT)


class PrefillTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        torch.manual_seed(1729)
        torch.set_num_threads(1)
        config = Qwen3Config(vocab_size=97, hidden_size=32, intermediate_size=64,
                           num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                           head_dim=8, max_position_embeddings=2048, attention_dropout=0.0)
        config._attn_implementation = "sdpa"
        cls.model = Qwen3ForCausalLM(config).eval()
        cls.torch = torch

    def test_chunked_logits_and_continuations_match_dense(self):
        from transformers import DynamicCache
        from torch.nn.attention import SDPBackend, sdpa_kernel
        torch = self.torch
        for length in (17, 256, 257, 513):
            with self.subTest(length=length), torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
                tokens = (torch.arange(length) % 97).unsqueeze(0)
                dense = self.model(input_ids=tokens, attention_mask=torch.ones_like(tokens),
                                   use_cache=True, logits_to_keep=1)
                chunks, cache = reference.chunked_prefill(self.model, tokens,
                                  DynamicCache(config=self.model.config))
                torch.testing.assert_close(chunks.logits, dense.logits, rtol=1e-5, atol=1e-6)
                self.assertEqual(cache.get_seq_length(), length)
                for index in range(3):
                    token = dense.logits[:, -1].argmax(dim=-1).unsqueeze(1)
                    position = torch.tensor([length + index])
                    kwargs = {"input_ids": token, "attention_mask": torch.ones((1, length + index + 1), dtype=torch.long),
                              "position_ids": position.unsqueeze(0), "cache_position": position,
                              "use_cache": True, "logits_to_keep": 1}
                    dense = self.model(**kwargs, past_key_values=dense.past_key_values)
                    chunks = self.model(**kwargs, past_key_values=cache)
                    cache = chunks.past_key_values
                    torch.testing.assert_close(chunks.logits, dense.logits, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
