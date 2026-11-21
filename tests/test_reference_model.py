"""Offline reference transport, provenance, and bounded-prefill controls."""

import hashlib
from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

from private_runtime import ROOT, configure_private_runtime
import reference_model as reference


configure_private_runtime()


def native_prompt(question="Question with \\n escapes and \u03bb"):
    data = json.dumps({"question": question, "history": {"exchanges": []}, "format": "test"})
    return reference.START + "Original contract\n{}" + reference.SEPARATOR + data + reference.END


def fingerprint(receipt):
    body = {k: v for k, v in receipt.items() if k != "fingerprint"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@contextmanager
def fixture_reference():
    with tempfile.TemporaryDirectory(dir=ROOT / ".runtime" / "tmp") as directory:
        root, manifest = Path(directory), {}
        for name, expected in reference.PINNED_FILES.items():
            data = (json.dumps({"model_type": "qwen3", "num_hidden_layers": 28}).encode()
                    if name == "config.json" else name.encode())
            path = reference.snapshot_path(root, expected["source"]) / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            item = {"source": expected["source"], "bytes": len(data)}
            if expected.get("sha256"):
                item["sha256"] = hashlib.sha256(data).hexdigest()
            else:
                item["git_blob"] = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            manifest[name] = item
        def download(model_id, **kwargs):
            source = next(key for key, pair in reference.SOURCE_PINS.items() if pair[0] == model_id)
            return str(reference.snapshot_path(root, source))
        try:
            with patch.object(reference, "PINNED_FILES", manifest), \
                    patch.object(reference, "MODEL_FILE_BYTES", sum(item["bytes"] for item in manifest.values())), \
                    patch("huggingface_hub.snapshot_download", side_effect=download) as fetch, \
                    patch("huggingface_hub.file_download.are_symlinks_supported", return_value=True), \
                    patch.object(reference, "require_reference_dependencies"), \
                    patch.object(reference, "require_space") as space:
                yield root, fetch, space
        finally:
            configure_private_runtime(ROOT)


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
        self.fixture = fixture_reference()
        self.root, self.fetch, self.space = self.fixture.__enter__()
        self.addCleanup(self.fixture.__exit__, None, None, None)
        self.snapshot = reference.snapshot_path(self.root, "config")
        self.receipt = reference.prepare_reference(self.root)

    def test_full_receipt_hash(self):
        self.assertEqual(reference.validate_receipt(self.receipt, self.root), self.snapshot)
        path = self.snapshot / "config.json"
        path.write_bytes(b"x" * path.stat().st_size)
        with self.assertRaisesRegex(ValueError, "content differs"):
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

    def test_gguf_path_and_provenance_cannot_change(self):
        self.receipt["gguf_path"] = str(Path(self.receipt["snapshot"]) / reference.GGUF_FILE)
        self.receipt["fingerprint"] = fingerprint(self.receipt)
        with self.assertRaisesRegex(ValueError, "GGUF location"):
            reference.validate_receipt(self.receipt, self.root)

    def test_receipt_is_identical_when_resuming_offline(self):
        offline = reference.prepare_reference(self.root, download=False)
        self.assertEqual(offline, self.receipt)
        self.assertTrue(all(call.kwargs["local_files_only"] for call in self.fetch.call_args_list[-2:]))


class DownloadTests(unittest.TestCase):
    def test_download_is_pinned_private_and_content_verified(self):
        with fixture_reference() as (root, fetch, space):
            reserve = reference.DISK_RESERVE_BYTES + 384 * 1024**2
            admit = Mock()
            receipt = reference.prepare_reference(root, reserve_bytes=reserve, admit_download=admit)
            self.assertEqual(space.call_args.kwargs["reserve"], reserve)
            self.assertTrue(receipt["pinned_file_hashes_verified"])
            self.assertEqual(reference.validate_receipt(receipt, root), reference.snapshot_path(root, "config"))
            self.assertEqual(len(fetch.call_args_list), 2)
            for call in fetch.call_args_list:
                source = next(key for key, pair in reference.SOURCE_PINS.items() if pair[0] == call.args[0])
                self.assertEqual(call.kwargs["revision"], reference.SOURCE_PINS[source][1])
                self.assertFalse(call.kwargs["token"])
                self.assertEqual(call.kwargs["max_workers"], 1)
                self.assertTrue(call.kwargs["cache_dir"].is_relative_to(root))
                self.assertEqual(set(call.kwargs["allow_patterns"]),
                                 {name for name, item in reference.PINNED_FILES.items() if item["source"] == source})
            admit.assert_called_once_with(reference.CACHE_OVERHEAD_BYTES)
            path = reference.snapshot_path(root, "weights") / reference.GGUF_FILE
            path.write_bytes(b"x" * path.stat().st_size)
            with self.assertRaisesRegex(ValueError, "content differs"):
                reference.prepare_reference(root)

    def test_space_admission_precedes_network_write(self):
        with fixture_reference() as (root, fetch, _):
            with self.assertRaisesRegex(RuntimeError, "Budget exceeded"):
                reference.prepare_reference(root, admit_download=Mock(side_effect=RuntimeError("Budget exceeded")))
            fetch.assert_not_called()

    def test_no_duplicate_copy_fallback(self):
        with fixture_reference() as (root, fetch, _), \
                patch("huggingface_hub.file_download.are_symlinks_supported", return_value=False):
            with self.assertRaisesRegex(RuntimeError, "one weight copy"):
                reference.prepare_reference(root)
            fetch.assert_not_called()

    def test_missing_bytes_only_credit_exact_pinned_partial(self):
        with fixture_reference() as (root, _, _):
            path = reference.snapshot_path(root, "weights") / reference.GGUF_FILE
            content = path.read_bytes()
            path.unlink()
            self.assertEqual(reference.missing_download_bytes(root), len(content))
            blobs = path.parent.parent.parent / "blobs"
            blobs.mkdir(exist_ok=True)
            (blobs / "unrelated.incomplete").write_bytes(b"x" * 1000)
            partial = blobs / (reference.PINNED_FILES[reference.GGUF_FILE]["sha256"] + ".incomplete")
            partial.write_bytes(content[:5])
            self.assertEqual(reference.missing_download_bytes(root), len(content) - 5)
            partial.rename(partial.with_suffix(""))
            partial.with_suffix("").write_bytes(content)
            self.assertEqual(reference.missing_download_bytes(root), 0)


class LoaderTests(unittest.TestCase):
    def test_gguf_is_explicit_local_and_never_saved_expanded(self):
        snapshot, gguf = Path("/personal/config"), Path("/personal/weights/model.gguf")
        with patch.object(reference, "version", side_effect=reference.REFERENCE_DEPENDENCIES.get), \
                patch("transformers.AutoConfig.from_pretrained") as config, \
                patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=(Mock(), {})) as load:
            model = reference.load_gguf_model(snapshot, gguf, "float16")
            config.assert_called_once_with(snapshot, local_files_only=True, trust_remote_code=False)
            load.assert_called_once_with(snapshot, config=config.return_value, gguf_file=str(gguf),
                                         local_files_only=True, trust_remote_code=False, torch_dtype="float16",
                                         attn_implementation="sdpa", output_loading_info=True)
            model.save_pretrained.assert_not_called()

    def test_unmapped_or_initialized_parameters_abort(self):
        with patch.object(reference, "version", side_effect=reference.REFERENCE_DEPENDENCIES.get), \
                patch("transformers.AutoConfig.from_pretrained"), \
                patch("transformers.AutoModelForCausalLM.from_pretrained",
                      return_value=(Mock(), {"missing_keys": ["model.layers.0.mlp.up_proj.weight"]})):
            with self.assertRaisesRegex(ValueError, "load exactly"):
                reference.load_gguf_model(Path("config"), Path("model.gguf"), "float16")

    def test_gguf_dependency_is_pinned(self):
        with patch.object(reference, "version", return_value="0.0"):
            with self.assertRaisesRegex(RuntimeError, "gguf==0.17.1"):
                reference.load_gguf_model(Path("config"), Path("model.gguf"), "float16")

    def test_accelerate_is_required_before_any_weight_download(self):
        installed = {"gguf": reference.GGUF_VERSION, "accelerate": "missing", "psutil": "7.0.0"}
        with patch.object(reference, "version", side_effect=installed.get):
            with self.assertRaisesRegex(RuntimeError, "accelerate==1.10.1"):
                reference.require_reference_dependencies()

    def test_real_qwen3_q8_0_load_preserves_quantized_parameters_and_logits(self):
        import gguf
        import numpy as np
        import torch
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from torch.nn.attention import SDPBackend, sdpa_kernel

        reference.require_reference_dependencies()
        torch.manual_seed(1729)
        torch.set_num_threads(1)
        config = Qwen3Config(vocab_size=96, hidden_size=64, intermediate_size=128,
                            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
                            head_dim=32, max_position_embeddings=1024, attention_dropout=0.0,
                            tie_word_embeddings=True)
        config._attn_implementation = "sdpa"
        original = Qwen3ForCausalLM(config).eval()
        quantized_state = {}
        names = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN3, config.num_hidden_layers)
        with tempfile.TemporaryDirectory(dir=ROOT / ".runtime" / "tmp") as directory:
            snapshot = Path(directory)
            config.save_pretrained(snapshot)
            path = snapshot / "synthetic-qwen3-q8_0.gguf"
            writer = gguf.GGUFWriter(str(path), "qwen3")
            writer.add_vocab_size(config.vocab_size)
            writer.add_context_length(config.max_position_embeddings)
            writer.add_embedding_length(config.hidden_size)
            writer.add_block_count(config.num_hidden_layers)
            writer.add_feed_forward_length(config.intermediate_size)
            writer.add_head_count(config.num_attention_heads)
            writer.add_head_count_kv(config.num_key_value_heads)
            writer.add_rope_dimension_count(config.head_dim)
            writer.add_rope_freq_base(config.rope_theta)
            writer.add_layer_norm_rms_eps(config.rms_norm_eps)
            for key, tensor in original.state_dict().items():
                if key == "lm_head.weight":
                    continue
                base, suffix = key.rsplit(".", 1)
                gguf_name = names.get_name(base)
                self.assertIsNotNone(gguf_name, key)
                array = tensor.numpy().astype(np.float32)
                if array.ndim == 2:
                    packed = gguf.quantize(array, gguf.GGMLQuantizationType.Q8_0)
                    writer.add_tensor(gguf_name + "." + suffix, packed,
                                      raw_dtype=gguf.GGMLQuantizationType.Q8_0)
                    array = gguf.dequantize(packed, gguf.GGMLQuantizationType.Q8_0)
                else:
                    writer.add_tensor(gguf_name + "." + suffix, array)
                quantized_state[key] = torch.from_numpy(array.copy())
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            quantized_state["lm_head.weight"] = quantized_state["model.embed_tokens.weight"]
            previous_dtype = torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.float16)
                expected = Qwen3ForCausalLM(config).eval()
            finally:
                torch.set_default_dtype(previous_dtype)
            expected.load_state_dict(quantized_state, strict=True)
            loaded = reference.load_gguf_model(snapshot, path, torch.float16).eval()
            self.assertEqual(set(loaded.state_dict()), set(quantized_state))
            for key, value in loaded.state_dict().items():
                torch.testing.assert_close(value, quantized_state[key].half(), rtol=0, atol=0)
            tokens = torch.tensor([[3, 7, 11, 19, 23]])
            with torch.inference_mode(), sdpa_kernel(SDPBackend.MATH):
                torch.testing.assert_close(loaded(tokens).logits, expected(tokens).logits, rtol=0, atol=0)
            self.assertEqual({item.name for item in snapshot.iterdir()}, {"config.json", path.name})


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
