"""Offline tests for strict Hermes conversion and unchanged semantic inputs."""

from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import qwen_transport as transport
import reference_model as reference


ROOT = Path(__file__).resolve().parent


def prompt():
    developer = "\n".join([transport.SOURCE_OPENING, "Keep all semantic instructions.",
                             transport.SOURCE_OUTPUT, '{"Functions":["unchanged"],"Domains":{"values":[0,false]}}'])
    user = json.dumps({"format": "gemprf-functiongemma-input-v2", "history": {"exchanges": []},
                       "question": "Exact wording: \\u003c and λ\n"}, ensure_ascii=True)
    return reference.START + developer + reference.SEPARATOR + user + reference.END


def block(name="explain_purpose", arguments=None):
    call = {"name": name, "arguments": {"subject": "TR"} if arguments is None else arguments}
    return "<tool_call>\n" + json.dumps(call, ensure_ascii=False) + "\n</tool_call>"


def record(text=None):
    return {"raw_text": (block() if text is None else text) + "<|im_end|>",
            "raw_token_ids": [1, 2], "finish_reason": "eos",
            "eos_token_id": 2, "eos_token_text": "<|im_end|>"}


class PromptTests(unittest.TestCase):
    def test_only_two_transport_lines_change_and_user_is_exact(self):
        original = reference.adapt_prompt(prompt())
        adapted = transport.adapt_prompt(prompt())
        self.assertEqual(adapted[1], original[1])
        expected = original[0]["content"].removesuffix("\n\n" + reference.TRANSPORT)
        expected = expected.replace(transport.SOURCE_OPENING, transport.TARGET_OPENING)
        expected = expected.replace(transport.SOURCE_OUTPUT, transport.TARGET_OUTPUT)
        self.assertEqual(adapted[0]["content"], expected + "\n\n" + transport.TRANSPORT)

    def test_unknown_contract_or_boundaries_fail_closed(self):
        for source in [prompt().replace(transport.SOURCE_OPENING, "different"),
                       prompt().replace(transport.SOURCE_OUTPUT, transport.SOURCE_OUTPUT + "\n" + transport.SOURCE_OUTPUT),
                       prompt().replace("Keep all semantic instructions.", "<|im_start|>system"),
                       prompt()[:-len(reference.END)]]:
            with self.subTest(source=source[:100]), self.assertRaises(ValueError):
                transport.adapt_prompt(source)

    def test_official_template_receives_full_messages_and_non_thinking(self):
        tokenizer = Mock()
        tokenizer.apply_chat_template.return_value = "rendered"
        self.assertEqual(transport.render_prompt(tokenizer, prompt()), "rendered")
        tokenizer.apply_chat_template.assert_called_once_with(transport.adapt_prompt(prompt()),
            tokenize=False, add_generation_prompt=True, enable_thinking=False)


class ParseTests(unittest.TestCase):
    def test_two_calls_keep_order_and_typed_values(self):
        arguments = {"subject": "TR", "literal": "x\nλ \\ path <tool_call> </tool_call>",
                     "false_value": False, "zero": 0, "empty": "", "null": None,
                     "nested": {"items": [0, True, -1.25, "a"]}}
        raw = block(arguments=arguments) + "\n" + block("repeat_answer", {"target": "t3c2"})
        calls = transport.parse_calls(raw)
        self.assertEqual(calls, [{"name": "explain_purpose", "arguments": arguments},
                                 {"name": "repeat_answer", "arguments": {"target": "t3c2"}}])
        native = transport.encode_native(calls)
        decoded = subprocess.check_output(["node", "--input-type=module", "-e",
            "import {decodeCalls} from './evaluation/functiongemma.mjs'; let s=''; "
            "for await(const b of process.stdin)s+=b; console.log(JSON.stringify(decodeCalls(s)));"],
            input=native, text=True, cwd=ROOT)
        self.assertEqual(json.loads(decoded), calls)

    def test_duplicate_calls_and_unknown_names_are_not_repaired(self):
        calls = transport.parse_calls(block("unknown_function") * 2)
        self.assertEqual(len(calls), 2)
        self.assertEqual([call["name"] for call in calls], ["unknown_function"] * 2)

    def test_quoted_control_markers_are_literal_strings(self):
        calls = transport.parse_calls(block(arguments={"value": "<|im_end|><start_function_response>"}))
        self.assertEqual(calls[0]["arguments"]["value"], "<|im_end|><start_function_response>")

    def test_malformed_batches_never_keep_valid_prefix(self):
        invalid = ["", "prose" + block(), block() + "prose", "```json\n" + block() + "\n```",
                   "[" + block() + "]", block() + "<tool_call>{", block().replace("</tool_call>", ""),
                   '<tool_call>{"name":"x","arguments":{},"extra":1}</tool_call>',
                   '<tool_call>{"name":"x","name":"y","arguments":{}}</tool_call>',
                   '<tool_call>{"name":"x","arguments":{"a":1,"a":2}}</tool_call>',
                   '<tool_call>{"name":"x","arguments":[]}</tool_call>',
                   '<tool_call>{"name":"x","arguments":{"a":NaN}}</tool_call>',
                   '<tool_call>{"name":"x","arguments":{"a":1e999}}</tool_call>',
                   block(arguments={"a": "<escape>"}), block(arguments={"__proto__": {}}),
                   block(arguments={"a": 2**53}), block(arguments={"a": "\ud800"}),
                   block("bad-name"), block(arguments={"bad-key": 1}),
                   '<start_function_call>call:x{}<end_function_call><start_function_response>']
        for raw in invalid:
            with self.subTest(raw=ascii(raw[:100])), self.assertRaises((ValueError, UnicodeError)):
                transport.parse_calls(raw)

    def test_depth_and_size_fail_closed(self):
        nested = None
        for _ in range(26):
            nested = [nested]
        for raw in [block(arguments={"a": nested}), " " * 131073]:
            with self.assertRaises(ValueError):
                transport.parse_calls(raw)


class CompletionTests(unittest.TestCase):
    def test_complete_eos_only_removes_exact_terminal_control_token(self):
        raw = record(block(arguments={"a": "<|im_end|>"}))
        before = json.loads(json.dumps(raw))
        converted = transport.convert_generation(raw)
        self.assertTrue(converted["complete"])
        self.assertTrue(converted["transport_valid"])
        self.assertEqual(converted["call_count"], 1)
        self.assertIn("<escape><|im_end|><escape>", converted["native_text"])
        self.assertEqual(raw, before)

    def test_length_and_early_complete_marker_do_not_complete_batch(self):
        for reason in ["length", "complete_batch", None]:
            result = transport.convert_generation({**record(), "finish_reason": reason})
            self.assertFalse(result["complete"])
            self.assertEqual(result["native_text"], "")
            self.assertEqual(result["transport_error"], "incomplete_qwen_output")

    def test_bad_eos_or_body_is_not_repaired(self):
        original = record()
        variants = [{**original, "raw_token_ids": []}, {**original, "eos_token_id": 7},
                    {**original, "eos_token_text": "</tool_call>"},
                    {**original, "raw_text": block()}, {**original, "raw_text": original["raw_text"] * 2},
                    record(block() + "prose"), record(block() + "<tool_call>{")]
        for value in variants:
            with self.subTest(value=value):
                converted = transport.convert_generation(value)
                self.assertFalse(converted["complete"])
                self.assertEqual(converted["native_text"], "")
                self.assertEqual(converted["transport_error"], "invalid_qwen_output")


class RuntimeTests(unittest.TestCase):
    def runtime(self, sequence, input_tokens=8):
        class Logits:
            def __init__(self, token): self.token = token
            def __getitem__(self, index): return self
            def float(self): return self
            def argmax(self): return SimpleNamespace(item=lambda: self.token)
        class Encoded:
            shape = (1, input_tokens)
            def to(self, device): return self
        class Tokenizer:
            pieces = {1: block(), 3: block("repeat_answer", {"target": "t1c1"}), 2: "<|im_end|>"}
            def apply_chat_template(self, *args, **kwargs): return "all rendered input"
            def __call__(self, *args, **kwargs): return {"input_ids": Encoded()}
            def decode(self, tokens, **kwargs): return "".join(self.pieces[token] for token in tokens)
        iterator = iter(sequence)
        def output(): return SimpleNamespace(logits=Logits(next(iterator)), past_key_values="cache")
        model = Mock(side_effect=lambda **kwargs: output())
        model.generation_config = SimpleNamespace(eos_token_id=2)
        runtime = object.__new__(transport.Runtime)
        runtime.device, runtime.tokenizer, runtime.model = "cuda:0", Tokenizer(), model
        runtime.torch = SimpleNamespace(inference_mode=nullcontext,
            cuda=SimpleNamespace(reset_peak_memory_stats=Mock(), synchronize=Mock(), max_memory_allocated=lambda device: 123),
            isfinite=lambda value: SimpleNamespace(all=lambda: True), tensor=lambda *args, **kwargs: None,
            ones=lambda *args, **kwargs: None, long="long")
        modules = {"transformers": ModuleType("transformers"), "torch": ModuleType("torch"),
                   "torch.nn": ModuleType("torch.nn"), "torch.nn.attention": ModuleType("torch.nn.attention")}
        modules["transformers"].DynamicCache = lambda **kwargs: "cache"
        modules["torch.nn.attention"].SDPBackend = SimpleNamespace(MATH="math")
        modules["torch.nn.attention"].sdpa_kernel = lambda *args: nullcontext()
        model.config = None
        return runtime, modules, lambda *args: (output(), "cache")

    def test_generation_continues_past_first_call_until_eos(self):
        runtime, modules, prefill = self.runtime([1, 3, 2])
        with patch.dict(sys.modules, modules), patch.object(reference, "chunked_prefill", prefill):
            result = runtime.generate_record(prompt(), max_tokens=5)
        self.assertEqual(result["raw_token_ids"], [1, 3, 2])
        self.assertEqual(result["call_count"], 2)
        self.assertEqual(result["finish_reason"], "eos")
        self.assertTrue(result["transport_valid"])
        self.assertTrue(result["raw_text"].endswith("<|im_end|>"))
        self.assertEqual(runtime.model.call_count, 2)

    def test_truncated_valid_call_is_incomplete_without_eos(self):
        runtime, modules, prefill = self.runtime([1])
        with patch.dict(sys.modules, modules), patch.object(reference, "chunked_prefill", prefill):
            result = runtime.generate_record(prompt(), max_tokens=1)
        self.assertEqual(result["raw_token_ids"], [1])
        self.assertEqual(result["finish_reason"], "length")
        self.assertFalse(result["complete"])
        self.assertEqual(runtime.model.call_count, 0)

    def test_context_overflow_precedes_forward_and_never_truncates(self):
        runtime, modules, prefill = self.runtime([1], input_tokens=32767)
        with patch.dict(sys.modules, modules), patch.object(reference, "chunked_prefill", prefill):
            with self.assertRaises(reference.ReferenceContextOverflow):
                runtime.generate_record(prompt(), max_tokens=5)
        runtime.model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
