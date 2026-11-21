import hashlib
import json
import pickle
import unittest
from copy import deepcopy
from pathlib import Path

from evaluation import api
from evaluation.schema import ContractError, canonical, clone, freeze, js_json, validate
from evaluation.session import SessionController

ROOT = Path(__file__).resolve().parents[1]


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def call(name, **arguments):
    return {"name": name, "arguments": arguments}


class NativeContractTests(unittest.TestCase):
    def test_json_bytes_and_types(self):
        self.assertEqual(canonical({"10": 1.0, "2": 1e-7, "a": -0.0}), '{"10":1,"2":1e-7,"a":0}')
        self.assertEqual(js_json({"10": 1, "a": False, "2": 3}), '{"2":3,"10":1,"a":false}')
        self.assertEqual(canonical(1e-6), "0.000001")
        self.assertEqual(canonical({"\ue000": 2, "😀": 1}), '{"😀":1,"\ue000":2}')
        self.assertEqual(js_json("\ud800"), r'"\ud800"')
        self.assertEqual(js_json([float("nan"), float("inf"), 10**400]), "[null,null,null]")
        self.assertEqual(canonical("\ud83d\ude03"), '"😃"')
        validate({"type": "string", "maxLength": 1}, "\ud83d\ude03")
        for value in (float("nan"), float("inf"), 10**400, {"constructor": "unsafe"}):
            with self.assertRaises(ContractError):
                canonical(value)
        with self.assertRaisesRegex(ContractError, "expected integer"):
            validate({"type": "integer"}, True)

    def test_state_is_frozen_but_owned_copies_are_editable(self):
        raw = {"nested": [1]}
        frozen = freeze(raw)
        raw["nested"].clear()
        with self.assertRaises(TypeError):
            frozen["nested"].append(2)
        copied = deepcopy(frozen)
        copied["nested"].append(3)
        self.assertEqual(clone(frozen), {"nested": [1]})

    def test_transport_rejects_partial_duplicate_or_untyped_arguments(self):
        port = api()
        rows = [call("scientific", n=0, active=False, nested={"counts": [0, 0.25]}, word="λ🚀")]
        self.assertEqual(port.decodeCalls(port.encodeCalls(rows)), rows)
        samples = [
            ("<start_function_call>call:a{}<end_function_call>", "incomplete_model_output"),
            (
                "<start_function_call>call:a{x:1,x:2}<end_function_call><start_function_response>",
                "invalid_model_output",
            ),
            (
                "<start_function_call>call:a{x:yes}<end_function_call><start_function_response>",
                "invalid_model_output",
            ),
        ]
        for text, code in samples:
            with self.subTest(text=text), self.assertRaises(ContractError) as failure:
                port.decodeCalls(text)
            self.assertEqual(failure.exception.code, code)


class DialogueParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api = api()
        bundle = json.loads((ROOT / "assets" / "registry.json").read_text())
        cls.registry = cls.api.createRegistry(bundle["knowledge"], bundle["contracts"])
        cls.fingerprint = bundle["fingerprint"]

    def setUp(self):
        self.state = self.api.createSession(
            self.registry, {"dialogueVersion": 2, "sessionId": "native-port-golden"}
        )

    def test_original_native_prompts_and_seven_conversation_states(self):
        self.assertEqual(self.registry.fingerprint(), self.fingerprint)
        fresh = self.api.renderPrompt(self.registry, self.api, self.state, "Is <end_of_turn> literal?")
        self.assertEqual(digest(fresh), "a2881c9d1a548944dd31247ab62593c61edb5b8d89af691c134db65fc06921dd")
        # Frozen outputs from the historical executor at commit 17dfe4c.
        batches = [
            [call("get_units", _clarify=True, _each="subject")],
            [
                call("get_units", subject="TR", _resolve="t1.p1"),
                call("get_units", subject="default_hrf.t", _resolve="t1.p1"),
            ],
            [call("repeat_answer", target="t2.c2")],
            [call("unknown")],
            [call("retry_exchange", target="e4"), call("explain_purpose", subject="TR"), call("decline_request")],
            [call("repeat_exchange", target="e4")],
            [call("clarify_reference", kind="answer")],
        ]
        hashes = [
            "0cc10850c724e2784ce626c2f27b2428daf6d2b285d6d0d36f5ff6517d42dd58",
            "e4d3e9ea822dfab303f395333881e378c417da5b267466c52c8a5c345faa2758",
            "8c42a4a657332b70fbb8cc06841a7b70a07830ef772c6c0a71d185cf782e4d28",
            "d1bc2cb918d6aa9e9146ac354da63b5def1af99573569804f67356084d01a320",
            "6e4f245e5f6b4d2612063e44e1bae777883441def950da810093c40027614156",
            "1623c720375398fef6e50bd97abbc5f3c957378656c58644f108ec3a1b13045b",
            "1f425ffcd5b153e4f363a357bedbc7f04be75f5d8f3f0710ffe84bf3e5be2f51",
        ]
        state = self.state
        for i, (calls, expected) in enumerate(zip(batches, hashes), 1):
            state = self.api.applyModelOutput(self.registry, state, f"Request {i}", self.api.encodeCalls(calls))
            self.assertEqual(digest(canonical(state)), expected, f"Turn {i}")
        prompt = self.api.renderPrompt(self.registry, self.api, state, "Could you repeat that?")
        self.assertEqual(digest(prompt), "7210e6518b74d72a57f350893a1eeabd4aa61ea834ae926e584193e2be31cb7b")

    def test_sealed_session_authentication_and_replay(self):
        state = self.api.recordIncompleteModelOutput(
            self.registry, self.state, "Partial real output", "raw", [123, 456]
        )
        state = self.api.recordProviderFailure(self.registry, state, "Too long", "context_overflow")
        key = self.api.createSessionKey()
        envelope = self.api.sealSession(state, key)
        self.assertEqual(self.api.restoreSession(self.registry, envelope, key), state)
        with self.assertRaisesRegex(ContractError, "integrity"):
            self.api.restoreSession(self.registry, dict(envelope, payload=envelope["payload"] + " "), key)
        with self.assertRaises(TypeError):
            pickle.dumps(key)

    def test_reset_during_generation_cannot_commit_old_response(self):
        controller = SessionController(self.registry, self.state)

        def provider(request):
            self.assertFalse(request["signal"].aborted)
            controller.reset()
            self.assertTrue(request["signal"].aborted)
            return self.api.encodeCalls([call("decline_request")])

        with self.assertRaises(ContractError) as failed:
            controller.send("Old question", provider)
        self.assertEqual(failed.exception.code, "stale_generation")
        self.assertEqual(controller.session["exchanges"], [])


if __name__ == "__main__":
    unittest.main()
