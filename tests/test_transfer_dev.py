"""Exercise the fresh DEV protocol with recorded, synthetic provider replies."""

from private_runtime import python_tool
import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CHECK = [*python_tool("data_tools/transfer_dev.py"), "assets/registry.json", "assets/transfer-dev-v1.json"]
DECLINE = "<start_function_call>call:decline_request{}<end_function_call><start_function_response>"


def simulate(path, *, disconnect_after=None, binding=None, provider=None):
    generated = []
    result = None
    spec = {"diagnostic_worker_sha256": binding} if binding else {}
    with subprocess.Popen([*CHECK, str(path), json.dumps(spec)], cwd=ROOT, text=True, encoding="utf-8",
                          stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as proc:
        for line in proc.stdout:
            request = json.loads(line)
            if request["kind"] == "result":
                result = request["score"]
                continue
            if request["kind"] != "generate":
                raise AssertionError(request["kind"])
            if len(generated) == disconnect_after:
                proc.stdin.close()
                break
            generated.append(request["case_id"])
            reply = {"raw_ids": [50], "raw_text": DECLINE}
            if provider:
                reply["diagnostic_worker_sha256"] = provider
            proc.stdin.write(json.dumps(reply) + "\n")
            proc.stdin.flush()
        proc.stdin.close()
        error = proc.stderr.read()
        return proc.wait(), generated, result, error


class TransferDevTests(unittest.TestCase):
    def test_other_checkpoint_cannot_resume_a_partial_run(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as scratch:
            path = Path(scratch, "observations.jsonl")
            status, generated, _, error = simulate(path, disconnect_after=1, binding="a"*64, provider="a"*64)
            self.assertNotEqual(status, 0)
            self.assertEqual(len(generated), 1)
            self.assertIn("provider disconnected", error)
            original = path.read_bytes()
            status, generated, _, error = simulate(path, binding="b"*64, provider="b"*64)
            self.assertNotEqual(status, 0)
            self.assertEqual(generated, [])
            self.assertIn("identity changed", error)
            self.assertEqual(original, path.read_bytes())

    def test_resume_and_replay_use_model_history(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as scratch:
            path = Path(scratch, "observations.jsonl")
            status, partial, _, error = simulate(path, disconnect_after=69)
            self.assertNotEqual(status, 0)
            self.assertIn("provider disconnected", error)
            self.assertEqual(len(partial), 69)
            with path.open("ab") as stream:
                stream.write(b'{"case_id":')
            status, new, result, error = simulate(path)
            self.assertEqual(status, 0, error)
            self.assertEqual(len(set(partial + new)), result["observed_records"])
            self.assertEqual(result["outside_scope"]["passed"], result["outside_scope"]["planned"])
            records = [json.loads(line) for line in path.read_text().splitlines()]
            first = next(i for i, r in enumerate(records) if r.get("dialogue_id"))
            self.assertEqual(records[first]["after_sha256"], records[first + 1]["before_sha256"])
            # A refusal commits a displayed message, but contains no scientific answer.
            absent = [r for r in records if r["panel"] == "closed_loop" and r["policy_branch"] == "absent"]
            self.assertTrue(any(r["expected_model_call_batches"][0][0]["name"] == "clarify_reference" for r in absent))
            status, new, replayed, error = simulate(path)
            self.assertEqual(status, 0, error)
            self.assertEqual(new, [])
            self.assertEqual(replayed, result)
            records[first]["success"] = not records[first]["success"]
            path.write_text("".join(json.dumps(r) + "\n" for r in records))
            status, _, _, error = simulate(path)
            self.assertNotEqual(status, 0)
            self.assertIn("replay differs", error)


if __name__ == "__main__":
    unittest.main()
