from array import array
from collections import Counter
from contextlib import contextmanager, redirect_stdout, nullcontext
from hashlib import sha256
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import json
import os
import sys
import unittest
from unittest.mock import Mock, patch

import new_fit_workers as workers


class JournalTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory(dir=workers.ROOT / ".runtime/tmp")
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / "gpu-0.jsonl"
        self.binding = "b" * 64
        self.cases = ["a", "b"]
        self.hashes = {"a": "1"*64, "b": "2"*64}
        self.rows = [workers.make_record(self.binding, cid, self.hashes[cid], [75, 50]) for cid in self.cases]
        self.lines = [workers.encode(row) + b"\n" for row in self.rows]
        self.budget = SimpleNamespace(writing=lambda n: nullcontext())

    def read(self):
        return workers.journal_prefix(self.path, self.cases, self.binding, self.hashes, self.budget)

    def test_recover_each_prefix_of_new_record(self):
        for end in range(1, len(self.lines[1])-1):
            with self.subTest(end=end):
                self.assertTrue(workers.torn_prefix(self.lines[1][:end], self.binding, "b", self.hashes["b"]))
        for end in (25, self.lines[1].index(b"75")+1, len(self.lines[1])-18):
            self.path.write_bytes(self.lines[0] + self.lines[1][:end])
            self.assertEqual(self.read(), {"a": [75, 50]})
            self.assertEqual(self.path.read_bytes(), self.lines[0])
        self.path.write_bytes(self.lines[0] + self.lines[1][:-1])
        self.assertEqual(self.read(), {"a": [75, 50], "b": [75, 50]})
        self.assertEqual(self.path.read_bytes(), b"".join(self.lines))

    def test_rejects_corruption_wrong_shard_and_duplicate_without_modification(self):
        damaged = self.lines[1].replace(b"[75,50]", b"[76,50]")
        for tail in (b"broken", b"broken\n", self.lines[0], damaged, damaged[:-5],
                     self.lines[1].replace(b"[75,50]", b"[75]"), self.lines[1] + self.lines[0]):
            with self.subTest(tail=tail):
                original = self.lines[0] + tail
                self.path.write_bytes(original)
                with self.assertRaises(ValueError):
                    self.read()
                self.assertEqual(self.path.read_bytes(), original)
        self.path.write_bytes(self.lines[0])
        self.hashes["a"] = "9"*64
        with self.assertRaises(ValueError):
            self.read()

    def test_never_follows_foreign_symlinks_or_modifies_multilink_records(self):
        foreign = self.path.with_name("saved.jsonl")
        foreign.write_bytes(self.lines[0])
        self.path.symlink_to(foreign)
        with self.assertRaises(ValueError):
            self.read()
        self.path.unlink()
        os.link(foreign, self.path)
        with self.assertRaises(ValueError):
            workers.append_record(self.path, self.rows[1], self.budget)
        with self.assertRaises(ValueError):
            self.read()
        self.assertEqual(foreign.read_bytes(), self.lines[0])

    def test_native_termination_rules(self):
        for invalid in ([], [49], [True, 50], [50, 1, 50], [262144, 50], [24]*513):
            with self.assertRaises(ValueError):
                workers.validate_ids(invalid)
        workers.validate_ids([26]*512)
        workers.validate_ids([26]*511 + [50])


class ReplicatedGenerationTests(unittest.TestCase):
    def test_schedules_actual_prefixes_recovers_only_unfinished_work_and_authenticates_reuse(self):
        with TemporaryDirectory(dir=workers.ROOT / ".runtime/tmp") as tmp:
            root = Path(tmp).resolve()
            output = root / "runs/new-train-fit-v1"
            tokenizer = root / "tokenizer.json"
            tokenizer.write_bytes(b"fixture")
            record = {"path": tokenizer.name, "sha256": sha256(b"fixture").hexdigest(), "bytes": 7}
            budget = SimpleNamespace(output=output, writing=lambda n: nullcontext())
            prompts = {f"case-{i}": array("I", [4, i+1000, 8]) for i in range(640)}
            requests = []
            arms_started = []
            binding = "a"*64
            fake_decoder = Mock()
            fake_decoder.decode.side_effect = lambda ids, skip_special_tokens: f"native:{ids}"
            module = SimpleNamespace(Tokenizer=SimpleNamespace(from_file=lambda path: fake_decoder))

            @contextmanager
            def replicas(gpus, spec, lease):
                self.assertEqual(lease, 27)
                self.assertEqual(spec, dict(checkpoint=record, base=record, tokenizer=record))
                arms_started.append(list(gpus))
                yield {gpu: gpu for gpu in gpus}

            def exchange(gpu, message):
                cid, prefix = message
                index = int(cid.removeprefix("case-"))
                self.assertEqual(prefix, [4, index+1000, 8])
                self.assertEqual(gpu, index % 4)
                requests.append(cid)
                return [cid, [index+1000, 50]]

            def generate():
                return workers.generate_arm(output, "ordinary-1729", binding, record, record, record,
                                            prompts, 27, budget)

            with (patch.object(workers, "ROOT", root), patch.object(workers, "replicas", replicas),
                  patch.object(workers, "exchange", exchange), patch.dict("sys.modules", {"tokenizers": module}),
                  redirect_stdout(StringIO())):
                generations = generate()
                self.assertEqual(arms_started, [[0, 1, 2, 3]])
                self.assertEqual(Counter(requests), Counter(prompts.keys()))
                self.assertEqual(set(prompts), set(generations))
                for cid, value in generations.items():
                    ids = [prompts[cid][1], 50]
                    self.assertEqual(value["raw_ids"], ids)
                    self.assertEqual(value["raw_text"], f"native:{ids}")
                    self.assertEqual((value["input_tokens"], value["complete"], value["finish_reason"]),
                                     (3, True, "complete_batch"))
                folder = output / "ordinary-1729"
                self.assertEqual(set(p.name for p in folder.iterdir()), {f"gpu-{i}.jsonl" for i in range(4)})
                shard = folder / "gpu-1.jsonl"
                lines = shard.read_bytes().splitlines(keepends=True)
                self.assertEqual(len(lines), 160)
                self.assertEqual(set(json.loads(lines[0])),
                                 {"binding", "case_id", "prompt_ids_sha256", "raw_ids", "sha256"})
                shard.write_bytes(b"".join(lines[:-2])+lines[-2][:80])
                requests.clear()
                self.assertEqual(generate(), generations)
                self.assertEqual(requests, ["case-633", "case-637"])
                self.assertEqual(arms_started[-1], [1])
                self.assertEqual(shard.read_bytes(), b"".join(lines))
                with (patch.object(workers, "replicas", side_effect=AssertionError("GPU on cached run")),
                      patch.object(budget, "writing", side_effect=AssertionError("write on cached run"))):
                    self.assertEqual(generate(), generations)
                    prompts["case-1"] = array("I", [4, 999, 8])
                    with self.assertRaisesRegex(ValueError, "binding or checksum"):
                        generate()


def fixture_child():
    import run
    root = Path(os.environ["FIT_TEST_ROOT"])
    workers.ROOT, workers.RUNS = root, root / "runs"
    gpu, lease, liveness = map(int, sys.argv[1:])
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 4, set_device=lambda i: None,
                                 get_device_properties=lambda i: SimpleNamespace(uuid=f"fixture-{i}")))
    runtime = Mock(step=8151, tokenizer_sha256=sha256(b"fixture").hexdigest())
    def generate(ids, *, max_new_tokens, stop_ids):
        assert max_new_tokens == 512 and stop_ids == {50}
        return [ids[-1] + 100, 50]
    runtime.generate = generate
    model = SimpleNamespace(Runtime=lambda base, checkpoint, *, device, expected_batch_size: runtime)
    with (patch.dict(sys.modules, {"model": model, "torch": fake_torch}),
          patch.object(run, "configure_parallel_numerics")):
        workers.child(gpu, lease, liveness)


class WorkerProcessTests(unittest.TestCase):
    def test_four_child_processes_inherit_the_same_lease_and_serve_only_assigned_prompts(self):
        from ledger import execution_lease
        with TemporaryDirectory(dir=workers.ROOT / ".runtime/tmp") as tmp:
            root = Path(tmp).resolve()
            runs = root / "runs"
            weight = root / "fixture.json"
            weight.write_bytes(b"fixture")
            record = {"path": weight.name, "bytes": 7, "sha256": sha256(b"fixture").hexdigest()}
            with (patch.object(workers, "ROOT", root), patch.object(workers, "RUNS", runs),
                  patch.object(workers, "__file__", __file__), patch.dict(os.environ, {"FIT_TEST_ROOT": str(root)}),
                  execution_lease(runs) as lease):
                spec = {"base": record, "checkpoint": record, "tokenizer": record}
                with workers.replicas(range(4), spec, lease) as processes:
                    self.assertEqual(len({p.pid for p in processes.values()}), 4)
                    with self.assertRaises(RuntimeError), execution_lease(runs):
                        self.fail("A second coordinator acquired the shared lease")
                    for gpu, process in processes.items():
                        self.assertEqual(workers.exchange(process, [f"id-{gpu}", [101, gpu]]),
                                         [f"id-{gpu}", [gpu+100, 50]])
                self.assertTrue(all(p.poll() == 0 for p in processes.values()))


class BudgetTests(unittest.TestCase):
    def test_old_artifacts_count_new_hardlinks_once_and_respect_physical_reserve(self):
        import diagnose
        with TemporaryDirectory(dir=workers.ROOT / ".runtime/tmp") as tmp:
            root = Path(tmp).resolve()
            runs = root / "runs"
            previous = runs / "train-fit-reference-v1"
            transfer = runs / "semantic-transfer-v1"
            output = runs / "new-train-fit-v1"
            for directory in (previous, transfer, output):
                directory.mkdir(parents=True)
            budget = object.__new__(workers.FitBudget)
            budget.previous = SimpleNamespace(output=previous, usage=lambda: {"charged_bytes": diagnose.DISK_LIMIT-200000})
            budget.output = output
            with patch.object(workers, "ROOT", root), patch.object(workers, "RUNS", runs):
                with patch.object(workers.shutil, "disk_usage", return_value=SimpleNamespace(free=10**12)):
                    before = budget.require(1)
                    saved = transfer / "saved-file"
                    saved.write_bytes(b"x"*48000)
                    os.link(saved, output / "second-name")
                    after = budget.require(1)
                    self.assertGreaterEqual(before - after, 48000)
                    self.assertLess(before - after, 60000)
                    with self.assertRaisesRegex(RuntimeError, "quota/reserve"):
                        budget.require(after)
                with (patch.object(workers.shutil, "disk_usage", return_value=SimpleNamespace(free=diagnose.RESERVE+5)),
                      self.assertRaisesRegex(RuntimeError, "quota/reserve")):
                    budget.require(1)
                self.assertEqual(saved.stat().st_size, 48000)


if __name__ == "__main__":
    if os.environ.get("FIT_TEST_ROOT") and len(sys.argv) == 4:
        fixture_child()
    else:
        unittest.main()
