"""CPU-only interruption and decision tests; no model is loaded."""

from copy import deepcopy
import errno
import importlib.util
import json
import io
import os
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
import sys
import struct
import subprocess
import tempfile
from types import ModuleType, SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))
import history as h
import history_parity as parity
import history_prepare as prepare
import private_runtime
import ledger


class PublicationLifetimeTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.root = Path(self.stack.enter_context(TemporaryDirectory()))
        self.destination = self.root / 'result.json'
        self.streams, self.events = {}, []
        self.write_error = self.unlink_error = None
        self.real_named = ledger.NamedTemporaryFile
        self.real_link, self.real_unlink = ledger.os.link, Path.unlink
        self.stack.enter_context(patch.object(ledger, 'NamedTemporaryFile', self.tracked_file))
        self.stack.enter_context(patch.object(ledger.os, 'link', self.link_closed))
        self.stack.enter_context(patch.object(Path, 'unlink', lambda path, *args, **kwargs:
                                               self.unlink_closed(path, *args, **kwargs)))

    def tearDown(self):
        self.stack.close()

    def tracked_file(self, *args, **kwargs):
        owner, stream = self, self.real_named(*args, **kwargs)
        self.streams[Path(stream.name)] = stream

        class Stream:
            def __enter__(self):
                stream.__enter__()
                return self

            def __exit__(self, *error):
                return stream.__exit__(*error)

            def __getattr__(self, name):
                return getattr(stream, name)

            def write(self, data):
                if owner.write_error is not None:
                    raise owner.write_error
                return stream.write(data)

        return Stream()

    def link_closed(self, source, destination, *args, **kwargs):
        source = Path(source)
        if source in self.streams:
            self.assertTrue(self.streams[source].closed, 'Publication linked an open temporary stream')
            self.events.append('link')
        return self.real_link(source, destination, *args, **kwargs)

    def unlink_closed(self, path, *args, **kwargs):
        path = Path(path)
        if path in self.streams:
            if not self.streams[path].closed:
                raise OSError(errno.ENOTEMPTY, 'Simulated NFS open-file cleanup failure', str(path))
            self.events.append('unlink')
            if self.unlink_error is not None:
                raise self.unlink_error
        return self.real_unlink(path, *args, **kwargs)

    def assert_closed_and_only(self, *names):
        self.assertTrue(all(stream.closed for stream in self.streams.values()))
        self.assertEqual(sorted(path.name for path in self.root.iterdir()), sorted(names))

    def test_atomic_publication_closes_before_link_and_unlink(self):
        ledger.publish_json(self.destination, {'status': 'ready'})
        self.assertEqual(json.loads(self.destination.read_text()), {'status': 'ready'})
        self.assertEqual(self.events, ['link', 'unlink'])
        self.assert_closed_and_only('result.json')

    def test_write_failure_closes_and_removes_only_own_temporary(self):
        (self.root / 'existing.txt').write_text('keep')
        failure = RuntimeError('write failed')
        self.write_error = failure
        with self.assertRaises(RuntimeError) as caught:
            ledger.publish_json(self.destination, {'status': 'ready'})
        self.assertIs(caught.exception, failure)
        self.assertEqual(self.events, ['unlink'])
        self.assert_closed_and_only('existing.txt')
        self.assertEqual((self.root / 'existing.txt').read_text(), 'keep')

    def test_existing_destination_is_never_overwritten(self):
        before = b'{"status":"original"}\n'
        self.destination.write_bytes(before)
        with self.assertRaises(FileExistsError):
            ledger.publish_json(self.destination, {'status': 'replacement'})
        self.assertEqual(self.destination.read_bytes(), before)
        self.assertEqual(self.events, ['link', 'unlink'])
        self.assert_closed_and_only('result.json')

    def test_file_sync_failure_closes_before_cleanup(self):
        failure = OSError(errno.EIO, 'file sync failed')
        with patch.object(ledger.os, 'fsync', side_effect=failure), self.assertRaises(OSError) as caught:
            ledger.publish_json(self.destination, {'status': 'ready'})
        self.assertIs(caught.exception, failure)
        self.assertEqual(self.events, ['unlink'])
        self.assert_closed_and_only()

    def test_directory_sync_failure_preserves_published_destination(self):
        failure = OSError(errno.EIO, 'directory sync failed')
        with patch.object(ledger, '_sync_directory', side_effect=failure), self.assertRaises(OSError) as caught:
            ledger.publish_json(self.destination, {'status': 'ready'})
        self.assertIs(caught.exception, failure)
        self.assertEqual(json.loads(self.destination.read_text()), {'status': 'ready'})
        self.assertEqual(self.events, ['link', 'unlink'])
        self.assert_closed_and_only('result.json')

    def test_cleanup_failure_is_reported_after_successful_publication(self):
        failure = OSError(errno.ENOTEMPTY, 'cleanup failed')
        self.unlink_error = failure
        with self.assertRaises(OSError) as caught:
            ledger.publish_json(self.destination, {'status': 'ready'})
        self.assertIs(caught.exception, failure)
        self.assertTrue(all(stream.closed for stream in self.streams.values()))
        self.assertEqual(json.loads(self.destination.read_text()), {'status': 'ready'})
        self.assertEqual(self.events, ['link', 'unlink'])

    def test_cleanup_error_does_not_mask_original_write_error(self):
        failure = RuntimeError('original write failure')
        self.write_error = failure
        self.unlink_error = OSError(errno.ENOTEMPTY, 'secondary cleanup failure')
        with self.assertRaises(RuntimeError) as caught:
            ledger.publish_json(self.destination, {'status': 'ready'})
        self.assertIs(caught.exception, failure)
        self.assertTrue(any('cleanup' in note for note in caught.exception.__notes__))
        self.assertTrue(all(stream.closed for stream in self.streams.values()))
        self.assertFalse(self.destination.exists())

    def test_creation_failure_does_not_remove_existing_files(self):
        self.destination.write_text('preserve')
        failure = OSError(errno.ENOSPC, 'cannot create staging file')
        with patch.object(ledger, 'NamedTemporaryFile', side_effect=failure), self.assertRaises(OSError) as caught:
            ledger.publish_json(self.destination, {'status': 'ready'})
        self.assertIs(caught.exception, failure)
        self.assertEqual(self.events, [])
        self.assert_closed_and_only('result.json')
        self.assertEqual(self.destination.read_text(), 'preserve')



class PrivateRuntimeTests(unittest.TestCase):
    def test_tmux_socket_is_short_and_cannot_be_redirected(self):
        cluster_root = Path('/ceph/mri.meduniwien.ac.at/departments/physics/fmrilab/home/vsavin/for-mu-cluster')
        locations = {'root': str(cluster_root / '.runtime')}
        command = private_runtime.private_command(['tmux', 'new', '-s', 'example'], locations)
        self.assertEqual(command[1:3], ['-S', str(cluster_root / '.runtime/tmux')])
        self.assertLess(len(os.fsencode(command[2])), 104)
        with self.assertRaisesRegex(ValueError, 'fixes the tmux socket'):
            private_runtime.private_command(['tmux', '-S', '/outside/socket', 'new'], locations)

    def test_overrides_inherited_paths_and_memoized_tempdir(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            outside = root / 'outside'
            outside.mkdir()
            project = root / 'project'
            project.mkdir()
            names = [*private_runtime.CACHE_PATHS, 'TMPDIR', 'TMP', 'TEMP', 'PYTHONPYCACHEPREFIX']
            with patch.dict(os.environ, {name: str(outside) for name in names}), \
                    patch.object(tempfile, 'tempdir', str(outside)), \
                    patch.object(sys, 'dont_write_bytecode', False), patch.object(sys, 'pycache_prefix', None):
                locations = private_runtime.configure_private_runtime(project)
                for name in names:
                    self.assertTrue(Path(os.environ[name]).is_relative_to(project / '.runtime'), name)
                self.assertEqual(tempfile.gettempdir(), locations['temporary'])
                self.assertTrue(sys.dont_write_bytecode)
                with TemporaryDirectory() as created:
                    self.assertTrue(Path(created).is_relative_to(project / '.runtime/tmp'))
                    (Path(created) / 'owned.json').write_text('{}')
                self.assertFalse(Path(created).exists())
            self.assertEqual(list(outside.iterdir()), [])

    def test_rejects_symlinked_scratch_and_nested_cache(self):
        for relative in ('.runtime', '.runtime/cache/huggingface'):
            with self.subTest(relative=relative), TemporaryDirectory() as temporary:
                root = Path(temporary)
                project, outside = root / 'project', root / 'outside'
                project.mkdir()
                outside.mkdir()
                link = project / relative
                link.parent.mkdir(parents=True, exist_ok=True)
                link.symlink_to(outside, target_is_directory=True)
                with self.assertRaisesRegex(ValueError, 'real directory'):
                    private_runtime.configure_private_runtime(project)
                self.assertEqual(list(outside.iterdir()), [])

    def test_launcher_child_inherits_private_temporary_paths(self):
        probe = ('import json,os,tempfile; from pathlib import Path; '
                 'd=tempfile.mkdtemp(); p=Path(d)/"probe"; p.write_text("owned"); '
                 'p.unlink(); os.rmdir(d); '
                 'print(json.dumps({"directory":d,"cache":os.environ["PIP_CACHE_DIR"],'
                 '"bytecode":os.environ["PYTHONDONTWRITEBYTECODE"]}))')
        with TemporaryDirectory() as foreign:
            root = Path(foreign)
            binary = root / 'bin'
            binary.mkdir()
            (binary / 'python').symlink_to(sys.executable)
            startup = root / 'startup.json'
            (root / 'sitecustomize.py').write_text(
                'import json,os; from pathlib import Path; '
                f'Path({str(startup)!r}).write_text(json.dumps(dict(os.environ)))')
            result = subprocess.run([str(REPO / 'cluster'), sys.executable, '-c', probe], cwd=foreign,
                                    env={**os.environ, 'TMPDIR': foreign, 'PIP_CACHE_DIR': foreign,
                                         'PYTHONPATH': foreign, 'PATH': str(binary) + os.pathsep + os.environ['PATH']},
                                    check=True, capture_output=True, text=True)
            initial = json.loads(startup.read_text())
            self.assertEqual(initial['TMPDIR'], str(REPO / '.runtime/tmp'))
            self.assertEqual(initial.get('HOME'), os.environ.get('HOME'))
        observed = json.loads(result.stdout)
        self.assertTrue(Path(observed['directory']).is_relative_to(REPO / '.runtime/tmp'))
        self.assertEqual(observed['cache'], str(REPO / '.runtime/cache/pip'))
        self.assertEqual(observed['bytecode'], '1')
        self.assertFalse(Path(observed['directory']).exists())


class DiagnosticDispatchTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.root = Path(self.stack.enter_context(TemporaryDirectory()))
        self.runs = self.root / "runs"
        self.directory = self.runs / "blocked"
        (self.directory / "parity").mkdir(parents=True)
        self.initial = self.root / "checkpoint.safetensors"
        self.initial.write_bytes(b"fake-7205")
        self.assets = {}
        for name in ("base", "tokenizer", "history_replay", "history_template"):
            path = self.root / name
            path.write_text(name)
            self.assets[name] = (path, h.file_record(path, relative_to=self.root))
        self.config, self.original = {"id": "fixture"}, {"assets": "fixture"}
        self.origin = {"step": 7205, "checkpoint": h.file_record(self.initial, relative_to=self.root)}
        code = []
        for name in ("history_parity.py", "model.py", "private_runtime.py", "cluster"):
            path = self.root / name
            path.write_text(name)
            code.append(h.file_record(path, relative_to=self.root))
        self.manifest = {"config": self.config, "original": self.original, "origin": self.origin,
                         "assets": deepcopy({key: value[1] for key, value in self.assets.items()}),
                         "runtime": {"gpu": "fake"}, "code": {"files": code}}
        self.manifest["assets"]["initial_checkpoint"] = h.file_record(self.initial, relative_to=self.root)
        self.write("experiment.json", self.manifest)
        self.account = {"identity_sha256": h.digest(self.manifest), "maximum_updates": h.MAX_UPDATES,
                        "charged_updates": 0, "maximum_seconds": h.MAX_SECONDS, "elapsed_seconds": 25.0,
                        "inflight": None, "stages": [{"name": "official-parity", "seconds": 25.0,
                                                       "interrupted": True}]}
        self.write("budget.json", self.account)
        binding = {name: self.manifest["assets"][key] for name, key in (
            ("checkpoint", "initial_checkpoint"), ("base", "base"), ("tokenizer", "tokenizer"),
            ("probes", "history_replay"), ("native_template", "history_template"))}
        binding.update(implementation=code[0], runtime=code[1], reference_version=parity.REFERENCE_VERSION,
                       tolerances=parity.TOLERANCES)
        self.failure = {"passed": False, "binding": binding, "error": "AssertionError", "message": "boundary-513"}
        self.write("parity/parity-failed.json", self.failure)
        self.args = SimpleNamespace(runs=self.runs, run_id="blocked", source_run_id="completed")
        self.runner = SimpleNamespace(four_gpu_identity=lambda: self.manifest["runtime"],
                                      require_run_space=lambda *args: None)
        self.calls = []
        self.status = "DIAGNOSTIC_COMPLETE"
        self.stack.enter_context(patch.object(h, "ROOT", self.root))
        real_contained = h.contained
        self.stack.enter_context(patch.object(h, "contained", lambda path, parent=None:
                                              real_contained(path, parent or self.root)))
        self.stack.enter_context(patch.object(h, "load", lambda runner:
                                              (deepcopy(self.config), deepcopy(self.original), dict(self.assets))))
        self.stack.enter_context(patch.object(h, "source_checkpoint", lambda *args: (self.initial, self.origin)))
        self.stack.enter_context(patch.object(h, "code_record", lambda root, paths:
                                              {"files": [h.file_record(root / p, relative_to=root) for p in paths]}))
        self.stack.enter_context(patch.object(parity, "diagnose", self.diagnose, create=True))
        self.stack.enter_context(redirect_stdout(io.StringIO()))

    def tearDown(self):
        self.stack.close()

    def write(self, name, value):
        (self.directory / name).write_text(json.dumps(value))

    def diagnose(self, assets, checkpoint, destination, deadline, budget):
        self.calls.append(destination)
        self.assertEqual(checkpoint, self.initial)
        self.assertEqual(budget.account["record"]["inflight"]["name"], "diagnose-parity")
        with self.assertRaises(RuntimeError):
            with h.execution_lease(self.runs):
                pass
        return {"status": self.status}

    def test_success_preserves_failed_evidence_and_budget(self):
        preserved = {name: (self.directory / name).read_bytes()
                     for name in ("experiment.json", "parity/parity-failed.json")}
        h.diagnose_parity(self.args, self.runner)
        account = h.read(self.directory / "budget.json")
        self.assertEqual(account["identity_sha256"], self.account["identity_sha256"])
        self.assertEqual(account["charged_updates"], 0)
        self.assertGreaterEqual(account["elapsed_seconds"], 25)
        self.assertEqual([p["name"] for p in account["stages"]], ["official-parity", "diagnose-parity"])
        self.assertIsNone(account["inflight"])
        self.assertEqual(h.read(self.calls[0] / "budget-after.json"), account)
        self.assertFalse(h.read(self.calls[0] / "context.json")["training_allowed"])
        for name, before in preserved.items():
            self.assertEqual((self.directory / name).read_bytes(), before)

    def test_repeated_diagnostic_has_unique_directory_and_accumulates_time(self):
        h.diagnose_parity(self.args, self.runner)
        first = h.read(self.directory / "budget.json")["elapsed_seconds"]
        h.diagnose_parity(self.args, self.runner)
        self.assertNotEqual(self.calls[0], self.calls[1])
        account = h.read(self.directory / "budget.json")
        self.assertGreaterEqual(account["elapsed_seconds"], first)
        self.assertEqual(len(account["stages"]), 3)

    def test_incomplete_is_charged_and_not_promoted(self):
        self.status = "DIAGNOSTIC_INCOMPLETE"
        with self.assertRaisesRegex(RuntimeError, "remains blocked"):
            h.diagnose_parity(self.args, self.runner)
        self.assertEqual(h.read(self.directory / "budget.json")["stages"][-1]["name"], "diagnose-parity")
        self.assertFalse((self.directory / "result.json").exists())
        self.assertFalse((self.directory / "parity/parity.json").exists())

    def test_changed_old_failure_binding_rejected_before_budget_mutation(self):
        self.failure["binding"]["runtime"] = {"path": "model.py", "sha256": "changed", "bytes": 0}
        self.write("parity/parity-failed.json", self.failure)
        self.assert_rejected("frozen input and code")

    def test_missing_failure_cannot_initialize_new_experiment(self):
        (self.directory / "parity/parity-failed.json").unlink()
        self.assert_rejected("boundary-513")

    def test_budget_identity_change_rejected(self):
        self.account["identity_sha256"] = "wrong"
        self.write("budget.json", self.account)
        self.assert_rejected("does not match")

    def test_consumed_training_updates_rejected(self):
        self.account["charged_updates"] = 1
        self.write("budget.json", self.account)
        self.assert_rejected("pre-training")

    def test_later_stage_rejected(self):
        (self.directory / "collection").mkdir()
        self.assert_rejected("Later experiment stages")

    def test_revised_tolerance_rejected(self):
        self.failure["binding"]["tolerances"] = {}
        self.write("parity/parity-failed.json", self.failure)
        self.assert_rejected("original reference version and parity tolerances")

    def test_changed_assets_rejected(self):
        self.manifest["assets"]["base"]["sha256"] = "changed"
        self.write("experiment.json", self.manifest)
        self.account["identity_sha256"] = h.digest(self.manifest)
        self.write("budget.json", self.account)
        self.assert_rejected("checkpoint, data, or protocol")

    def assert_rejected(self, message):
        before = (self.directory / "budget.json").read_bytes()
        with self.assertRaisesRegex(ValueError, message):
            h.diagnose_parity(self.args, self.runner)
        self.assertEqual(self.calls, [])
        self.assertEqual((self.directory / "budget.json").read_bytes(), before)


class ReferenceRevisionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = DiagnosticDispatchTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        f = self.fixture
        f.stack.enter_context(patch.object(h, "REFERENCE_REVISION", "math-reference-v1"))
        f.args.hours, f.args.repair_parity = None, True
        f.runner.BudgetExhausted = type("FakeBudgetExhausted", (RuntimeError,), {})
        for name in ("registry", "history_dev"):
            path = f.root / name
            path.write_text(name)
            record = h.file_record(path, relative_to=f.root)
            f.assets[name] = (path, record)
            f.manifest["assets"][name] = deepcopy(record)
        f.config["evaluation"] = {}
        f.write("experiment.json", f.manifest)
        f.account["identity_sha256"] = h.digest(f.manifest)
        before = deepcopy(f.account)
        f.account["elapsed_seconds"] += 10
        f.account["stages"].append({"name": "diagnose-parity", "seconds": 10, "interrupted": False})
        f.write("budget.json", f.account)
        self.diagnostics = f.directory / "diagnostics" / "parity-123"
        self.diagnostics.mkdir(parents=True)
        context = {"experiment_identity_sha256": h.digest(f.manifest),
                   "original_manifest": h.file_record(f.directory / "experiment.json", relative_to=f.root),
                   "original_failures": [h.file_record(f.directory / "parity/parity-failed.json", relative_to=f.root)],
                   "runtime": f.manifest["runtime"], "training_allowed": False,
                   "diagnostic_code": f.manifest["code"], "budget_before": before}
        # The real hardware identity includes these fields before any model load.
        f.manifest["runtime"].update(python="3.13", torch="2.7.1+cu118", cuda="11.8")
        f.write("experiment.json", f.manifest)
        identity = h.digest(f.manifest)
        for value in (f.account, before):
            value["identity_sha256"] = identity
        f.write("budget.json", f.account)
        context.update(experiment_identity_sha256=identity,
                       original_manifest=h.file_record(f.directory / "experiment.json", relative_to=f.root))
        h.publish_json(self.diagnostics / "context.json", context)
        h.publish_json(self.diagnostics / "budget-after.json", f.account)
        code = {row["path"]: row for row in f.manifest["code"]["files"]}
        binding = {name: f.manifest["assets"][key] for name, key in (
            ("checkpoint", "initial_checkpoint"), ("base", "base"), ("probes", "history_replay"))}
        binding.update(implementation=code["history_parity.py"], runtime=code["model.py"])
        self.report = {"diagnostic_only": True, "training_performed": False,
                       "status": "DIAGNOSTIC_INCOMPLETE", "error": "AttributeError",
                       "message": "'tuple' object has no attribute 'detach'", "binding": binding,
                       "environment": dict(f.manifest["runtime"], transformers=parity.REFERENCE_VERSION),
                       "tolerances": parity.TOLERANCES, "boundaries": []}
        for length in (511, 512, 513, 514):
            good = {"within_tolerance": True, "nonfinite_count": 0, "mismatch_count": 0,
                    "maximum_absolute_error": 0.0001, "argmax_agreement": True, **parity.TOLERANCES["logits"]}
            default = dict(good) if length != 513 else dict(good, within_tolerance=False,
                mismatch_count=262010, maximum_absolute_error=24.6, argmax_agreement=False)
            self.report["boundaries"].append({"length": length,
                "mask": {"exact_equal": True, "mismatch_count": 0, "last_query_key_count": min(length, 512)},
                "custom_vs_math": good, "custom_vs_default": default, "default_vs_math": dict(default)})
        self.report_path = self.diagnostics / "diagnostic-124.json"
        h.publish_json(self.report_path, self.report)
        self.current = deepcopy(f.manifest)
        self.parent_bytes = (f.directory / "experiment.json").read_bytes()
        self.failure_bytes = (f.directory / "parity/parity-failed.json").read_bytes()

    def revision(self):
        f = self.fixture
        return h.reference_revision(f.args, f.runner, self.current, parity)

    def rewrite_report(self):
        h.replace_json(self.report_path, self.report)

    def test_partial_diagnostic_admits_only_full_gate_revision(self):
        f = self.fixture
        directory, manifest, budget_path, binding, hours, job = self.revision()
        self.assertEqual(directory, f.directory / "revisions/math-reference-v1")
        self.assertEqual(manifest["revision"]["reference_policy"], "math-chunked-v1")
        self.assertEqual(budget_path, f.directory / "budget.json")
        self.assertEqual(binding, h.digest(f.manifest))
        self.assertEqual(hours, 36)
        self.assertEqual(job.run_id, "blocked-math-reference-v1")
        self.assertFalse((directory / "budget.json").exists())
        self.assertFalse((directory / "parity/parity.json").exists())
        self.assertEqual((f.directory / "experiment.json").read_bytes(), self.parent_bytes)
        self.assertEqual((f.directory / "parity/parity-failed.json").read_bytes(), self.failure_bytes)

    def test_resume_retains_spent_work_after_training_has_started(self):
        initial = self.revision()
        budget = h.Budget(initial[2], initial[3], initial[4])
        budget.claim_updates(12)
        with budget.phase("math-reference-v1/pilot"):
            pass
        resumed = self.revision()
        self.assertEqual(resumed[:5], initial[:5])
        self.assertEqual(h.read(resumed[2])["charged_updates"], 12)
        self.assertGreaterEqual(h.read(resumed[2])["elapsed_seconds"], 35)

    def test_existing_revision_rejects_new_code(self):
        self.revision()
        self.current["code"]["files"][0]["sha256"] = "new-code"
        with self.assertRaisesRegex(ValueError, "Frozen record changed"):
            self.revision()

    def test_hours_cannot_reset_even_to_same_value(self):
        self.fixture.args.hours = 36
        with self.assertRaisesRegex(ValueError, "--hours"):
            self.revision()

    def test_new_revision_rejects_already_trained_parent(self):
        f = self.fixture
        f.account["charged_updates"] = 1
        f.write("budget.json", f.account)
        with self.assertRaisesRegex(ValueError, "pre-training"):
            self.revision()

    def test_rejects_missing_boundary(self):
        self.report["boundaries"].pop()
        self.rewrite_report()
        with self.assertRaisesRegex(ValueError, "four-boundary"):
            self.revision()

    def test_rejects_math_mismatch(self):
        self.report["boundaries"][2]["custom_vs_math"].update(within_tolerance=False, mismatch_count=1)
        self.rewrite_report()
        with self.assertRaisesRegex(ValueError, "four-boundary"):
            self.revision()

    def test_rejects_nonfinite_math(self):
        self.report["boundaries"][2]["custom_vs_math"]["nonfinite_count"] = 1
        self.rewrite_report()
        with self.assertRaisesRegex(ValueError, "four-boundary"):
            self.revision()

    def test_rejects_unbound_diagnostic_source(self):
        self.report["binding"]["checkpoint"] = {"path": "other", "sha256": "other", "bytes": 0}
        self.rewrite_report()
        with self.assertRaisesRegex(ValueError, "four-boundary"):
            self.revision()

    def test_rejects_changed_original_failure_after_creation(self):
        self.revision()
        f = self.fixture
        f.failure["message"] = "different"
        f.write("parity/parity-failed.json", f.failure)
        with self.assertRaisesRegex(ValueError, "repair evidence changed"):
            self.revision()

    def test_rejects_budget_rollback_after_creation(self):
        self.revision()
        f = self.fixture
        f.account["elapsed_seconds"] = 0
        f.write("budget.json", f.account)
        with self.assertRaisesRegex(ValueError, "rolled back"):
            self.revision()

    def test_rejects_changed_budget_stage_history(self):
        self.revision()
        f = self.fixture
        f.account["stages"][0]["seconds"] = 0
        f.write("budget.json", f.account)
        with self.assertRaisesRegex(ValueError, "rolled back"):
            self.revision()

    def test_rejects_removed_budget_stage_history(self):
        self.revision()
        f = self.fixture
        f.account["stages"] = []
        f.write("budget.json", f.account)
        with self.assertRaisesRegex(ValueError, "rolled back"):
            self.revision()

    def test_rejects_changed_assets_or_hardware(self):
        self.current["runtime"]["gpu"] = "different"
        with self.assertRaisesRegex(ValueError, "original runtime"):
            self.revision()

    def test_strict_gate_failure_blocks_collection_and_charges_old_budget(self):
        f = self.fixture
        with patch.object(h, "code_record", return_value=self.current["code"]), \
             patch.object(h.subprocess, "run"), patch.object(prepare, "check"), \
             patch.object(parity, "verify", side_effect=RuntimeError("strict parity block")) as gate, \
             patch.object(prepare, "prepare") as collection:
            with self.assertRaisesRegex(RuntimeError, "strict parity block"):
                h.compare(f.args, f.runner)
        collection.assert_not_called()
        self.assertEqual(gate.call_args.kwargs, {"reference_policy": "math-chunked-v1"})
        account = h.read(f.directory / "budget.json")
        self.assertEqual(account["identity_sha256"], h.digest(f.manifest))
        self.assertEqual(account["charged_updates"], 0)
        self.assertGreaterEqual(account["elapsed_seconds"], 35)
        self.assertEqual(account["stages"][-1]["name"], "math-reference-v1/official-parity")
        self.assertTrue(account["stages"][-1]["interrupted"])
        self.assertFalse((f.directory / "revisions/math-reference-v1/collection").exists())

    def test_passing_full_gate_precedes_collection(self):
        f = self.fixture
        with patch.object(h, "code_record", return_value=self.current["code"]), \
             patch.object(h.subprocess, "run"), patch.object(prepare, "check"), \
             patch.object(parity, "verify", return_value={"passed": True}), \
             patch.object(prepare, "prepare", side_effect=RuntimeError("stop at collection")) as collection:
            with self.assertRaisesRegex(RuntimeError, "stop at collection"):
                h.compare(f.args, f.runner)
        collection.assert_called_once()
        stage = f.directory / "revisions/math-reference-v1/stages/official-parity.json"
        self.assertTrue(h.read(stage)["result"]["passed"])
        self.assertFalse((f.directory / "parity/parity.json").exists())


class RopeRevisionTests(unittest.TestCase):
    def setUp(self):
        self.fixture = ReferenceRevisionTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.fixture.tearDown)
        self.f = self.fixture.fixture
        first = self.fixture.revision()
        self.first_directory, self.first_manifest = first[:2]
        self.first_bytes = (self.first_directory / "experiment.json").read_bytes()
        budget = h.Budget(first[2], first[3], first[4])
        try:
            with budget.phase("math-reference-v1/official-parity"):
                raise RuntimeError("fixture failed strict parity")
        except RuntimeError:
            pass
        self.before_budget = h.read(first[2])
        self.first_parity = self.first_directory / "parity"
        self.first_parity.mkdir()
        old_code = {row["path"]: row for row in self.first_manifest["code"]["files"]}
        binding = {name: self.first_manifest["assets"][key] for name, key in (
            ("checkpoint", "initial_checkpoint"), ("base", "base"), ("tokenizer", "tokenizer"),
            ("probes", "history_replay"), ("native_template", "history_template"))}
        binding.update(implementation=old_code["history_parity.py"], runtime=old_code["model.py"],
                       reference_version=parity.REFERENCE_VERSION,
                       reference_policy=parity.reference_policy_record(h.REFERENCE_POLICY), tolerances=parity.TOLERANCES)
        labels = [f"boundary-{length}" for length in (17, 511, 512, 513, 1025)]
        for length in (17, 511, 512, 513, 1025):
            labels.extend(f"reference-chunked-dense-{length}-positions-{first}:{min(first + 128, length)}"
                          for first in range(0, length, 128))
            labels.extend((f"reference-chunked-dense-cache-offset-{length}",
                           f"reference-dense-cache-uncached-offset-{length}"))
        checks = [{"label": label, "within_tolerance": True, "nonfinite_count": 0, "mismatch_count": 0,
                   "argmax_agreement": True, **parity.TOLERANCES["logits"]} for label in labels]
        mismatch = {"label": "complete-prompt:train-friday-dialogue-020-turn-5", "within_tolerance": False,
                    "nonfinite_count": 0, "actual_shape": [1, 1, 262144], "expected_shape": [1, 1, 262144],
                    "mismatch_count": 15, "argmax_agreement": True, "maximum_absolute_error": .003757,
                    **parity.TOLERANCES["logits"]}
        self.failure = {"passed": False, "binding": binding, "completed_checks": checks,
                        "error": "TensorMismatch", "mismatch": mismatch}
        self.failure_path = self.first_parity / "parity-failed.json"
        h.publish_json(self.failure_path, self.failure)
        self.failure_bytes = self.failure_path.read_bytes()
        self.current = deepcopy(self.fixture.current)
        model_path = self.f.root / "model.py"
        model_path.write_text("canonical CPU reciprocal FP32 frequencies")
        self.current["code"]["files"] = [h.file_record(model_path, relative_to=self.f.root)
                                          if row["path"] == "model.py" else row
                                          for row in self.current["code"]["files"]]
        self.current["reference_policy"] = parity.reference_policy_record(h.REFERENCE_POLICY)
        self.f.stack.enter_context(patch.object(h, "REFERENCE_REVISION", "rope-alignment-v1"))

    def revision(self):
        return h.reference_revision(self.f.args, self.f.runner, self.current, parity)

    def rewrite_failure(self):
        h.replace_json(self.failure_path, self.failure)

    def test_successor_preserves_both_failed_manifests_and_original_budget(self):
        directory, manifest, path, binding, hours, job = self.revision()
        self.assertEqual(directory, self.f.directory / "revisions/rope-alignment-v1")
        self.assertEqual(path, self.f.directory / "budget.json")
        self.assertEqual(binding, h.digest(self.f.manifest))
        self.assertEqual(hours, 36)
        self.assertEqual(job.run_id, "blocked-rope-alignment-v1")
        revision = manifest["revision"]
        self.assertEqual(revision["predecessor"]["identity_sha256"], h.digest(self.first_manifest))
        self.assertEqual(revision["runtime_numerics"]["id"], "canonical-cpu-rope-reciprocal-fp32-v1")
        self.assertNotEqual(revision["runtime_numerics"]["old_model"]["sha256"],
                            revision["runtime_numerics"]["new_model"]["sha256"])
        self.assertEqual(revision["budget_at_revision"], self.before_budget)
        self.assertEqual((self.first_directory / "experiment.json").read_bytes(), self.first_bytes)
        self.assertEqual(self.failure_path.read_bytes(), self.failure_bytes)
        self.assertEqual((self.f.directory / "experiment.json").read_bytes(), self.fixture.parent_bytes)
        self.assertFalse((directory / "budget.json").exists())
        self.assertFalse((directory / "parity/parity.json").exists())

    def test_successor_resumes_after_training_without_reset(self):
        original = self.revision()
        budget = h.Budget(original[2], original[3], original[4])
        budget.claim_updates(12)
        with budget.phase("rope-alignment-v1/pilot"):
            pass
        repeated = self.revision()
        self.assertEqual(repeated[:5], original[:5])
        self.assertEqual(h.read(repeated[2])["charged_updates"], 12)
        self.assertGreaterEqual(h.read(repeated[2])["elapsed_seconds"], self.before_budget["elapsed_seconds"])

    def test_rejects_other_numerical_failure(self):
        self.failure["mismatch"]["label"] = "boundary-513"
        self.rewrite_failure()
        with self.assertRaisesRegex(ValueError, "specific finite longest-prompt"):
            self.revision()

    def test_rejects_incomplete_reference_controls(self):
        self.failure["completed_checks"].pop()
        self.rewrite_failure()
        with self.assertRaisesRegex(ValueError, "complete dense/chunked controls"):
            self.revision()

    def test_rejects_nonfinite_failure(self):
        self.failure["mismatch"]["nonfinite_count"] = 1
        self.rewrite_failure()
        with self.assertRaisesRegex(ValueError, "specific finite longest-prompt"):
            self.revision()

    def test_rejects_predecessor_binding_change(self):
        self.failure["binding"]["runtime"] = {"path": "model.py", "sha256": "other", "bytes": 0}
        self.rewrite_failure()
        with self.assertRaisesRegex(ValueError, "preserved code and inputs"):
            self.revision()

    def test_rejects_collection_or_pass_in_predecessor(self):
        path = self.first_directory / "collection"
        path.mkdir()
        with self.assertRaisesRegex(ValueError, "passed parity or started later stages"):
            self.revision()

    def test_rejects_predecessor_training_run_even_before_update(self):
        (self.f.runs / "blocked-math-reference-v1-pilot").mkdir()
        with self.assertRaisesRegex(ValueError, "training or evaluation runs"):
            self.revision()

    def test_rejects_uncharged_predecessor_failure(self):
        account = deepcopy(self.before_budget)
        account["stages"] = account["stages"][:-1]
        self.f.write("budget.json", account)
        with self.assertRaisesRegex(ValueError, "charged interrupted phase"):
            self.revision()

    def test_rejects_predecessor_budget_rollback(self):
        account = deepcopy(self.before_budget)
        account["elapsed_seconds"] = 0
        self.f.write("budget.json", account)
        with self.assertRaisesRegex(ValueError, "rolled back"):
            self.revision()

    def test_rejects_unmodified_model_retry(self):
        self.current["code"] = deepcopy(self.first_manifest["code"])
        with self.assertRaisesRegex(ValueError, "corrected runtime"):
            self.revision()

    def test_rejects_new_code_after_successor_is_frozen(self):
        self.revision()
        self.current["code"]["files"][0]["sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "Frozen record changed"):
            self.revision()

    def test_successor_full_gate_failure_blocks_collection(self):
        f = self.f
        with patch.object(h, "code_record", return_value=self.current["code"]), \
             patch.object(h.subprocess, "run"), patch.object(prepare, "check"), \
             patch.object(parity, "verify", side_effect=RuntimeError("strict parity block")) as gate, \
             patch.object(prepare, "prepare") as collection:
            with self.assertRaisesRegex(RuntimeError, "strict parity block"):
                h.compare(f.args, f.runner)
        collection.assert_not_called()
        self.assertEqual(gate.call_args.kwargs, {"reference_policy": "math-chunked-v1"})
        account = h.read(f.directory / "budget.json")
        self.assertEqual(account["identity_sha256"], h.digest(f.manifest))
        self.assertEqual(account["charged_updates"], 0)
        self.assertGreaterEqual(account["elapsed_seconds"], self.before_budget["elapsed_seconds"])
        self.assertEqual(account["stages"][-1]["name"], "rope-alignment-v1/official-parity")
        self.assertEqual(self.failure_path.read_bytes(), self.failure_bytes)

    def test_fresh_run_pins_math_reference_policy(self):
        f = self.f
        f.args.repair_parity, f.args.run_id = False, "fresh"
        with patch.object(h, "code_record", return_value=self.current["code"]), \
             patch.object(h.subprocess, "run"), patch.object(prepare, "check"), \
             patch.object(parity, "verify", side_effect=RuntimeError("strict parity block")) as gate:
            with self.assertRaisesRegex(RuntimeError, "strict parity block"):
                h.compare(f.args, f.runner)
        self.assertEqual(gate.call_args.kwargs, {"reference_policy": "math-chunked-v1"})
        manifest = h.read(f.runs / "fresh/experiment.json")
        self.assertEqual(manifest["reference_policy"], parity.reference_policy_record("math-chunked-v1"))


class CheckpointIORecoveryTests(unittest.TestCase):
    def setUp(self):
        self.fixture = RopeRevisionTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.f.tearDown)
        self.f = self.fixture.f
        ledger = self.f.root / "ledger.py"
        ledger.write_text("original atomic receipt")
        self.fixture.current["code"]["files"].append(h.file_record(ledger, relative_to=self.f.root))
        self.rope = self.fixture.revision()
        self.rope_directory, self.rope_manifest = self.rope[:2]
        self.rope_bytes = (self.rope_directory / "experiment.json").read_bytes()
        budget = h.Budget(self.rope[2], self.rope[3], self.rope[4])
        try:
            with budget.phase("rope-alignment-v1/official-parity"):
                raise OSError("fixture cleanup failed")
        except OSError:
            pass
        self.before_budget = h.read(self.rope[2])
        self.parity_directory = self.rope_directory / "parity"
        self.parity_directory.mkdir()
        self.failure_path = self.parity_directory / "parity-failed.json"
        code = {row["path"]: row for row in self.rope_manifest["code"]["files"]}
        binding = {name: self.rope_manifest["assets"][key] for name, key in (
            ("checkpoint", "initial_checkpoint"), ("base", "base"), ("tokenizer", "tokenizer"),
            ("probes", "history_replay"), ("native_template", "history_template"))}
        binding.update(implementation=code["history_parity.py"], runtime=code["model.py"],
                       reference_version=parity.REFERENCE_VERSION,
                       reference_policy=parity.reference_policy_record(h.REFERENCE_POLICY), tolerances=parity.TOLERANCES)
        self.probes = [{"case_id": "train-friday-dialogue-020-turn-5"}, {"case_id": "tn-r-be28005914909558e6"}]
        labels = h.reference_control_labels() | {f"complete-prompt:{row['case_id']}" for row in self.probes}
        labels.update(f"{kind}-{index}" for kind in ("cached-reference", "custom-cache", "reference-cache") for index in range(8))
        self.gradients = {f"gradient-{index}" for index in range(252)}
        checks = []
        for kind, entries in (("logits", labels), ("loss", {"supervised-target-loss"}), ("gradient", self.gradients)):
            checks.extend({"label": label, "within_tolerance": True, "nonfinite_count": 0,
                           "mismatch_count": 0, **parity.TOLERANCES[kind]} for label in sorted(entries))
        exact = {f"rope-{kind}-inverse-frequencies" for kind in ("local", "global")}
        exact.update(f"rope-{kind}-rotation-{offset}" for kind in ("local", "global")
                     for offset in (0, 511, 512, 513, 1025, 12743, 23024, 32765))
        checks.extend({"label": label, "exact": True} for label in sorted(exact))
        checks.extend({"label": f"native-envelope:{row['case_id']}", "exact_render": True,
                       "exact_token_ids": True} for row in self.probes)
        self.failure = {"passed": False, "error": "OSError", "binding": binding, "completed_checks": checks,
                        "message": f"[Errno 39] Directory not empty: {str(self.parity_directory / 'tmpfixture')!r}",
                        "mismatch": None}
        h.publish_json(self.failure_path, self.failure)
        self.failure_bytes = self.failure_path.read_bytes()
        self.current = deepcopy(self.fixture.current)
        (self.f.root / "model.py").write_text("canonical CPU reciprocal frequencies with owned checkpoint storage")
        ledger.write_text("close temporary receipt before link and unlink")
        self.current["code"]["files"] = [h.file_record(self.f.root / row["path"], relative_to=self.f.root)
                                          if row["path"] in ("model.py", "ledger.py") else row
                                          for row in self.current["code"]["files"]]
        self.f.stack.enter_context(patch.object(parity, "selected_probes", return_value=self.probes))
        self.f.stack.enter_context(patch.object(h, "checkpoint_header", return_value={f"lora.{name}": {} for name in self.gradients}))

    def recovery(self):
        return h.reference_revision(self.f.args, self.f.runner, self.current, parity)

    def rewrite_failure(self):
        h.replace_json(self.failure_path, self.failure)

    def test_nested_recovery_preserves_chain_and_math_identity(self):
        directory, manifest, budget_path, binding, hours, job = self.recovery()
        self.assertEqual(directory, self.rope_directory / "recovery/checkpoint-io-v1")
        self.assertEqual(budget_path, self.f.directory / "budget.json")
        self.assertEqual(binding, h.digest(self.f.manifest))
        self.assertEqual(hours, 36)
        self.assertEqual(job.run_id, "blocked-rope-alignment-v1-checkpoint-io-v1")
        revision, old = manifest["revision"], self.rope_manifest["revision"]
        self.assertEqual(revision["runtime_numerics"], old["runtime_numerics"])
        self.assertEqual(revision["budget_at_revision"], self.before_budget)
        operation = revision["io_recovery"]
        self.assertEqual(operation["parent_identity_sha256"], h.digest(self.rope_manifest))
        self.assertEqual({row["path"] for row in operation["code_changes"]}, {"model.py", "ledger.py"})
        self.assertFalse(operation["prior_checkpoint_reload_accepted"])
        self.assertTrue(operation["full_gate_required"])
        self.assertFalse((directory / "budget.json").exists())
        self.assertEqual((self.rope_directory / "experiment.json").read_bytes(), self.rope_bytes)
        self.assertEqual(self.failure_path.read_bytes(), self.failure_bytes)
        self.assertEqual((self.fixture.first_directory / "experiment.json").read_bytes(), self.fixture.first_bytes)

    def test_recovery_resumes_after_training_with_same_budget(self):
        original = self.recovery()
        budget = h.Budget(original[2], original[3], original[4])
        budget.claim_updates(8)
        with budget.phase("rope-alignment-v1/checkpoint-io-v1/pilot"):
            pass
        repeated = self.recovery()
        self.assertEqual(repeated[:5], original[:5])
        self.assertEqual(h.read(repeated[2])["charged_updates"], 8)

    def test_numerical_failure_cannot_be_hidden_by_later_io(self):
        numerical = deepcopy(self.failure)
        numerical.update(error="TensorMismatch", mismatch={"label": "some-numerical-failure"})
        h.publish_json(self.parity_directory / "parity-failed-older.json", numerical)
        with self.assertRaisesRegex(ValueError, "numerical failures cannot be bypassed"):
            self.recovery()

    def test_rejects_different_io_error(self):
        self.failure["message"] = "[Errno 5] Input/output error"
        self.rewrite_failure()
        with self.assertRaisesRegex(ValueError, "ENOTEMPTY"):
            self.recovery()

    def test_rejects_cleanup_outside_parity_scratch(self):
        self.failure["message"] = f"[Errno 39] Directory not empty: {str(self.f.root / 'tmpelsewhere')!r}"
        self.rewrite_failure()
        with self.assertRaisesRegex(ValueError, "ENOTEMPTY"):
            self.recovery()

    def test_rejects_missing_gradient_check(self):
        self.failure["completed_checks"] = [row for row in self.failure["completed_checks"] if row["label"] != "gradient-0"]
        self.rewrite_failure()
        with self.assertRaisesRegex(ValueError, "complete strict logits, loss, and gradient"):
            self.recovery()

    def test_rejects_missing_full_prompt_check(self):
        self.failure["completed_checks"] = [row for row in self.failure["completed_checks"]
                                            if row["label"] != "complete-prompt:train-friday-dialogue-020-turn-5"]
        self.rewrite_failure()
        with self.assertRaisesRegex(ValueError, "complete strict logits, loss, and gradient"):
            self.recovery()

    def test_rejects_wrong_binding(self):
        self.failure["binding"]["runtime"] = {"path": "model.py", "sha256": "wrong", "bytes": 0}
        self.rewrite_failure()
        with self.assertRaisesRegex(ValueError, "Every preserved RoPE failure"):
            self.recovery()

    def test_rejects_already_passed_gate(self):
        h.publish_json(self.parity_directory / "parity.json", {"passed": True})
        with self.assertRaisesRegex(ValueError, "already passed parity"):
            self.recovery()

    def test_rejects_training_started(self):
        value = deepcopy(self.before_budget)
        value["charged_updates"] = 1
        self.f.write("budget.json", value)
        with self.assertRaisesRegex(ValueError, "pre-training"):
            self.recovery()

    def test_rejects_unrelated_code_change(self):
        next(row for row in self.current["code"]["files"] if row["path"] == "private_runtime.py")["sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "declared I/O implementation"):
            self.recovery()

    def test_recovery_code_is_frozen(self):
        self.recovery()
        self.current["code"]["files"][0]["sha256"] = "different"
        with self.assertRaisesRegex(ValueError, "Frozen record changed"):
            self.recovery()

    def test_budget_rollback_rejected_on_recovery_resume(self):
        self.recovery()
        value = deepcopy(self.before_budget)
        value["elapsed_seconds"] = 0
        self.f.write("budget.json", value)
        with self.assertRaisesRegex(ValueError, "rolled back"):
            self.recovery()

    def test_recovery_gate_failure_still_blocks_collection(self):
        with patch.object(h, "code_record", return_value=self.current["code"]), \
             patch.object(h.subprocess, "run"), patch.object(prepare, "check"), \
             patch.object(parity, "verify", side_effect=RuntimeError("fresh strict gate failed")), \
             patch.object(prepare, "prepare") as collection:
            with self.assertRaisesRegex(RuntimeError, "fresh strict gate failed"):
                h.compare(self.f.args, self.f.runner)
        collection.assert_not_called()
        account = h.read(self.f.directory / "budget.json")
        self.assertEqual(account["charged_updates"], 0)
        self.assertEqual(account["stages"][-1]["name"], "rope-alignment-v1/checkpoint-io-v1/official-parity")
        self.assertGreaterEqual(account["elapsed_seconds"], self.before_budget["elapsed_seconds"])
        self.assertEqual(self.failure_path.read_bytes(), self.failure_bytes)

    def test_recovery_full_gate_pass_precedes_collection(self):
        with patch.object(h, "code_record", return_value=self.current["code"]), \
             patch.object(h.subprocess, "run"), patch.object(prepare, "check"), \
             patch.object(parity, "verify", return_value={"passed": True}), \
             patch.object(prepare, "prepare", side_effect=RuntimeError("stop at collection")) as collection:
            with self.assertRaisesRegex(RuntimeError, "stop at collection"):
                h.compare(self.f.args, self.f.runner)
        collection.assert_called_once()
        path = self.rope_directory / "recovery/checkpoint-io-v1/stages/official-parity.json"
        self.assertTrue(h.read(path)["result"]["passed"])
        self.assertFalse((self.parity_directory / "parity.json").exists())



class CheckpointOwnershipTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import private_runtime
        private_runtime.configure_private_runtime()
        try:
            import torch
            from safetensors import safe_open
        except ImportError as error:
            raise unittest.SkipTest('Checkpoint ownership controls require torch and safetensors') from error
        import model
        import history_parity
        cls.torch, cls.model, cls.parity = torch, model, history_parity
        cls.safe_open = staticmethod(safe_open)

    def setUp(self):
        from contextlib import ExitStack, contextmanager
        from types import MethodType, SimpleNamespace
        import weakref

        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.raw_refs, self.raw_addresses = [], {}
        self.stack.enter_context(patch.object(self.model, 'adapter_shapes', lambda: {'probe': (2, 3)}))
        self.stack.enter_context(patch.object(self.model, 'checkpoint_origin', lambda metadata: {'step': 4847}))
        owner = self

        @contextmanager
        def tracked_reader(*args, **kwargs):
            with owner.safe_open(*args, **kwargs) as source:
                class Reader:
                    def __getattr__(self, name):
                        return getattr(source, name)

                    def get_tensor(self, key):
                        value = source.get_tensor(key)
                        owner.raw_refs.append(weakref.ref(value))
                        owner.raw_addresses[key] = value.data_ptr()
                        return value
                yield Reader()

        self.stack.enter_context(patch.object(self.model, 'safe_open', tracked_reader))
        self.runtime = SimpleNamespace(
            step=7205, lineage={'fixture': 'owned-storage'}, _metadata={},
            parameters={'probe': self.torch.arange(6, dtype=self.torch.float32).reshape(2, 3).requires_grad_()},
            rows={'probe': self.torch.tensor([1., 2.])}, cols={'probe': self.torch.tensor([3., 4., 5.])})
        self.runtime.save_checkpoint = MethodType(self.model.Runtime.save_checkpoint, self.runtime)
        self.path = self.root / 'fixture.safetensors'
        self.runtime.save_checkpoint(self.path, lineage=self.runtime.lineage)

    def test_cpu_read_owns_storage_and_preserves_exact_values(self):
        before = self.path.read_bytes()
        loaded = self.model.read_checkpoint(self.path, self.torch.device('cpu'))
        self.assertEqual(loaded[2:4], (self.runtime.step, self.runtime.lineage))
        for field, prefix, table in zip(('parameters', 'rows', 'cols'), ('lora', 'adafactor.row', 'adafactor.col'), loaded[4:]):
            actual = table['probe']
            self.assertTrue(self.torch.equal(actual, getattr(self.runtime, field)['probe']))
            self.assertNotEqual(actual.data_ptr(), self.raw_addresses[f'{prefix}.probe'])
            self.assertEqual(actual.requires_grad, field == 'parameters')
        self.assertTrue(all(reference() is None for reference in self.raw_refs))
        self.assertEqual(self.path.read_bytes(), before)

    def test_validation_exception_releases_views_with_traceback_retained(self):
        with patch.object(self.model, 'tensor_digest', lambda value: 'invalid-digest'):
            try:
                self.model.read_checkpoint(self.path, self.torch.device('cpu'))
            except ValueError as error:
                retained = error
            else:
                self.fail('Invalid checkpoint tensor was accepted')
        self.assertIsNotNone(retained.__traceback__)
        self.assertIn('Invalid checkpoint tensor', str(retained))
        self.assertTrue(all(reference() is None for reference in self.raw_refs))

    def test_reload_releases_views_before_successful_cleanup(self):
        owner, paths = self, []

        class CleanupGuard:
            def __init__(self, *args, **kwargs):
                self.temporary = TemporaryDirectory(*args, **kwargs)
                self.name = self.temporary.name
                paths.append(Path(self.name))

            def cleanup(self):
                owner.assertTrue(all(reference() is None for reference in owner.raw_refs))
                self.temporary.cleanup()

        with patch.object(self.parity, 'TemporaryDirectory', CleanupGuard):
            self.parity.verify_checkpoint_reload(self.runtime, self.root)
        self.assertTrue(paths)
        self.assertTrue(all(not path.exists() for path in paths))

    def test_cleanup_error_blocks_success_without_masking_comparison_error(self):
        import errno
        original_read = self.model.read_checkpoint

        class FailingCleanup:
            def __init__(self, *args, **kwargs):
                self.temporary = TemporaryDirectory(*args, **kwargs)
                self.name = self.temporary.name

            def cleanup(self):
                self.temporary.cleanup()
                raise OSError(errno.ENOTEMPTY, 'Injected cleanup failure', self.name)

        with patch.object(self.parity, 'TemporaryDirectory', FailingCleanup):
            with self.assertRaises(OSError) as cleanup:
                self.parity.verify_checkpoint_reload(self.runtime, self.root)
            self.assertEqual(cleanup.exception.errno, errno.ENOTEMPTY)
            self.assertEqual(cleanup.exception.parity_stage, 'checkpoint-reload-cleanup')

            def wrong_cursor(*args, **kwargs):
                result = list(original_read(*args, **kwargs))
                result[2] += 1
                return tuple(result)

            with patch.object(self.model, 'read_checkpoint', wrong_cursor), self.assertRaises(AssertionError) as primary:
                self.parity.verify_checkpoint_reload(self.runtime, self.root)
            self.assertIn('cursor or lineage differs', str(primary.exception))
            self.assertEqual(primary.exception.parity_stage, 'checkpoint-reload-compare')
            self.assertEqual(primary.exception.cleanup_error['errno'], errno.ENOTEMPTY)



class Clock:
    def __init__(self):
        self.value = 1000.0
    def __call__(self):
        return self.value


def metric(passed, planned):
    return {'passed': passed, 'planned': planned}


def fixture(passed, *, unresolved=0, reference=20, whole=0, critical=None):
    summary = {'supported': metric(passed, 100), 'outside_scope': metric(10, 10),
               'whole_dialogues': metric(whole, 10), 'standalone': metric(20, 20),
               'exact_reference': metric(1, 1), 'critical_review_required': unresolved,
               'critical_types': critical or {}, 'critical_detected': sum((critical or {}).values()), 'missing': 0}
    fixed = dict(deepcopy(summary), exact_reference=metric(reference, 100), critical_review_required=0,
                 critical_types={})
    return {'score': {'summary': summary, 'panels': {'closed_loop': summary, 'fixed_prefix': fixed}}, 'seconds': 1.0}


def observations(passed):
    return {f'case-{i}': {'case_id': f'case-{i}', 'family_id': f'family-{i % 10}',
        'panel': 'closed_loop', 'cohort': 'supported', 'success': i < passed,
        'critical_status': 'none_observed'} for i in range(100)}


class FakeBackend:
    def __init__(self, root, budget, *, fail_at=None, unresolved=0):
        self.root, self.budget, self.fail_at = root, budget, fail_at
        self.calls, self.unresolved = [], unresolved
    def after(self, name, value):
        self.calls.append(name)
        if name == self.fail_at:
            self.fail_at = None
            raise OSError('simulated interruption after atomic output')
        return value
    def checkpoint(self, name, step):
        path = self.root / f'{name}-{step}.bin'
        h.immutable(path, {'name': name, 'step': step})
        return h.file_record(path, relative_to=self.root)
    def validate_artifact(self, record):
        if h.file_record(self.root / record['path'], relative_to=self.root) != record:
            raise ValueError('changed checkpoint')
    def pilot(self, deadline):
        checkpoint = self.root / 'pilot-7207.bin'
        if not checkpoint.exists():
            self.budget.claim_updates(6)
        value = {'checkpoint': self.checkpoint('pilot', 7207),
                 'measurements': [{'seconds': 1.0}, {'seconds': 1.0}]}
        return self.after('pilot', value)
    def remaining_updates(self, name):
        if (self.root / f'{name}-7991.bin').exists():
            return 0
        return 393 if (self.root / f'{name}-7598.bin').exists() else 786
    def train(self, name, milestone, deadline):
        filename = self.root / f'{name}-{7205 + milestone}.bin'
        if not filename.exists():
            done = 786 - self.remaining_updates(name)
            self.budget.claim_updates(milestone - done)
        value = {'checkpoint': self.checkpoint(name, 7205 + milestone), 'step': 7205 + milestone}
        return self.after(f'{name}-{milestone}', value)
    def evaluate(self, name, checkpoint, deadline, *, known=False):
        passed = 20 if name == 'baseline' else 30 if name.startswith('control') else 60
        value = fixture(passed, unresolved=self.unresolved if name.startswith('treatment') else 0,
                        reference=20 if name == 'baseline' else 30 if name.startswith('control') else 70,
                        whole=4 if name.startswith('treatment') else 0)
        value['checkpoint'] = h.file_record(checkpoint, relative_to=self.root)
        path = self.root / f'{name}-{known}-observations.json'
        h.immutable(path, observations(passed))
        value['observations'] = h.file_record(path, relative_to=self.root)
        return self.after(f'{name}-{"regression" if known else "final"}', value)
    def validate_evaluation(self, value, *, known=False):
        self.validate_artifact(value['checkpoint'])
        self.validate_artifact(value['observations'])
    def outcomes(self, value):
        return h.read(self.root / value['observations']['path'])
    def space(self, evaluations):
        pass


class BudgetTests(unittest.TestCase):
    def test_precharge_and_frozen_limit(self):
        with TemporaryDirectory() as directory:
            path, clock = Path(directory) / 'budget.json', Clock()
            budget = h.Budget(path, 'identity', clock=clock)
            budget.claim_updates(h.MAX_UPDATES)
            with self.assertRaises(RuntimeError):
                budget.claim_updates(1)
            with self.assertRaises(ValueError):
                h.Budget(path, 'new-identity', clock=clock)
            with self.assertRaises(ValueError):
                h.Budget(path, 'identity', hours=35, clock=clock)

    def test_interrupted_phase_is_charged_without_reset(self):
        with TemporaryDirectory() as directory:
            path, clock = Path(directory) / 'budget.json', Clock()
            budget = h.Budget(path, 'identity', clock=clock)
            record = budget.account['record']
            record['inflight'] = {'name': 'evaluation', 'started_at': clock()}
            h.replace_json(path, record)
            clock.value += 125
            resumed = h.Budget(path, 'identity', clock=clock)
            self.assertEqual(resumed.account['record']['elapsed_seconds'], 125)
            with resumed.phase('finish'):
                clock.value += 5
                resumed.claim_updates(2)
                with self.assertRaises(RuntimeError):
                    with resumed.phase('parallel'):
                        pass
            self.assertEqual(resumed.account['record']['elapsed_seconds'], 130)
            self.assertEqual(h.read(path)['charged_updates'], 2)


class GateTests(unittest.TestCase):
    def values(self):
        return fixture(20), fixture(30, reference=30), fixture(60, reference=70, whole=4)
    def test_gain_passes_and_unresolved_blocks(self):
        b, a, t = self.values()
        self.assertTrue(h.decision(b, a, t, (observations(30), observations(60)))['passed'])
        t['score']['panels']['fixed_prefix']['critical_review_required'] = 1
        result = h.decision(b, a, t, (observations(30), observations(60)))
        self.assertEqual(result['status'], 'inconclusive')
        self.assertFalse(result['passed'])
    def test_challenge_critical_blocks_and_kind_offset_cannot_hide(self):
        b, a, t = self.values()
        for value in (b, a, t):
            value['score']['panels']['challenge'] = deepcopy(value['score']['panels']['fixed_prefix'])
        a['score']['panels']['challenge']['critical_types'] = {'wrong_answer': 5}
        t['score']['panels']['challenge']['critical_types'] = {'fabrication': 1}
        result = h.decision(b, a, t, (observations(30), observations(60)))
        self.assertFalse(result['gates']['other_critical'])
    def test_denominator_or_family_mutation_fails(self):
        b, a, t = self.values()
        t['score']['summary']['supported']['planned'] = 99
        with self.assertRaises(ValueError):
            h.decision(b, a, t, (observations(30), observations(60)))
        changed = observations(60)
        changed['case-0']['family_id'] = 'different'
        with self.assertRaises(ValueError):
            h.paired_bootstrap(observations(30), changed, repeats=50)
    def test_bootstrap_samples_whole_families(self):
        result = h.paired_bootstrap(observations(30), observations(60), repeats=500)
        self.assertEqual(result['families'], 10)
        self.assertEqual(result['lower'], .3)
        self.assertEqual(result['upper'], .3)
    def test_bootstrap_joins_overlapping_families_without_splitting(self):
        left, right = observations(30), observations(60)
        for records in (left, right):
            for row in records.values():
                row['bootstrap_family_id'] = f"block-{int(row['family_id'].split('-')[1]) // 2}"
        self.assertEqual(h.paired_bootstrap(left, right, repeats=50)['families'], 5)
        for records in (left, right):
            records['case-0']['bootstrap_family_id'] = 'split-original-family'
        with self.assertRaises(ValueError):
            h.paired_bootstrap(left, right, repeats=50)
    def test_infrastructure_failure_blocks_gain_but_overflow_stays_in_denominator(self):
        rows = [{'case_id': 'overflow', 'generation': {'error': 'context_overflow'}},
                {'case_id': 'broken', 'after': {'exchanges': [{'outcome': 'provider_error', 'error_code': 'inference_failed'}]}},
                {'case_id': 'model-error', 'after': {'exchanges': [{'outcome': 'error', 'error_code': 'invalid_argument'}]}}]
        self.assertEqual(h.infrastructure_failures(rows), ['broken'])
        b, a, t = self.values()
        a['score']['infrastructure_failures'] = 1
        result = h.decision(b, a, t, (observations(30), observations(60)))
        self.assertFalse(result['gates']['infrastructure_clean'])
        self.assertEqual(result['status'], 'inconclusive')
    def test_terminal_summary_does_not_dump_raw_results(self):
        baseline = fixture(20)
        baseline['score']['raw_results'] = 'large-raw-content' * 10000
        result = {'status': 'inconclusive', 'baseline': baseline, 'pairs': [], 'selection': None}
        stream = io.StringIO()
        report = REPO / '.runtime/tmp/history-test-result.json'
        with redirect_stdout(stream):
            h.print_summary(result, report)
        self.assertLess(len(stream.getvalue()), 1000)
        self.assertNotIn('large-raw-content', stream.getvalue())
        self.assertIn(str(report), stream.getvalue())


class BridgeTests(unittest.TestCase):
    def test_resumed_timing_profile_excludes_offline_gap_and_keeps_completed_work(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / 'timings.jsonl'
            h.append_timing(path, 'load', 2.0)
            h.append_timing(path, 'generation', 5.0, case_id='first')
            h.append_timing(path, 'overhead', 1.0)
            preserved = path.read_bytes()
            with path.open('ab') as stream:
                stream.write(b'{"interrupted":')
            self.assertEqual(sum(row['seconds'] for row in h.timing_records(path, repair=True)), 8.0)
            self.assertEqual(path.read_bytes(), preserved)
            with patch.object(h, 'time', return_value=100000000.0):
                h.append_timing(path, 'load', 3.0)
                h.append_timing(path, 'generation', 6.0, case_id='last')
            self.assertEqual(sum(row['seconds'] for row in h.timing_records(path)), 17.0)

    def test_native_generate_api_and_overflow_preserve_attempts(self):
        class Input(io.StringIO):
            def close(self):
                self.saved = self.getvalue()
                super().close()
        class Process:
            def __init__(self):
                self.stdin = Input()
                self.stdout = io.StringIO('\n'.join(json.dumps(value) for value in [
                    {'kind': 'generate', 'case_id': 'fits', 'prompt': 'short'},
                    {'kind': 'generate', 'case_id': 'overflow', 'prompt': 'long'},
                    {'kind': 'result', 'score': {'observed_records': 2}},
                ]))
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
            def wait(self):
                return 0
            def kill(self):
                pass
        class Tokenizer:
            @classmethod
            def from_file(cls, path):
                return cls()
            def no_padding(self):
                pass
            def no_truncation(self):
                pass
            def encode(self, text, *, add_special_tokens):
                return SimpleNamespace(ids=[2] * (32768 if text == 'long' else 3))
            def decode(self, ids, *, skip_special_tokens):
                return '<start_function_response>'
        class Runtime:
            calls = 0
            def generate(self, prompt, *, max_new_tokens, stop_ids):
                assert len(prompt) == 3 and max_new_tokens == 512 and stop_ids == {50}
                self.calls += 1
                return [50]
        process, runtime = Process(), Runtime()
        with TemporaryDirectory() as temporary, patch.dict(sys.modules, {'tokenizers': SimpleNamespace(Tokenizer=Tokenizer)}), \
                patch.object(h.subprocess, 'Popen', return_value=process):
            score = h.evaluate_bridge(runtime, {'tokenizer': ('tokenizer', {}), 'registry': ('registry', {})},
                                      'panel', Path(temporary) / 'observations.jsonl', {}, h.time() + 60)
        self.assertEqual(runtime.calls, 1)
        replies = [json.loads(line) for line in process.stdin.saved.splitlines()]
        self.assertEqual(replies[1]['error'], 'context_overflow')
        self.assertEqual(score['observed_records'], 2)


class SourceTests(unittest.TestCase):
    def source(self, root, *, missing_optimizer=False):
        import ledger
        runs = root / 'runs'
        runs.mkdir()
        assets = {}
        for name in ('base', 'tokenizer', 'initial_checkpoint'):
            path = root / name
            path.write_text(name)
            assets[name] = (path, h.file_record(path, relative_to=root))
        identity = {'experiment_id': 'full-course', 'architecture_id': 'functiongemma270m-lora16-volta-fp32-v2',
                    'recipe': {'initial_step': 4847, 'examples_per_update': 4, 'training': {'updates': 2358}},
                    'code': {'files': [{'path': 'run.py', 'sha256': '0' * 64}]}, 'runtime': {'test': True},
                    'train': {'sha256': '0' * 64}, 'evaluation': {'sha256': '0' * 64, 'split': 'DEV-VOLUME'},
                    **{key: value[1] for key, value in assets.items()}}
        lineage = {'identity_sha256': h.digest(identity), 'initial_checkpoint_sha256': assets['initial_checkpoint'][1]['sha256']}
        course_id = 'full-train-v2-4gpu-full-course'
        with ledger.open_run(runs, course_id, identity) as run:
            checkpoint = run.path / 'checkpoint-step-7205.safetensors'
            header = {'__metadata__': {'step': '7205', 'lineage': json.dumps(lineage)}}
            for index in range(252):
                for prefix in ('lora', 'adafactor.row', 'adafactor.col'):
                    if missing_optimizer and index == 0 and prefix == 'adafactor.col':
                        continue
                    header[f'{prefix}.tensor{index}'] = {'dtype': 'F32'}
            raw = json.dumps(header).encode()
            checkpoint.write_bytes(struct.pack('<Q', len(raw)) + raw)
            observations_path = run.path / 'original-observations.jsonl'
            observations_path.write_text('{}\n' * 155)
            evaluated = run.write_result({'missing': 0}, h.file_record(checkpoint), name='result-step-7205.json',
                       step=7205, observed_records=155, observations=h.file_record(observations_path))
        outer = runs / 'full-train-v2-4gpu'
        outer.mkdir()
        h.immutable(outer / 'result.json', {'status': 'complete', 'completed_epochs': 3,
                   'epochs': [{'details': {'step': 5633}}, {'details': {'step': 6419}}, evaluated]})
        return runs, assets, checkpoint
    def test_real_source_receipt_and_optimizer_are_mandatory(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs, assets, checkpoint = self.source(root)
            with patch.object(h, 'ROOT', root):
                path, receipt = h.source_checkpoint(runs, 'full-train-v2-4gpu', assets)
                self.assertEqual(path, checkpoint)
                self.assertEqual(receipt['step'], 7205)
                checkpoint.rename(checkpoint.with_name('checkpoint-step-6419.safetensors'))
                with self.assertRaises(FileNotFoundError):
                    h.source_checkpoint(runs, 'full-train-v2-4gpu', assets)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            runs, assets, _ = self.source(root, missing_optimizer=True)
            with patch.object(h, 'ROOT', root), self.assertRaisesRegex(ValueError, 'optimizer'):
                h.source_checkpoint(runs, 'full-train-v2-4gpu', assets)


class PanelContractTests(unittest.TestCase):
    def test_real_frozen_panel_schema_and_all_planned_denominators(self):
        config = h.read(REPO / 'experiments-history.json')
        bundle_path = REPO / config['assets']['history_dev']['path']
        bundle = h.read(bundle_path)
        replay = subprocess.run(
            ['node', str(REPO / 'evaluate-actual.mjs'), str(REPO / 'assets/registry.json'),
             str(bundle_path), '--selftest'], check=True, capture_output=True, text=True, timeout=180)
        gold = json.loads(replay.stdout)
        self.assertEqual(gold['cases'], 1250)
        rows = []
        for panel in ('fixed_prefix', 'closed_loop', 'challenge'):
            for authored in bundle['panels'][panel]:
                row = {key: authored.get(key) for key in ('case_id', 'panel', 'cohort', 'family_id',
                       'bootstrap_family_id', 'reference_eligible', 'dialogue_id')}
                row.update(success=True, critical_status='none_observed', critical_category=None,
                           reference_success=True if row['reference_eligible'] else None)
                rows.append(row)
        score = {'summary': gold['panels']['closed_loop'], 'panels': gold['panels'], 'observed_records': len(rows)}
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / 'observations.jsonl'
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            h.validate_score(score, path, config)
            self.assertEqual(len(rows), 1250)
            rows[0]['family_id'] = 'changed-family'
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            with self.assertRaisesRegex(ValueError, 'metadata'):
                h.validate_score(score, path, config)


class ResumeTests(unittest.TestCase):
    def execute(self, directory, fail_at=None, unresolved=0):
        budget = h.Budget(directory / 'budget.json', 'binding')
        stages = h.Stages(directory, 'binding', budget)
        backend = FakeBackend(directory, budget, fail_at=fail_at, unresolved=unresolved)
        initial = directory / 'initial.bin'
        h.immutable(initial, {'step': 7205})
        with patch.object(h, 'ROOT', directory):
            result = h.orchestrate(backend, stages, budget, initial)
        return result, budget, backend
    def test_resume_after_every_model_phase(self):
        phases = ['pilot', 'baseline-final']
        for seed in h.SEEDS:
            for arm in ('control', 'treatment'):
                name = f'{arm}-{seed}'
                phases.extend([f'{name}-393', f'{name}-regression', f'{name}-786', f'{name}-final'])
        for phase in phases:
            with self.subTest(phase=phase), TemporaryDirectory() as temporary:
                directory = Path(temporary)
                with self.assertRaises(OSError):
                    self.execute(directory, fail_at=phase)
                preserved = {p: p.read_bytes() for p in directory.glob('*.bin')}
                result, budget, _ = self.execute(directory)
                self.assertEqual(result['status'], 'confirmed_diagnostic_gain')
                self.assertEqual(budget.account['record']['charged_updates'], 3150)
                self.assertTrue(all(path.read_bytes() == raw for path, raw in preserved.items()))
                repeated, _, backend = self.execute(directory)
                self.assertEqual(repeated, result)
                self.assertEqual(backend.calls, [])
    def test_unresolved_stops_second_seed(self):
        with TemporaryDirectory() as temporary:
            result, budget, backend = self.execute(Path(temporary), unresolved=1)
            self.assertEqual(result['status'], 'inconclusive')
            self.assertEqual(len(result['pairs']), 1)
            self.assertEqual(budget.account['record']['charged_updates'], 1578)
            self.assertFalse(any('2718' in call for call in backend.calls))
    def test_changed_saved_checkpoint_is_rejected(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.execute(directory)
            (directory / 'control-1729-7598.bin').write_text('changed')
            with self.assertRaises(ValueError):
                self.execute(directory)
    def test_source_paths_cannot_leave_repository(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(ValueError):
                h.contained(root.parent / 'other-user', root)
            (root / 'alias').symlink_to(root.parent)
            with self.assertRaises(ValueError):
                h.contained(root / 'alias' / 'outside', root)


class ParityLayerOutputTests(unittest.TestCase):
    def test_direct_and_tuple_hidden_states(self):
        hidden = SimpleNamespace(shape=(1, 513, 32), detach=lambda: None)
        self.assertIs(parity.decoder_hidden_states(hidden), hidden)
        self.assertIs(parity.decoder_hidden_states((hidden,)), hidden)
        self.assertIs(parity.decoder_hidden_states((hidden, None)), hidden)

    def test_invalid_decoder_output_is_rejected(self):
        for value in (None, (), (None,), [], {}, (object(),)):
            with self.subTest(value=value), self.assertRaises(TypeError):
                parity.decoder_hidden_states(value)

    def test_matching_but_unterminated_generation_is_rejected(self):
        for tokens in ([], [1] * 512, [50, 1]):
            with self.subTest(length=len(tokens)), self.assertRaisesRegex(AssertionError, 'function-response boundary'):
                parity.require_complete_native_call(tokens)

    def test_complete_generation_reaches_required_boundary(self):
        parity.require_complete_native_call([2, 40, 50])


class ParityRecoveryTests(unittest.TestCase):
    def test_io_retry_cannot_hide_later_numerical_failure_and_constructors_are_closed(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            assets = {}
            for name in ('base', 'tokenizer', 'history_replay', 'history_template'):
                assets[name] = root / name
                assets[name].write_text(name)
            checkpoint = root / 'checkpoint'
            checkpoint.write_text('checkpoint')
            model_path = root / 'model.py'
            model_path.write_text('fake model module')
            fake_torch = ModuleType('torch')
            fake_torch.cuda = SimpleNamespace(empty_cache=lambda: None)
            fake_nn = ModuleType('torch.nn')
            fake_nn.functional = SimpleNamespace()
            fake_model = ModuleType('model')
            fake_model.__file__ = str(model_path)
            attempts, closed = [], []
            def construct(*args, **kwargs):
                attempts.append('runtime')
                if len(attempts) == 1:
                    raise OSError('fixture transient I/O failure')
                return SimpleNamespace(close=lambda: closed.append(True))
            fake_model.Runtime = construct
            tokenizer = SimpleNamespace(no_truncation=lambda: None, no_padding=lambda: None,
                         encode=lambda text, **kwargs: SimpleNamespace(ids=[2 if text == '<bos>' else 50]))
            fake_tokens = ModuleType('tokenizers')
            fake_tokens.Tokenizer = SimpleNamespace(from_file=lambda path: tokenizer)
            output = root / 'parity'
            imports = {'torch': fake_torch, 'torch.nn': fake_nn, 'model': fake_model, 'tokenizers': fake_tokens}
            with patch.dict(sys.modules, imports), patch.object(parity, 'version', return_value='4.57.3'), \
                    patch.object(parity, 'selected_probes', return_value=[{}]), \
                    patch.object(parity, 'template_check', return_value=[]), \
                    patch.object(parity, 'official_model', side_effect=AssertionError('fixture numerical failure')):
                budget = SimpleNamespace(check=lambda: None)
                with self.assertRaises(OSError):
                    parity.verify(assets, checkpoint, output, parity.time() + 60, budget)
                with self.assertRaises(AssertionError):
                    parity.verify(assets, checkpoint, output, parity.time() + 60, budget)
                self.assertEqual(len(list(output.glob('parity-failed*.json'))), 2)
                self.assertEqual(len(closed), 1)
                with self.assertRaisesRegex(RuntimeError, 'previously failed'):
                    parity.verify(assets, checkpoint, output, parity.time() + 60, budget)
                self.assertEqual(len(attempts), 2)
                self.assertEqual(len(closed), 1)


class CollectionRecoveryTests(unittest.TestCase):
    def test_remaining_estimate_counts_only_completed_bound_prefixes(self):
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / 'receipts.jsonl'
            specs = [{'prefix_id': 'a', 'prefix': [1, 2]}, {'prefix_id': 'b', 'prefix': [3]}]
            fingerprint = {'checkpoint_step': 7205, 'checkpoint_sha256': 'frozen'}
            first = {'prefix_id': 'a', 'model_fingerprint': fingerprint, 'observations': [1, 2]}
            path.write_text(json.dumps(first) + '\n' + '{"partial":')
            self.assertEqual(prepare.remaining_requests(specs, path, fingerprint), 1)
            second = {'prefix_id': 'b', 'model_fingerprint': fingerprint, 'observations': [3]}
            path.write_text(json.dumps(first) + '\n' + json.dumps(second) + '\n')
            self.assertEqual(prepare.remaining_requests(specs, path, fingerprint), 0)
            with self.assertRaises(ValueError):
                prepare.remaining_requests(specs, path, {'checkpoint_step': 6419})
    def test_provenance_artifacts_and_actual_origin_are_required(self):
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            saved = {}
            for key in ('control', 'treatment', 'prefix_receipts'):
                path = directory / f'{key}.jsonl'
                path.write_text(key)
                saved[key] = prepare.file_record(path, relative_to=directory)
            saved['collection'] = {'rows': 480, 'control_rows': 480, 'actual_observed_rows': 480,
                 'reconstructed_rows': 0, 'sha256': saved['treatment']['sha256'],
                 'control_sha256': saved['control']['sha256']}
            prepare.verify_collection(directory, saved)
            saved['collection']['actual_observed_rows'] = 0
            with self.assertRaises(ValueError):
                prepare.verify_collection(directory, saved)
            saved['collection']['actual_observed_rows'] = 480
            (directory / 'prefix_receipts.jsonl').write_text('changed')
            with self.assertRaises(ValueError):
                prepare.verify_collection(directory, saved)


if __name__ == '__main__':
    unittest.main(verbosity=2)
