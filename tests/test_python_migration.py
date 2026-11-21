"""Protect the boundary between a frozen source receipt and a new Python run."""

import json
from hashlib import sha256
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from ledger import file_record
from private_runtime import ROOT, private_command
from source_provenance import WEEKLY_REVISION, archived_record, runtime_sources, verify_sources
from week_train import same_corpus


class SourceReceiptTests(unittest.TestCase):
    def test_rejects_empty_and_duplicate_source_inventories(self):
        entry = file_record(ROOT / "source_provenance.py", relative_to=ROOT)
        with self.assertRaisesRegex(ValueError, "nonempty"):
            verify_sources(ROOT, {"files": []})
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            verify_sources(ROOT, {"files": [entry, entry]})

    def test_archived_native_source_follows_commit_ancestry_across_checkouts(self):
        with TemporaryDirectory(dir=ROOT / ".runtime/tmp") as directory:
            root = Path(directory)

            def git(*args):
                return subprocess.check_output(
                    [
                        "git",
                        "-C",
                        directory,
                        "-c",
                        "user.name=Test",
                        "-c",
                        "user.email=test@example.invalid",
                        "-c",
                        "commit.gpgSign=false",
                        "-c",
                        "core.hooksPath=/dev/null",
                        *args,
                    ],
                    text=True,
                    stderr=subprocess.PIPE,
                ).strip()

            git("init", "-q")
            source = root / "native.py"
            source.write_text("version = 1\n")
            original = file_record(source, relative_to=root)
            git("add", "native.py")
            git("commit", "-qm", "Original source")
            revision = git("rev-parse", "HEAD")
            code = {"git_commit": revision, "files": [original]}
            source.write_text("version = 2\n")
            git("commit", "-qam", "New native version")
            verify_sources(root, code)
            git("checkout", "--orphan", "independent")
            source.write_text("version = 3\n")
            git("add", "native.py")
            git("commit", "-qm", "Independent history")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                verify_sources(root, code)
            with patch("source_provenance.WEEKLY_REVISION", revision):
                verify_sources(root, code)

    def test_new_runtime_identities_cover_launcher_and_provenance(self):
        from transfer_run import bindings
        from week_train import SOURCES

        required = {"private_runtime.py", "source_provenance.py"}
        self.assertTrue(required <= set(runtime_sources(ROOT)))
        self.assertTrue(required <= set(SOURCES))
        self.assertTrue(required <= {entry["path"] for entry in bindings()["files"]})

    def test_archived_implementation_is_readable_without_running_it(self):
        entry = archived_record(ROOT, WEEKLY_REVISION, "evaluate.mjs")
        self.assertFalse((ROOT / entry["path"]).exists())
        verify_sources(ROOT, dict(git_commit=WEEKLY_REVISION, files=[entry]))
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            verify_sources(ROOT, dict(git_commit=WEEKLY_REVISION, files=[dict(entry, sha256="0" * 64)]))
        with self.assertRaisesRegex(ValueError, "Unsafe path"):
            verify_sources(ROOT, dict(files=[dict(entry, path="../evaluate.mjs")]))

    def test_legacy_report_commands_dispatch_to_the_python_package(self):
        cmd = private_command(["python", "-u", "inspect_qwen_cases.py", "--remaining-reviews"], {})
        self.assertEqual(cmd, ["python", "-u", "-m", "diagnostics.inspect_qwen_cases", "--remaining-reviews"])

    def test_rebuild_accepts_only_the_pinned_original_and_the_same_records(self):
        def dataset(path, author, question):
            rows = [
                {"type": "manifest", "author_files": {"policy": author}},
                {"type": "example", "question": question},
            ]
            body = "".join(json.dumps(r) + "\n" for r in rows).encode()
            path.write_bytes(
                body + json.dumps({"type": "summary", "body_sha256": sha256(body).hexdigest()}).encode() + b"\n"
            )
            return sha256(path.read_bytes()).hexdigest()

        with TemporaryDirectory(dir=ROOT / ".runtime/tmp") as directory:
            old, new = (Path(directory) / n for n in ("original", "candidate"))
            expected = dataset(old, "historical", "Repeat the first answer")
            dataset(new, "python", "Repeat the first answer")
            self.assertTrue(same_corpus(old, new, expected))
            self.assertFalse(same_corpus(old, new, "0" * 64))
            raw = new.read_bytes().splitlines()
            summary = json.loads(raw[-1])
            summary["body_sha256"] = "0" * 64
            new.write_bytes(b"\n".join([*raw[:-1], json.dumps(summary).encode()]) + b"\n")
            self.assertFalse(same_corpus(old, new, expected))
            dataset(new, "python", "Repeat the other answer")
            self.assertFalse(same_corpus(old, new, expected))
