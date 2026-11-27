"""Check that the detailed report uses the run's frozen public registry."""

from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from diagnostics import inspect_week_failure_details as details


class FailureDetailsTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.registry = self.root / "assets" / "registry.json"
        self.registry.parent.mkdir()
        raw = (details.ROOT / "assets" / "registry.json").read_bytes()
        self.registry.write_bytes(raw)
        config = json.dumps(
            {"assets": {"registry": {"path": "assets/registry.json", "sha256": sha256(raw).hexdigest()}}}
        ).encode()
        (self.root / "experiments.json").write_bytes(config)
        manifest = self.root / "runs" / details.RUN_ID / "experiment.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            json.dumps(
                {
                    "code": {
                        "files": [
                            {"path": "experiments.json", "bytes": len(config), "sha256": sha256(config).hexdigest()}
                        ]
                    }
                }
            )
        )
        self.data = tuple({split: object() for split in ("legacy155", "legacy1250")} for _ in range(3))

    def test_passes_the_verified_evidence_to_both_readers(self):
        plans, saved, journals = self.data
        with (
            patch.object(details, "load_evidence", return_value=self.data),
            patch.object(details, "render_dialogues", return_value=["dialogues"]) as dialogues,
            patch.object(details, "render_arguments", return_value=["arguments"]) as arguments,
        ):
            self.assertEqual(details.inspect(self.root), "dialogues\n\narguments")
        dialogues.assert_called_once_with(
            self.registry,
            self.root / details.DEV1250,
            plans["legacy1250"],
            saved["legacy1250"],
            journals["legacy1250"],
        )
        arguments.assert_called_once()
        registry, *inputs = arguments.call_args.args
        self.assertEqual(registry.fingerprint(), details.registry_at(self.registry)[0].fingerprint())
        self.assertEqual(inputs, [plans["legacy155"], saved["legacy155"], journals["legacy155"]])

    def test_changed_registry_is_rejected_before_either_reader(self):
        self.registry.write_text("{}")
        with (
            patch.object(details, "load_evidence", return_value=self.data),
            patch.object(details, "render_dialogues") as dialogues,
            patch.object(details, "render_arguments") as arguments,
            self.assertRaisesRegex(ValueError, "Function registry differs"),
        ):
            details.inspect(self.root)
        dialogues.assert_not_called()
        arguments.assert_not_called()
