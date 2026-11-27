"""Trace shared critical flags and invalid arguments from the saved weekly run."""

from hashlib import file_digest
import sys

from eval_tools.io import registry_at

from .inspect_week_comparison import RUN_ID
from .inspect_week_failures import DEV1250, load_evidence
from .student_evidence import ROOT, checked_path, inventory, read, verify
from .week_argument_trace import render_arguments
from .week_dialogue_trace import render_dialogues


def inspect(root=ROOT):
    plans, saved, journals = load_evidence(root)
    manifest = read(root, root / "runs" / RUN_ID / "experiment.json")
    files = inventory(manifest["code"]["files"])
    config_path = verify(root, root / "experiments.json", files["experiments.json"])
    declared = read(root, config_path)["assets"]["registry"]
    registry_path = checked_path(root, root / "assets" / "registry.json")
    with registry_path.open("rb") as stream:
        checksum = file_digest(stream, "sha256").hexdigest()
    if declared["path"] != "assets/registry.json" or checksum != declared["sha256"]:
        raise ValueError("Function registry differs from the weekly run")
    registry, _ = registry_at(registry_path)

    dialogue = render_dialogues(
        registry_path, root / DEV1250, plans["legacy1250"], saved["legacy1250"], journals["legacy1250"]
    )
    arguments = render_arguments(registry, plans["legacy155"], saved["legacy155"], journals["legacy155"])
    return "\n".join([*dialogue, "", *arguments])


if __name__ == "__main__":
    try:
        print(inspect())
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        sys.exit(f"Weekly failure details: {error}")
