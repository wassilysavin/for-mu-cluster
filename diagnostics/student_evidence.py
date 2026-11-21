"""Load the three completed FunctionGemma DEV runs and their recorded identities."""

from hashlib import file_digest, sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("runs/actual-history-v3-4gpu/revisions/rope-alignment-v1/recovery/checkpoint-io-v1")
NAMES = ("baseline-7205", "control-1729", "treatment-1729")


def digest(value):
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return sha256(data.encode()).hexdigest()


def checked_path(root, path):
    resolved = path.resolve(strict=True)
    if resolved != path or not resolved.is_relative_to(root):
        raise ValueError(f"Unexpected path for saved evidence: {path}")
    return resolved


def read(root, path):
    return json.loads(checked_path(root, path).read_text(encoding="utf-8"))


def verify(root, path, entry):
    path = checked_path(root, path)
    if entry["path"] != path.relative_to(root).as_posix() or path.stat().st_size != entry["bytes"]:
        raise ValueError(f"Saved path or size mismatch: {path}")
    with path.open("rb") as stream:
        if file_digest(stream, "sha256").hexdigest() != entry["sha256"]:
            raise ValueError(f"Saved SHA-256 mismatch: {path}")
    return path


def inventory(records):
    entries = {entry["path"]: entry for entry in records}
    if len(entries) != len(records):
        raise ValueError("Duplicate paths in saved evidence inventory")
    return entries


def from_inventory(root, path, entries):
    relative = path.relative_to(root).as_posix()
    if relative not in entries:
        raise ValueError(f"Missing evidence receipt for {relative}")
    return verify(root, path, entries[relative])


def validate_replay(saved_score, replay_score):
    if digest(saved_score) != digest(replay_score) or replay_score.get("observed_records") != 1250:
        raise ValueError("Replayed DEV1250 scores differ or the saved evaluation is incomplete")


def load_saved(root=ROOT):
    root = Path(root).resolve(strict=True)
    parent = root / SOURCE
    child = parent / "completion-v1"
    terminal = read(root, child / "result.json")
    artifacts = inventory(terminal["artifacts"])
    for filename in ("experiment.json", "manifest-binding.json"):
        from_inventory(root, child / filename, artifacts)
    completion = read(root, child / "experiment.json")
    binding = digest(completion)
    if (read(root, child / "manifest-binding.json") != {"sha256": binding}
            or terminal["identity_sha256"] != binding or completion["source_directory"] != SOURCE.as_posix()
            or terminal["completion_policy"] != completion["policy"]
            or terminal["status"] not in {"not_promoted", "inconclusive", "exploratory_gain_second_seed_not_run"}
            or terminal["product_acceptance"] is not False or terminal["TEST_opened"] is not False
            or terminal["exploratory"] is not True or len(terminal["pairs"]) != 1
            or terminal["pairs"][0]["seed"] != 1729):
        raise ValueError("Expected the completed seed-1729 comparison")
    verify(root, parent / "experiment.json", completion["source_manifest"])
    manifest = read(root, parent / "experiment.json")
    original_binding = digest(manifest)
    inherited = inventory(completion["source_snapshot"]["immutable"])
    from_inventory(root, parent / "identities.json", inherited)
    identities = read(root, parent / "identities.json")

    code = inventory(manifest["code"]["files"])
    from source_provenance import verify_sources
    legacy = "evaluate-actual.mjs" in code
    entrypoint = "evaluate-actual.mjs" if legacy else "eval_tools/evaluate_actual.py"
    if entrypoint not in code:
        raise ValueError("Missing evidence receipt for evaluator")
    if legacy:
        modules = ("actual-history", "development-evaluator", "dialogue", "evaluation", "functiongemma",
                   "history-control", "history-train", "known-review", "model-input", "model-result",
                   "model-tools", "prompt", "registry", "schema", "session")
        required = [f"evaluation/{name}.mjs" for name in modules]
    else:
        required = [p.relative_to(ROOT).as_posix() for p in (ROOT / "evaluation").glob("*.py")]
    names = [entrypoint, *required]
    if any(name not in code for name in names):
        raise ValueError("Missing evidence receipt for evaluator module")
    verify_sources(root, dict(manifest["code"], files=[code[name] for name in names]))
    assets = manifest["assets"]
    registry = verify(root, root / "assets/registry.json", assets["registry"])
    bundle = verify(root, root / "assets/actual-history-v3-dev.json", assets["history_dev"])

    def stage(name):
        path = parent / "stages" / f"{name}.json"
        if path.exists():
            entries, expected = inherited, original_binding
        else:
            path = child / "stages" / f"{name}.json"
            entries, expected = artifacts, binding
        from_inventory(root, path, entries)
        receipt = read(root, path)
        if receipt["identity_sha256"] != expected:
            raise ValueError(f"Stage belongs to a different run: {name}")
        return receipt["result"]

    arms = {}
    for label in NAMES:
        baseline = label == "baseline-7205"
        name = "baseline" if baseline else label
        evaluation = stage("baseline" if baseline else f"{name}-final")
        reported = terminal["baseline"] if baseline else terminal["pairs"][0][name.split("-")[0]]
        checkpoint = manifest["origin"]["checkpoint"] if baseline else stage(f"{name}-trained")["checkpoint"]
        directory = parent / "evaluations" / name
        identity = identities[name]
        if (reported != evaluation or checkpoint != evaluation["checkpoint"] or identity["experiment_id"] != name
                or read(root, directory / "input.json") != {"run": identity, "known": False, "checkpoint": checkpoint}):
            raise ValueError(f"Saved {name} evaluation does not match its training or input identity")
        observations = verify(root, directory / "observations.jsonl", evaluation["observations"])
        arms[label] = (evaluation["score"], observations)
    return registry, bundle, arms
