"""Bind cluster runs and evaluations to immutable, reproducible inputs."""

from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import fcntl
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
import re
import subprocess
from tempfile import NamedTemporaryFile, TemporaryDirectory


SCHEMA_VERSION = 1
SAFE_ID = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,99}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
STAGE_EVENTS = {"started", "resumed", "completed", "failed"}


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def file_record(path, *, relative_to=None):
    path = Path(path).resolve(strict=True)
    with path.open("rb") as stream:
        digest = file_digest(stream, "sha256").hexdigest()
        size = os.fstat(stream.fileno()).st_size
    name = path.name if relative_to is None else path.relative_to(Path(relative_to).resolve()).as_posix()
    return {"path": name, "sha256": digest, "bytes": size}


def _git(root, *arguments):
    try:
        result = subprocess.run(["git", "--no-optional-locks", "-C", str(root), *arguments],
                                capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def code_record(repo_root, paths):
    root = Path(repo_root).resolve(strict=True)
    files = [file_record(root / path, relative_to=root) for path in paths]
    if not files or len({entry["path"] for entry in files}) != len(files):
        raise ValueError("Expected a nonempty list of distinct code files")
    git_root = _git(root, "rev-parse", "--show-toplevel")
    commit = _git(root, "rev-parse", "--verify", "HEAD") if git_root == str(root) else None
    status = _git(root, "status", "--porcelain", "--untracked-files=normal") if commit else None
    return {"git_commit": commit, "dirty": None if status is None else bool(status),
            "files": sorted(files, key=lambda entry: entry["path"])}


def _sync_directory(directory):
    fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish_json(path, value):
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(_json_bytes(value) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
            os.link(temporary, path)
            _sync_directory(path.parent)
        finally:
            temporary.unlink()


def _repair_partial_event(path):
    if not path.exists():
        return
    with path.open("rb+") as stream:
        valid_end = 0
        while line := stream.readline():
            if not line.endswith(b"\n"):
                stream.truncate(valid_end)
                stream.flush()
                os.fsync(stream.fileno())
                return
            valid_end = stream.tell()


def _require_record(value, label):
    if not isinstance(value, dict) or not SHA256.fullmatch(str(value.get("sha256", ""))):
        raise ValueError(f"{label} must include its SHA-256")


def _check_identity(identity):
    if not isinstance(identity, dict):
        raise ValueError("Run identity must be an object")
    for key in ("experiment_id", "architecture_id"):
        if not isinstance(identity.get(key), str) or not identity[key].strip():
            raise ValueError(f"Missing {key}")
    for key in ("code", "recipe", "runtime"):
        if not isinstance(identity.get(key), dict) or not identity[key]:
            raise ValueError(f"Missing {key} identity")
    code_files = identity["code"].get("files")
    if not isinstance(code_files, list) or not code_files:
        raise ValueError("Code identity must include file hashes")
    for entry in code_files:
        _require_record(entry, "Code file")
    if "train" not in identity:
        raise ValueError("Missing train identity")
    if identity["train"] is not None:
        _require_record(identity["train"], "Training dataset")
    evaluation = identity.get("evaluation")
    _require_record(evaluation, "Evaluation dataset")
    if evaluation.get("split") not in {"DEV700", "DEV-VOLUME", "DEV-HISTORY"}:
        raise ValueError("The cluster evaluator requires an approved development split")


class Run:
    def __init__(self, path, manifest, stage):
        self.path = path
        self.identity_sha256 = manifest["identity_sha256"]
        self._stage = stage
        self._active = True

    def event(self, kind, **details):
        if not self._active:
            raise RuntimeError("Run lock is no longer held")
        entry = {"time": _now(), "stage": self._stage, "event": kind, "details": details}
        with (self.path / "events.jsonl").open("ab") as stream:
            stream.write(_json_bytes(entry) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())

    def write_result(self, metrics, checkpoint, *, name="result.json", **details):
        if not self._active:
            raise RuntimeError("Run lock is no longer held")
        _require_record(checkpoint, "Evaluated checkpoint")
        if not isinstance(metrics, dict) or not metrics:
            raise ValueError("Evaluation result must contain measured metrics")
        result = {"schema_version": SCHEMA_VERSION, "run_id": self.path.name,
                  "identity_sha256": self.identity_sha256, "created_at": _now(),
                  "checkpoint": checkpoint, "metrics": metrics, "details": details}
        publish_json(self.path / name, result)
        return result


@contextmanager
def execution_lease(runs_root):
    runs_root.mkdir(parents=True, exist_ok=True)
    fd = os.open(runs_root / ".execution.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another training or evaluation job holds the execution lock") from exc
        yield fd
    finally:
        os.close(fd)


@contextmanager
def open_run(runs_root, run_id, identity, *, resume=False, stage="train", lease_held=False):
    if not SAFE_ID.fullmatch(run_id) or not SAFE_ID.fullmatch(stage):
        raise ValueError("Invalid run ID or stage")
    _check_identity(identity)
    if stage == "train" and identity["train"] is None:
        raise ValueError("A training run requires a frozen training dataset")
    identity = json.loads(_json_bytes(identity))
    identity_hash = sha256(_json_bytes(identity)).hexdigest()
    root = Path(runs_root).resolve()
    path = root / run_id
    with nullcontext() if lease_held else execution_lease(root):
        if not resume:
            if path.exists() or path.is_symlink():
                raise FileExistsError(f"Run directory already exists: {path}")
            manifest = {"schema_version": SCHEMA_VERSION, "run_id": run_id,
                        "created_at": _now(), "identity_sha256": identity_hash,
                        "identity": identity}
            with TemporaryDirectory(prefix=".run-", dir=root) as staging:
                publish_json(Path(staging) / "run.json", manifest)
                os.rename(staging, path)
            _sync_directory(root)
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"Invalid run directory {path}")
        if resume:
            manifest = _read_manifest(path)
            if manifest["identity_sha256"] != identity_hash:
                raise ValueError("Run identity changed. Start a new run ID.")
            if (path / "result.json").exists():
                raise ValueError("Run already has an evaluation result. Start a new run ID.")
        _repair_partial_event(path / "events.jsonl")
        run = Run(path, manifest, stage)
        run.event("resumed" if resume else "started")
        try:
            yield run
        except BaseException as exc:
            try:
                run.event("failed", error=type(exc).__name__, message=str(exc))
            except OSError:
                pass
            raise
        else:
            run.event("completed")
        finally:
            run._active = False


def _read_manifest(path):
    manifest = json.loads((path / "run.json").read_bytes())
    identity_hash = sha256(_json_bytes(manifest["identity"])).hexdigest()
    if (manifest["schema_version"] != SCHEMA_VERSION or manifest["run_id"] != path.name
            or manifest["identity_sha256"] != identity_hash):
        raise ValueError(f"Invalid run manifest {path / 'run.json'}")
    return manifest


def read_result(path, name="result.json", *, identity=None):
    result_path = path / name
    if not result_path.is_file():
        return None
    manifest = _read_manifest(path)
    result = json.loads(result_path.read_bytes())
    if (result["schema_version"] != SCHEMA_VERSION or result["identity_sha256"] != manifest["identity_sha256"]
            or result["run_id"] != path.name or (identity is not None and manifest["identity"] != identity)):
        raise ValueError(f"Unbound evaluation result {result_path}")
    return result


def _stage_status(path):
    statuses = {}
    if path.exists():
        with path.open("rb") as stream:
            for line in stream:
                if not line.endswith(b"\n"):
                    break
                event = json.loads(line)
                if event["event"] in STAGE_EVENTS:
                    statuses[event["stage"]] = event["event"]
    return statuses


def report(runs_root):
    root = Path(runs_root)
    rows = []
    if not root.exists():
        return rows
    for path in sorted(root.iterdir()):
        if path.name.startswith(".") or path.is_symlink() or not path.is_dir() or not (path / "run.json").exists():
            continue
        manifest = _read_manifest(path)
        identity = manifest["identity"]
        result = read_result(path)
        milestones = [read_result(path, item.name) for item in sorted(path.glob("result-step-*.json"))]
        rows.append({"run_id": path.name, "created_at": manifest["created_at"],
                     "identity_sha256": manifest["identity_sha256"],
                     "experiment_id": identity["experiment_id"],
                     "architecture_id": identity["architecture_id"],
                     "train": identity["train"], "evaluation": identity["evaluation"],
                     "code": identity["code"], "recipe": identity["recipe"],
                     "runtime": identity["runtime"],
                     "status": _stage_status(path / "events.jsonl"),
                     "checkpoint": result["checkpoint"] if result else None,
                     "metrics": result["metrics"] if result else None,
                     **({"milestones": milestones} if milestones else {})})
    return rows
