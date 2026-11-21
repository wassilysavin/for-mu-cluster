"""Authenticate historical source receipts separately from the code used for a new run."""

from functools import lru_cache
from hashlib import sha256
from pathlib import Path, PurePosixPath
import re
import subprocess

# Last published tree containing the historical experiment implementations.
WEEKLY_REVISION = "013f8878f9658229f1c65d72af2a314db7631b7b"
SOURCE_DIRS = {"evaluation", "data_tools", "eval_tools", "tests", "training", "diagnostics", "configs"}


def runtime_sources(root):
    """Package implementations shared by the dataset tools and scoring commands."""
    packages = ("data_tools", "eval_tools", "evaluation")
    files = ["private_runtime.py", "source_provenance.py"]
    for package in packages:
        files.extend(
            p.relative_to(root).as_posix()
            for p in sorted((root / package).rglob("*.py"))
            if not p.name.startswith("test_")
        )
    return files


@lru_cache(maxsize=128)
def _require_ancestor(root, revision, head, published):
    for tip in dict.fromkeys((head, published)):
        result = subprocess.run(
            ["git", "--no-replace-objects", "-C", str(root), "merge-base", "--is-ancestor", revision, tip],
            capture_output=True,
        )
        if result.returncode == 0:
            return
    raise ValueError("Source revision is outside the current and published experiment histories")


@lru_cache(maxsize=512)
def _blob_record(root, revision, name):
    data = subprocess.check_output(
        ["git", "--no-replace-objects", "-C", str(root), "cat-file", "blob", f"{revision}:{name}"],
        stderr=subprocess.PIPE,
    )
    return {"path": name, "bytes": len(data), "sha256": sha256(data).hexdigest()}


def archived_record(root, revision, name):
    path = PurePosixPath(name)
    if (
        not isinstance(revision, str)
        or not re.fullmatch(r"[0-9a-f]{40}", revision)
        or not path.parts
        or path.is_absolute()
        or ".." in path.parts
        or str(path) != name
        or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", p) for p in path.parts)
        or (len(path.parts) > 1 and path.parts[0] not in SOURCE_DIRS)
    ):
        raise ValueError(f"Unexpected source path or revision: {name}")
    prefix, head = subprocess.check_output(
        ["git", "--no-replace-objects", "-C", str(root), "rev-parse", "--show-prefix", "HEAD"],
        text=True,
        stderr=subprocess.PIPE,
    ).splitlines()
    if prefix:
        raise ValueError("Source inventory must belong to the repository root")
    _require_ancestor(root, revision, head, WEEKLY_REVISION)
    return dict(_blob_record(root, revision, name))


def verify_sources(root, code):
    """Verify exact bytes for an immutable input run; never grant permission to resume it."""
    root = Path(root).resolve(strict=True)
    files = code.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Expected a nonempty frozen source inventory")
    seen = set()
    for expected in files:
        if not isinstance(expected, dict) or not isinstance(expected.get("path"), str):
            raise ValueError("Invalid source receipt")
        name = expected["path"]
        path = root / name
        if (
            not name
            or Path(name).as_posix() != name
            or Path(name).is_absolute()
            or path.resolve() != path
            or not path.is_relative_to(root)
        ):
            raise ValueError(f"Unsafe path in frozen source inventory: {name}")
        if name in seen:
            raise ValueError(f"Duplicate path in frozen source inventory: {name}")
        seen.add(name)
        if path.is_file() and path.stat().st_size == expected["bytes"]:
            from ledger import file_record

            if file_record(path, relative_to=root) == expected:
                continue
        try:
            archived = archived_record(root, code["git_commit"], name) if code.get("git_commit") else None
        except (ValueError, subprocess.CalledProcessError):
            archived = None
        if archived != expected:
            raise ValueError(f"Frozen artifact changed: {name}")
