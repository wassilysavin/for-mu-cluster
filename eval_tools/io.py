"""JSONL protocol and durable observation receipts shared by the evaluation commands."""

import hashlib
import json
import os
from pathlib import Path
import sys

from evaluation.schema import canonical
from evaluation.registry import createRegistry


ROOT = Path(__file__).resolve().parents[1]


def require(condition, detail):
    if not condition:
        raise ValueError(detail)


def digest(value):
    if not isinstance(value, (bytes, str)):
        value = canonical(value)
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode()).hexdigest()


def records(path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def registry_at(path):
    source = json.loads(Path(path).read_bytes())
    registry = createRegistry(source["knowledge"], source["contracts"])
    require(registry.fingerprint() == source["fingerprint"], "Registry fingerprint mismatch")
    return registry, source


def line_bytes(record):
    return (json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()


def emit(record):
    sys.stdout.buffer.write(line_bytes(record))
    sys.stdout.buffer.flush()


def receive():
    line = sys.stdin.buffer.readline()
    if not line:
        raise EOFError("Model provider disconnected; saved observations remain resumable")
    return json.loads(line)


def read_saved(path, *, retain_valid_tail=True):
    path = Path(path)
    try:
        handle = path.open("rb+")
    except FileNotFoundError:
        return []
    with handle:
        saved = []
        trailing_whitespace = False
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                return saved
            if line.endswith(b"\n"):
                if not retain_valid_tail and line == b"\n":
                    continue
                if retain_valid_tail and not line.strip():
                    trailing_whitespace = True
                    continue
                require(not trailing_whitespace, "Empty line inside observation journal")
                saved.append(json.loads(line))
                continue
            if retain_valid_tail:
                try:
                    value = json.loads(line)
                except (ValueError, UnicodeError):
                    pass
                else:
                    require(not trailing_whitespace, "Empty line inside observation journal")
                    saved.append(value)
                    handle.write(b"\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                    return saved
            handle.truncate(offset)
            handle.flush()
            os.fsync(handle.fileno())
            return saved


def save(handle, record, reserve, *, check_space=False):
    line = line_bytes(record)
    if reserve or check_space:
        space = os.statvfs(Path(handle.name).parent)
        require(space.f_bavail * space.f_frsize >= reserve + len(line),
                "Run filesystem lost its free-space reserve during evaluation")
    handle.write(line)
    handle.flush()
    os.fsync(handle.fileno())
