"""Shared file identities and streaming JSON for the corpus tools."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
SOURCE_SHA = "024d21fba1d2fe73dc9f8d795552303acaeb099675d9d78df179c082c7e7afcb"
REGISTRY_SHA = "b006fbacf4fd0f7f423b9178c7f365e98a6cc91bc99d8c9a16277bfd5954d4d0"


def check(value, message="Invalid corpus invariant"):
    if not value:
        raise ValueError(message)


def sha(value):
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def compact(value):
    from evaluation.schema import js_json

    return js_json(value)


def read_json(name, digest=None):
    raw = (ASSETS / name).read_bytes()
    if digest:
        check(sha(raw) == digest, f"Frozen TRAIN asset changed: {name}")
    return json.loads(raw)


def rows(path, digest=None):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for raw in stream:
            hasher.update(raw)
            yield json.loads(raw)
    if digest:
        check(hasher.hexdigest() == digest, f"Frozen TRAIN asset changed: {path.name}")


def send(value):
    print(compact(value), flush=True)


@lru_cache(maxsize=1)
def runtime():
    from evaluation import api

    return api()


def render(registry, before, question):
    from evaluation.history_train import renderPrompt

    return renderPrompt(registry, runtime(), before, question)


def registry():
    from .transfer_contract import transferRegistry

    return transferRegistry(read_json("registry.json", REGISTRY_SHA))


def session(reg, family):
    return runtime().createSession(reg, {"dialogueVersion": 2, "sessionId": family})


def run_generator(generator):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    for row in generator(parser.parse_args().render):
        send(row)
