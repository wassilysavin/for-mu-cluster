"""Build and verify the targeted dialogue and function-argument corpora."""

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

from data_tools.transfer_split import items, questions
from eval_tools.io import registry_at
from evaluation import api
from evaluation.actual_history import createActualHistoryHarness
from evaluation.history_train import renderPrompt
from transfer_contract import USER_TURN
from transfer_train import FROZEN, tokenized
import week_train


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
VERSION = "gemprf-targeted-dialogues-v1"
TRAIN = ASSETS / "targeted-train-v1.jsonl"
DEV = ASSETS / "targeted-dev-v1.json"
RECEIPT = ASSETS / "targeted-corpus-v1.json"
GROUPS = {"repeat_original", "replace_answer", "resolve_pending", "function_arguments"}
PANELS = ("closed_loop", "challenge")
SOURCE_MODULES = (
    "training/targeted_argument_cases.py",
    "training/targeted_history_cases.py",
    "training/targeted_dev_cases.py",
    "training/targeted_dialogues.py",
)


def ensure(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def json_bytes(obj):
    return (json.dumps(obj, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=2) + "\n").encode()


def registry():
    path = ASSETS / "registry.json"
    ensure(digest(path.read_bytes()) == FROZEN["registry.json"], "Frozen registry hash differs")
    return registry_at(path)[0]


def context(reg, row, native):
    before = native.createSession(reg, {"dialogueVersion": 2, "sessionId": row["case_id"]})
    for step in row.get("history", ()):
        raw = step.get("raw_output")
        if raw is None:
            raw = native.encodeCalls(step["calls"])
        before = native.applyModelOutput(reg, before, step["question"], raw)
        ensure(
            before["exchanges"][-1]["outcome"] == step["expected_outcome"],
            f"Unexpected history outcome in {row['case_id']}",
        )
    return before


def gold(reg, native, harness, row, before, *, verify_reference=True):
    resolved = harness.resolve(row, before)
    if verify_reference:
        ensure(
            resolved["reference_targets"] == row.get("expected_targets", []),
            f"Wrong authored reference for {row['case_id']}: {resolved['reference_targets']}",
        )
    calls = resolved["allowed_batches"][0]
    target = native.encodeCalls(calls)
    ensure(native.decodeCalls(target) == calls, f"Native call round-trip differs for {row['case_id']}")
    after = native.applyModelOutput(reg, before, resolved["question"], target)
    result = harness.grade(reg, row, before, after, resolved)
    ensure(
        result["success"] and result["critical_status"] == "none_observed",
        f"Gold fails for {row['case_id']}/{resolved['branch']}",
    )
    ensure(after["exchanges"][:-1] == before["exchanges"], f"The gold rewrites history in {row['case_id']}")
    refs = resolved["reference_targets"]
    if row["policy"]["kind"] == "pending_resolve" and refs:
        pending = [item for item in before["semantic"]["pending"] if item["id"] not in refs]
        ensure(
            after["semantic"]["pending"] == pending, f"Gold changes unrelated open questions in {row['case_id']}"
        )
    if row["policy"]["kind"] == "replace" and resolved["branch"] == "present":
        ensure(
            after["semantic"]["turns"][-1]["answers"][0]["revises"] == refs[0]
            and after["semantic"]["turns"][:-1] == before["semantic"]["turns"],
            f"Gold changes unrelated answers in {row['case_id']}",
        )
    return resolved, target, after


def normalized_turns(rows):
    return set(questions(rows))


def check_disjoint(train, dev):
    for split, examples in (("train", train), ("dev", dev)):
        identifiers = [row["case_id"] for row in examples]
        ensure(len(identifiers) == len(set(identifiers)), f"Duplicate {split.upper()} case")
        ensure(
            all(cid.startswith(f"targeted-{split}-") for cid in identifiers), f"Wrong {split.upper()} case prefix"
        )
        ensure({row["group"] for row in examples} == GROUPS, f"Missing {split.upper()} group")
    ensure(not ({r["family_id"] for r in train} & {r["family_id"] for r in dev}), "TRAIN and DEV share a family")
    train_words, dev_words = normalized_turns(train), normalized_turns(dev)
    existing_dev, _ = week_train.public_development()
    for title, overlap in (
        ("New TRAIN and new DEV", train_words & dev_words),
        ("New TRAIN and historical DEV", train_words & existing_dev),
        ("New DEV and historical DEV", dev_words & existing_dev),
    ):
        ensure(not overlap, f"{title} share normalized wording: {sorted(overlap)[:3]}")
    tokenizer = week_train.load_tokenizer()
    for name in (
        "week-train-v1.jsonl",
        "week-train-v2.jsonl",
        "train-full-v2.jsonl",
        "train-volume-v2.jsonl",
        "train-coverage-replay-v2.jsonl",
        "history-train-specs.jsonl",
        "diagnostic-train-contexts.jsonl",
        "transfer-train-roots.jsonl",
        "transfer-train-ordinary.jsonl",
        "transfer-train-contrast.jsonl",
    ):
        checked = set()
        for row in items(ASSETS / name):
            if "input_suffix" in row:
                row = week_train.visible(row["input_suffix"])
            elif "input_ids" in row and "question" not in row:
                prompt = tokenizer.decode(row["input_ids"][: row["prompt_tokens"]], skip_special_tokens=False)
                row = week_train.visible(prompt[prompt.index(USER_TURN) :])
            checked.update(questions(row))
        overlap = checked & dev_words
        ensure(
            checked and not overlap,
            f"No auditable questions or overlapping DEV wording in {name}: {sorted(overlap)[:3]}",
        )
    return {
        "train_distinct_wordings": len(train_words),
        "dev_distinct_wordings": len(dev_words),
        "normalized_cross_split_overlaps": 0,
    }


def pack_train(reg, native, harness, rows):
    tokenizer = week_train.load_tokenizer()
    output, body = BytesIO(), sha256()
    header = {
        "type": "manifest",
        "format": VERSION,
        "split": "train",
        "registry_sha256": reg.fingerprint(),
        "tokenizer_sha256": FROZEN["tokenizer.json"],
    }
    week_train.emit(output, body, header)
    prefixes, blocks, visible = set(), {}, set()
    history_pairs, groups = defaultdict(list), Counter()
    tokens, negatives = Counter(), Counter()
    for row in rows:
        before = context(reg, row, native)
        resolved, target, _ = gold(reg, native, harness, row, before)
        question = resolved["question"]
        for wrong in row.get("rejected_calls", ()):
            failure = native.applyModelOutput(reg, before, question, native.encodeCalls([wrong]))["exchanges"][-1]
            ensure(
                failure["outcome"] == "error" and failure["error_code"] == "invalid_arguments",
                f"Argument mutation was accepted for {row['case_id']}",
            )
            negatives[failure["error_code"]] += 1
        history_pairs[row["family_id"]].append((question, tuple(resolved["reference_targets"])))
        prepared = {
            **row,
            "model_input": renderPrompt(reg, native, before, question),
            "question": question,
            "target": target,
            "calls": resolved["allowed_batches"][0],
            "history_origin": "authored_counterfactual" if row.get("history") else "authored_argument_contrast",
            "ancestor_case_ids": [],
        }
        sequence = tokenized(prepared, tokenizer)
        packed, prefix, key, _ = week_train.pack(row["group"], prepared, sequence)
        ensure(key not in visible, f"Repeated visible TRAIN input at {row['case_id']}")
        visible.add(key)
        packed["reference_targets"] = resolved["reference_targets"]
        packed["history_outcomes"] = [exchange["outcome"] for exchange in before["exchanges"]]
        if packed["prefix"] not in prefixes:
            prefixes.add(packed["prefix"])
            week_train.write_prefix(output, body, packed["prefix"], prefix, blocks)
        week_train.emit(output, body, packed)
        groups[row["group"]] += 1
        tokens["prompt"] += packed["prompt_tokens"]
        tokens["target"] += packed["target_tokens"]

    paired = sum(
        len({q for q, _ in family}) == 1 and len({refs for _, refs in family}) > 1
        for family in history_pairs.values()
    )
    ensure(paired > 0 and negatives, "Targeted TRAIN lacks executable referent and argument contrasts")
    footer = {
        "type": "summary",
        "cases": len(rows),
        "groups": dict(groups),
        "families_with_same_question_and_different_refs": paired,
        "targeted_invalid_mutations_rejected": dict(negatives),
        "prefixes": len(prefixes),
        "tokens": dict(tokens),
        "body_sha256": body.hexdigest(),
    }
    output.write((week_train.compact(footer) + "\n").encode())
    return output.getvalue(), footer


def audit_dev(reg, native, harness, rows):
    panels = {panel: [] for panel in PANELS}
    dialogues = defaultdict(list)
    for row in rows:
        row = dict(row)
        if "history" in row:
            row["initial_prefix"] = row.pop("history")
        panel = row["panel"]
        panels[panel].append({**row, "split": "dev"})
        if panel == "closed_loop" and row.get("dialogue_id"):
            ensure(not row.get("initial_prefix"), "Closed-loop DEV must start from the model's own history")
            dialogues[row["dialogue_id"]].append(row)
        else:
            ensure(not row.get("dialogue_id"), "Fixed-history DEV must be evaluated independently")
            gold(reg, native, harness, row, harness.initial(reg, row))
    for name, turns in dialogues.items():
        ensure([r["turn_index"] for r in turns] == [1, 2, 3, 4, 5], f"Incomplete DEV dialogue: {name}")
        before = harness.initial(reg, turns[0])
        for row in turns:
            _, _, before = gold(reg, native, harness, row, before)
        # Probe each initial failure on its own, letting later policies follow real state.
        for failed in range(1, len(turns)):
            before = harness.initial(reg, turns[0])
            for row in turns:
                if row["turn_index"] == failed:
                    question = harness.resolve(row, before)["question"]
                    before = native.applyModelOutput(reg, before, question, "unparseable saved response")
                    ensure(before["exchanges"][-1]["outcome"] == "error", f"Failure probe committed in {name}")
                else:
                    _, _, before = gold(reg, native, harness, row, before, verify_reference=False)
    ensure(dialogues and all(panels.values()), "DEV is missing a complete panel")
    summary = {
        "cases": len(rows),
        "panels": {name: len(rows) for name, rows in panels.items()},
        "whole_dialogues": len(dialogues),
        "checked_early_failure_paths": sum(len(turns) - 1 for turns in dialogues.values()),
        "groups": dict(Counter(row["group"] for row in rows)),
    }
    return json_bytes(
        {"schema": VERSION, "split": "dev", "registry_sha256": reg.fingerprint(), "panels": panels}
    ), summary


def generated():
    from .targeted_argument_cases import cases as argument_cases
    from .targeted_dev_cases import cases as dev_cases
    from .targeted_history_cases import cases as history_cases

    reg, native = registry(), api()
    harness = createActualHistoryHarness(native)
    train = [
        {**row, "reference_eligible": row["group"] != "function_arguments"}
        for row in (*history_cases(reg), *argument_cases(reg))
    ]
    dev = [{**row, "reference_eligible": row["group"] != "function_arguments"} for row in dev_cases(reg)]
    train = [
        {**row, "panel": "challenge", "split": "train", "cohort": "supported", "turn_index": 1} for row in train
    ]
    dev = [{"turn_index": 1, "cohort": "supported", **row} for row in dev]
    separation = check_disjoint(train, dev)
    train_bytes, train_summary = pack_train(reg, native, harness, train)
    dev_bytes, dev_summary = audit_dev(reg, native, harness, dev)
    receipt = {
        "schema": VERSION,
        "train": {**train_summary, "sha256": digest(train_bytes), "bytes": len(train_bytes)},
        "dev": {**dev_summary, "sha256": digest(dev_bytes), "bytes": len(dev_bytes)},
        "separation": separation,
        "registry_file_sha256": FROZEN["registry.json"],
        "tokenizer_sha256": FROZEN["tokenizer.json"],
        "authored_sources": {name: digest((ROOT / name).read_bytes()) for name in SOURCE_MODULES},
        "public_dev_sha256": {name: week_train.digest(ASSETS / name) for name in week_train.PUBLIC_DEV},
    }
    return {TRAIN: train_bytes, DEV: dev_bytes, RECEIPT: json_bytes(receipt)}, receipt


def publish(path, payload):
    with NamedTemporaryFile(dir=path.parent, prefix=".targeted-", delete=False) as stream:
        staging = Path(stream.name)
        try:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
            try:
                os.link(staging, path)
            except FileExistsError:
                ensure(path.read_bytes() == payload, f"Frozen output differs: {path.name}")
        finally:
            staging.unlink()


def iter_train():
    """Read the frozen native training examples and their loss-mask boundaries."""
    receipt = json.loads(RECEIPT.read_bytes())
    raw = TRAIN.read_bytes()
    ensure(
        receipt["schema"] == VERSION and digest(raw) == receipt["train"]["sha256"], "Targeted TRAIN bytes changed"
    )
    blocks, prefixes, tokenizer = [], {}, week_train.load_tokenizer()
    for line in raw.splitlines():
        row = json.loads(line)
        if row["type"] == "developer_blocks":
            blocks.extend(row["fragments"])
        elif row["type"] == "developer_prefix":
            prefixes[row["sha256"]] = "},{".join(blocks[i] for i in row["blocks"])
        elif row["type"] == "example":
            yield {"case_id": row["case_id"], **week_train.unpack(row, prefixes, tokenizer)}


def load_dev():
    """Read the frozen DEV panels for ActualHistoryHarness."""
    receipt = json.loads(RECEIPT.read_bytes())
    raw = DEV.read_bytes()
    ensure(receipt["schema"] == VERSION and digest(raw) == receipt["dev"]["sha256"], "Targeted DEV bytes changed")
    return json.loads(raw)["panels"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "check"))
    args = parser.parse_args()
    expected, summary = generated()
    for path, payload in expected.items():
        if args.action == "build":
            publish(path, payload)
        else:
            ensure(path.read_bytes() == payload, f"Frozen output differs: {path.name}")
    ensure(sum(1 for _ in iter_train()) == summary["train"]["cases"], "Targeted TRAIN reader count differs")
    ensure(sum(map(len, load_dev().values())) == summary["dev"]["cases"], "Targeted DEV reader count differs")
    print(
        json.dumps(
            {"train": summary["train"], "dev": summary["dev"], "separation": summary["separation"]}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
