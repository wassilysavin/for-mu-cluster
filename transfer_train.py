"""Verify and materialize matched semantic-transfer TRAIN schedules without inference."""

from collections import Counter, defaultdict
from hashlib import file_digest, sha256
import json
import os
from pathlib import Path
import random
import subprocess

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
SOURCE = ROOT / "runs/actual-history-v3-4gpu/revisions/rope-alignment-v1/recovery/checkpoint-io-v1"
PREPARE_BYTES = 100 * 1024 ** 2
SEEDS = (1729, 2718)
ARMS = ("ordinary", "contrast")
# These selected legacy rows ask about an endpoint but supervise the generic direction operation.
EXCLUDED_SHARED = frozenset({"train-dialogue-0069-turn-1", "train-dialogue-0120-turn-4"})


def generic_interval_conflict(row):
    question = row.get("question", "").lower()
    for call in row.get("calls", []):
        if call["name"] in {"explain_change_effect", "change_explain_change_effect"}:
            if ("default_hrf.t" in compact(call["arguments"]) or ("endpoint" in question and "hrf" in question)):
                return True
    return False


FROZEN = {
    "train-full-v2.jsonl": "024d21fba1d2fe73dc9f8d795552303acaeb099675d9d78df179c082c7e7afcb",
    "transfer-train-roots.jsonl": "5af6e311f19ea24a2ca4d11f5055f6b9ba8c7c1ea0db411541b42f13247b184f",
    "transfer-train-ordinary.jsonl": "71c9a85aa5447985a1e23a7085e5ae17c488dac65bb0f3628f96b4df8ac38690",
    "transfer-train-contrast.jsonl": "05b038400323734c9d5fc9f3ec9748b064bdfcbc4773fe403b6d0b99b753efd7",
    "history-common-coverage.json": "b0d805dc4d0d6ebcd39fcbc1aabe5badd20f42483c00089405f2f12ff7ced316",
    "history-train-specs.jsonl": "beb6443c43aac3b47acec4a39ce31fcebe53d9aa0de47efbb471a17b899f89fc",
    "tokenizer.json": "b6b09a0b4a803ad453063ca4bb49a784540e8120004e2450e025df2b27d41fb2",
    "registry.json": "b006fbacf4fd0f7f423b9178c7f365e98a6cc91bc99d8c9a16277bfd5954d4d0",
}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def rows(path):
    with path.open() as stream:
        for line in stream:
            yield json.loads(line)


def declarations(part):
    return list(rows(ASSETS / f"transfer-train-{part}.jsonl"))


def common_ids(all_ids, coverage, count=320):
    required = {r["case_id"] for r in coverage}
    if not required <= set(all_ids) or len(required) > count:
        raise ValueError("Incomplete shared TRAIN function coverage")
    pool = sorted(set(all_ids) - required)
    random.Random(53321).shuffle(pool)
    eligible = [cid for cid in pool if cid not in EXCLUDED_SHARED]
    result = sorted(required | set(eligible[:count - len(required)]))
    if len(result) != count:
        raise ValueError("Insufficient unique shared TRAIN cases")
    return result


def check():
    for name, expected in FROZEN.items():
        if digest(ASSETS / name) != expected:
            raise ValueError(f"Frozen TRAIN input changed: {name}")
    roots = declarations("roots")
    by_root = {r["root_id"]: r for r in roots}
    arm_rows = {arm: declarations(arm) for arm in ARMS}
    families = defaultdict(list)
    for root in roots:
        families[root["lineage"]].append(root)
    if len(by_root) != len(roots) or any(len(group) != 2 for group in families.values()):
        raise ValueError("Invalid TRAIN root pairing")
    for arm, records in arm_rows.items():
        by_case = {r["case_id"] for r in records}
        groups = defaultdict(list)
        for r in records:
            if r["root_id"] not in by_root or r["slot"] not in (0, 1) or r["meaning"] not in (0, 1):
                raise ValueError("Unknown TRAIN root, slot or meaning")
            groups[r["root_id"]].append(r)
        if len(by_case) != len(records) or set(groups) != set(by_root):
            raise ValueError("TRAIN case identities or roots disagree")
        for pair in groups.values():
            if (len(pair) != 2 or len({r["slot"] for r in pair}) != 2
                    or len({r["continuation"]["question"] for r in pair}) != 2
                    or len({r["meaning"] for r in pair}) != (1 if arm == "ordinary" else 2)):
                raise ValueError(f"Invalid {arm} pair")
    def terms(records):
        return Counter((by_root[r["root_id"]]["lineage"], r["meaning"], r["slot"],
                        compact(r["continuation"])) for r in records)
    if terms(arm_rows["ordinary"]) != terms(arm_rows["contrast"]):
        raise ValueError("TRAIN arms differ in marginal question wording or intended policy")
    observed_roots = sum("observed_prefix_id" in root for root in roots)
    anchors = {case for root in roots for step in root["prefix"] for case in step["ancestor_case_ids"]}
    anchors.update(case for r in arm_rows["ordinary"] for case in r["continuation"].get("source_case_ids", []))
    scientific_targets = set()
    for record in arm_rows["contrast"]:
        policy = record["continuation"]
        scientific_targets.update(policy.get("source_case_ids", []))
        if policy["kind"] == "answer_again":
            step = by_root[record["root_id"]]["prefix"][policy["exchange_index"] - 1]
            scientific_targets.update(step["ancestor_case_ids"])
    return {"scientific_target_source_cases": len(scientific_targets), "roots": len(roots), "lineages": len(families), "source_anchor_cases": len(anchors),
            "observed_roots": observed_roots, "authored_roots": len(roots) - observed_roots,
            "variable_cases_per_arm": len(arm_rows["ordinary"]), "frozen_sha256": dict(FROZEN)}


def arrangements(common, arm_rows):
    positions = defaultdict(dict)
    for seed in SEEDS:
        ordered = list(common)
        rng = random.Random(seed)
        rng.shuffle(ordered)
        slots = list(range(len(ordered)))
        rng.shuffle(slots)
        for arm in ARMS:
            key = f"{arm}-{seed}"
            if len(arm_rows[arm]) != len(ordered):
                raise ValueError("Shared replay must account for exactly half of each arm")
            for index, (cid, slot) in enumerate(zip(ordered, slots)):
                positions[f"shared:{cid}"][key] = 2 * index
                positions[arm_rows[arm][slot]["case_id"]][key] = 2 * index + 1
    return positions


def receipt_source():
    directory = SOURCE / "collection"
    inputs = [directory / "collection.json", SOURCE / "stages/prepare.json", SOURCE / "experiment.json",
              directory / "prefix-receipts.jsonl"]
    if any(path.resolve() != path for path in inputs):
        raise ValueError("Unexpected prior TRAIN source location")
    collection = read(inputs[0])
    stage = read(inputs[1])["result"]
    if stage["collection_receipt"] != collection or stage["complete"] is not True:
        raise ValueError("Prior TRAIN receipt does not match the completed source prepare stage")
    if collection["prefix_receipts"]["path"] != "prefix-receipts.jsonl":
        raise ValueError("Unexpected previous TRAIN receipt path")
    path = directory / "prefix-receipts.jsonl"
    expected = collection["prefix_receipts"]
    fingerprint = collection["fingerprint"]
    original = read(SOURCE / "experiment.json")["assets"]["initial_checkpoint"]
    if (path.resolve() != path or digest(path) != expected["sha256"] or expected["bytes"] != path.stat().st_size
            or fingerprint["checkpoint_step"] != 7205 or fingerprint["checkpoint_sha256"] != original["sha256"]
            or any(digest(ASSETS / name) != fingerprint[key] for key, name in (
                ("registry_sha256", "registry.json"), ("tokenizer_sha256", "tokenizer.json"),
                ("specs_sha256", "history-train-specs.jsonl")))):
        raise ValueError("Prior TRAIN receipt provenance changed")
    if any(r["model_fingerprint"] != fingerprint for r in rows(path)):
        raise ValueError("Saved TRAIN history came from different weights or source")
    return path, expected["sha256"]


def native_rows(receipt_path):
    args = ["node", str(ROOT / "transfer_train.mjs"), str(receipt_path) if receipt_path else "--fixture"]
    with subprocess.Popen(args, cwd=ROOT, stdout=subprocess.PIPE, text=True) as worker:
        try:
            for line in worker.stdout:
                yield json.loads(line)
            if worker.wait():
                raise RuntimeError("Native TRAIN validation failed")
        finally:
            if worker.poll() is None:
                worker.terminate()


def tokenized(row, tokenizer):
    prompt = tokenizer.encode(row["model_input"], add_special_tokens=False).ids
    full = tokenizer.encode(row["model_input"] + row["target"], add_special_tokens=False).ids
    if (full[:len(prompt)] != prompt or len(prompt) + 512 > 32768
            or not 0 < len(full) - len(prompt) <= 512 or full[-1] != 50):
        raise ValueError(f"Invalid TRAIN context/target boundary: {row['case_id']}")
    return {"input_ids": full, "prompt_tokens": len(prompt)}


def note(stats, group, calls, ids, prompt, branch):
    stats["groups"][group] += 1
    stats["branches"][branch] += 1
    stats["arity"][len(calls)] += 1
    stats["tokens"]["prompt"] += prompt
    stats["tokens"]["target"] += len(ids) - prompt
    stats["prompt_lengths_1k"][prompt // 1024] += 1
    for call in calls:
        stats["functions"][call["name"]] += 1
        for field, value in call["arguments"].items():
            stats["arguments"][f"{field}={compact(value)}"] += 1


def build_file(path, receipts, available_bytes):
    from tokenizers import Tokenizer
    from transfer_contract import upgrade_replay

    tokenizer = Tokenizer.from_file(str(ASSETS / "tokenizer.json"))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    replay = ASSETS / "train-full-v2.jsonl"
    coverage = read(ASSETS / "history-common-coverage.json")
    if FROZEN["train-full-v2.jsonl"] != coverage["source_sha256"]:
        raise ValueError("Shared replay source changed")
    all_ids = [row["case_id"] for row in rows(replay)]
    arm_rows = {arm: declarations(arm) for arm in ARMS}
    common = common_ids(all_ids, coverage["functions"], len(arm_rows["ordinary"]))
    positions = arrangements(common, arm_rows)
    wanted, seen = set(common), set()
    stats = {arm: defaultdict(Counter) for arm in ARMS}
    source_calls = set()
    with path.open("wb") as output:
        def emit(case_id, tokens, group, calls, branch, arms):
            if case_id in seen or case_id not in positions:
                raise ValueError(f"Invalid TRAIN draw identity: {case_id}")
            seen.add(case_id)
            data = (compact({"case_id": case_id, "positions": positions[case_id], **tokens}) + "\n").encode()
            if output.tell() + len(data) + 256 * 1024 > min(PREPARE_BYTES, available_bytes):
                raise ValueError("TRAIN preparation exceeds the available private disk budget")
            output.write(data)
            for arm in arms:
                note(stats[arm], group, calls, tokens["input_ids"], tokens["prompt_tokens"], branch)

        for row in rows(replay):
            if row["case_id"] in wanted:
                if generic_interval_conflict(row):
                    raise ValueError(f"Ambiguous legacy HRF supervision in shared replay: {row['case_id']}")
                tokens = upgrade_replay(row, tokenizer)
                emit("shared:" + row["case_id"], {k: tokens[k] for k in ("input_ids", "prompt_tokens")},
                     "shared_replay", row["calls"], row["history_origin"], ARMS)
                source_calls.update((row["case_id"], c["name"]) for c in row["calls"])
        required = {(r["case_id"], r["function"]) for r in coverage["functions"]}
        if not required <= source_calls or "decline_request" not in {fn for _, fn in source_calls}:
            raise ValueError("Shared replay does not actually present all declared scientific calls")
        for row in native_rows(receipts):
            emit(row["case_id"], tokenized(row, tokenizer), row["group"], row["calls"], row["branch"], [row["arm"]])
        if seen != set(positions):
            raise ValueError("Native materialization omitted planned TRAIN draws")
        output.flush()
        os.fsync(output.fileno())
    exposure = {arm: dict(stats[arm]["tokens"]) for arm in ARMS}
    for field in ("prompt", "target"):
        a, b = [exposure[arm][field] for arm in ARMS]
        if abs(a - b) > .05 * max(a, b):
            raise ValueError(f"TRAIN arms differ by more than 5% on {field} token exposure")
    return {"count": len(common) * 2, "updates": len(common) // 2, "schedule_keys":
            {arm: {str(seed): f"{arm}-{seed}" for seed in SEEDS} for arm in ARMS},
            "token_balance": exposure, "presented": stats, "common_cases": common,
            "excluded_shared_conflicts": sorted(EXCLUDED_SHARED),
            "shared_scientific_functions": len({r["function"] for r in coverage["functions"]})}


def prepare(directory, *, available_bytes=None):
    directory = Path(directory).absolute()
    if directory != directory.resolve() or not directory.is_relative_to(ROOT / "runs"):
        raise ValueError("TRAIN materialization requires a private experiment directory inside runs")
    directory.mkdir(exist_ok=True, parents=True)
    path = directory / "train.jsonl"
    manifest = directory / "train-receipt.json"
    partial = directory / ".train.partial"
    pending = directory / ".receipt.pending"
    writing = directory / ".receipt.writing"
    if any(p.is_symlink() or (p.exists() and not p.is_file()) for p in (path, manifest, partial, pending, writing)):
        raise ValueError("Unexpected TRAIN artifact path")
    identity = check()
    source, old_digest = receipt_source()
    identity["observed_receipts_sha256"] = old_digest
    core = ["transfer_train.py", "transfer_train.mjs", "transfer_contract.mjs", "transfer_contract.py"]
    identity["code"] = {name: digest(ROOT / name) for name in core}
    def verify(record, data):
        if (record["identity"] != identity or record["bytes"] != data.stat().st_size
                or record["sha256"] != digest(data)):
            raise ValueError("Saved semantic-transfer TRAIN no longer matches its inputs")
        return {**record, "path": str(path)}
    if manifest.exists():
        return verify(read(manifest), path)
    if pending.exists():
        rec = read(pending)
        result = verify(rec, path if path.exists() else partial)
        if not path.exists():
            partial.replace(path)
        pending.replace(manifest)
        return result
    if path.exists():
        raise ValueError("Unbound TRAIN artifact: refusing to overwrite")
    limit = PREPARE_BYTES if available_bytes is None else available_bytes
    if partial.exists():
        limit += partial.stat().st_size
    result = build_file(partial, source, limit)
    rec = {"schema": "semantic-transfer-train-v1", "identity": identity, **result,
           "bytes": partial.stat().st_size, "sha256": digest(partial)}
    with writing.open("w") as stream:
        stream.write(compact(rec) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    writing.replace(pending)
    partial.replace(path)
    pending.replace(manifest)
    return {**read(manifest), "path": str(path)}
