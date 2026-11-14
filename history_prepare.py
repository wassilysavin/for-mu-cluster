"""Collect TRAIN-only histories once and freeze matched, token-balanced schedules."""

from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
import random
from shutil import disk_usage
import subprocess
from tempfile import TemporaryDirectory
from time import time

from ledger import file_record, publish_json

ROOT = Path(__file__).resolve().parent
SEEDS = (1729, 2718)
DRAW_COUNT = 1572
RESERVE = 512 * 1024 ** 2


def read(path):
    return json.loads(Path(path).read_text())


def compact(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def rows(path):
    with Path(path).open() as stream:
        for line in stream:
            yield json.loads(line)


def require_space(directory, size):
    if disk_usage(directory).free < RESERVE + size:
        raise RuntimeError(f"TRAIN preparation needs {size / 1024**2:.0f} MiB plus its 512 MiB disk reserve")


def train_command(assets, action):
    return ["node", str(ROOT / "history-data.mjs"), action,
            "--specs", str(assets["history_train_specs"]), "--registry", str(assets["registry"]),
            "--runtime", str(ROOT / "evaluation")]


def check(config, assets):
    manifest = read(assets["history_train_manifest"])
    review = read(assets["history_train_review"])
    specs = file_record(assets["history_train_specs"])
    if (review.get("status") != "approved" or review.get("specs_sha256") != specs["sha256"]
            or not review.get("reviewer") or review["reviewer"] == manifest.get("author")):
        raise ValueError("TRAIN specs require a separate, hash-bound semantic review")
    for path, expected in review.get("deployed_code_sha256", {}).items():
        if file_record(ROOT / path)["sha256"] != expected:
            raise ValueError(f"Reviewed TRAIN policy implementation changed: {path}")
    if len(review.get("deployed_code_sha256", {})) != 3:
        raise ValueError("TRAIN review must bind all three deployed policy modules")
    if file_record(assets["history_replay"])["sha256"] != "024d21fba1d2fe73dc9f8d795552303acaeb099675d9d78df179c082c7e7afcb":
        raise ValueError("The original full TRAIN must remain frozen")
    subprocess.run(train_command(assets, "check"), check=True)
    subprocess.run(["node", str(ROOT / "evaluate-known.mjs"), str(assets["registry"]),
                    str(assets["regression_dev"]), "--check", compact(config["regression"])], check=True)


def remaining_requests(specs, receipts, fingerprint):
    prefixes = {spec["prefix_id"]: spec for spec in specs}
    completed = set()
    if receipts.exists():
        with receipts.open() as stream:
            for line in stream:
                if not line.endswith("\n"):
                    break
                record = json.loads(line)
                prefix = record["prefix_id"]
                if (prefix not in prefixes or prefix in completed or record["model_fingerprint"] != fingerprint
                        or len(record["observations"]) != len(prefixes[prefix]["prefix"])):
                    raise ValueError("Stored TRAIN prefix receipt changed or is incomplete")
                completed.add(prefix)
    return sum(len(spec["prefix"]) for prefix, spec in prefixes.items() if prefix not in completed)


def verify_collection(directory, saved):
    for name in ("treatment", "control", "prefix_receipts"):
        if file_record(directory / saved[name]["path"], relative_to=directory) != saved[name]:
            raise ValueError("Frozen collected histories or raw prefix receipts changed")
    result = saved["collection"]
    if (result.get("actual_observed_rows") != 480 or result.get("reconstructed_rows") != 0
            or result.get("rows") != 480 or result.get("control_rows") != 480
            or result["sha256"] != saved["treatment"]["sha256"]
            or result["control_sha256"] != saved["control"]["sha256"]):
        raise ValueError("The TRAIN must preserve all 480 paired, actually observed student histories")


def collect(assets, directory, checkpoint, deadline, budget):
    from model import Runtime
    from tokenizers import Tokenizer

    outcome = directory / "collection.json"
    fingerprint = {"checkpoint_step": 7205, "checkpoint_sha256": file_record(checkpoint)["sha256"],
                   "specs_sha256": file_record(assets["history_train_specs"])["sha256"],
                   "registry_sha256": file_record(assets["registry"])["sha256"],
                   "tokenizer_sha256": file_record(assets["tokenizer"])["sha256"]}
    if outcome.exists():
        saved = read(outcome)
        if saved["fingerprint"] != fingerprint:
            raise ValueError("Collected TRAIN belongs to another source checkpoint or data version")
        verify_collection(directory, saved)
        return saved
    require_space(directory, 400 * 1024 ** 2)
    specs = list(rows(assets["history_train_specs"]))
    requests = remaining_requests(specs, directory / "prefix-receipts.jsonl", fingerprint)
    if requests and budget.remaining_seconds < requests * 15 + 3600:
        raise RuntimeError("Insufficient budget for TRAIN history collection and subsequent verification")
    fingerprint_path = directory / "collection-source.json"
    if fingerprint_path.exists():
        if read(fingerprint_path) != fingerprint:
            raise ValueError("Collection source changed on resume")
    else:
        publish_json(fingerprint_path, fingerprint)
    tokenizer = Tokenizer.from_file(str(assets["tokenizer"]))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    runtime = Runtime(assets["base"], checkpoint, expected_batch_size=4) if requests else None
    started = time()
    generations = 0
    try:
        with TemporaryDirectory(dir=directory) as temporary:
            candidate, control = Path(temporary) / "treatment.jsonl", Path(temporary) / "control.jsonl"
            command = train_command(assets, "collect") + ["--output", str(candidate), "--control-output", str(control),
                "--receipts", str(directory / "prefix-receipts.jsonl"), "--model-fingerprint", str(fingerprint_path)]
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
            result = None
            try:
                for line in process.stdout:
                    budget.check()
                    if time() >= deadline:
                        raise TimeoutError("TRAIN collection reached the experiment time limit")
                    message = json.loads(line)
                    if message["kind"] == "generate":
                        if runtime is None:
                            raise ValueError("Completed TRAIN prefix receipts unexpectedly requested a new model generation")
                        ids = tokenizer.encode(message["prompt"], add_special_tokens=False).ids
                        if len(ids) + 512 > 32768:
                            raise ValueError(f"Full TRAIN prefix exceeds context: {message['case_id']}; no truncation allowed")
                        raw = runtime.generate(ids, max_new_tokens=512, stop_ids={50})
                        reply = {"case_id": message["case_id"], "raw_text": tokenizer.decode(raw, skip_special_tokens=False), "raw_ids": raw}
                        process.stdin.write(compact(reply) + "\n")
                        process.stdin.flush()
                        generations += 1
                    elif message["kind"] == "prefix_complete":
                        print(f"TRAIN histories: {message['completed']}/{message['planned']} prefixes", flush=True)
                        require_space(directory, 300 * 1024 ** 2)
                    elif message["kind"] == "receipt_tail_recovered":
                        print("Recovered incomplete TRAIN receipt tail; complete prefixes are retained", flush=True)
                    elif message["kind"] == "result":
                        result = message
                    else:
                        raise ValueError(f"Unknown TRAIN collection message: {message['kind']}")
                if process.wait() or result is None:
                    raise RuntimeError("TRAIN collection did not complete; saved prefixes remain available")
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                process.stdin.close()
                process.stdout.close()
            for source, name in ((candidate, "treatment"), (control, "control")):
                destination = directory / f"{name}-histories.jsonl"
                if destination.exists():
                    if file_record(destination)["sha256"] != file_record(source)["sha256"]:
                        raise ValueError(f"Completed {name} histories changed during recovery")
                else:
                    os.replace(source, destination)
            saved = {"fingerprint": fingerprint, "collection": result, "seconds_this_attempt": time() - started,
                     "generations_this_attempt": generations,
                     "prefix_receipts": file_record(directory / "prefix-receipts.jsonl", relative_to=directory),
                     **{name: file_record(directory / f"{name}-histories.jsonl", relative_to=directory)
                        for name in ("control", "treatment")}}
            verify_collection(directory, saved)
            publish_json(outcome, saved)
            return saved
    finally:
        if runtime is not None:
            runtime.close()
        del runtime
        import torch
        torch.cuda.empty_cache()


def encode(row, tokenizer, arm):
    prompt = tokenizer.encode(row["model_input"], add_special_tokens=False).ids
    full = tokenizer.encode(row["model_input"] + row["target"], add_special_tokens=False).ids
    if (full[:len(prompt)] != prompt or len(prompt) + 512 > 32768 or full[-1] != 50
            or not 0 < len(full) - len(prompt) <= 512):
        raise ValueError(f"Invalid full token sequence: {row['case_id']}")
    return {"case_id": f"{arm}:{row['case_id']}", "question": row["question"], "calls": row["calls"],
            "family_id": row["family_id"], "group": row["group"], "history_origin": row["history_origin"],
            "reference_targets": row.get("reference_targets", []), "provenance": row["provenance"],
            "input_ids": full, "prompt_tokens": len(prompt), "positions": {}}


def freeze_schedules(assets, directory, collection):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(assets["tokenizer"]))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    require_space(directory, 350 * 1024 ** 2)
    common_cover = read(assets["history_common_coverage"])
    coverage = common_cover if isinstance(common_cover, list) else common_cover["functions"]
    required = {row["case_id"] for row in coverage}
    original, metadata = {}, {}
    with Path(assets["history_replay"]).open("rb") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            row = json.loads(line)
            original[row["case_id"]] = offset
            metadata[row["case_id"]] = {"prompt": row["prompt_tokens"], "target": len(row["input_ids"]) - row["prompt_tokens"]}
    if not required <= set(original):
        raise ValueError("Common replay coverage references absent frozen TRAIN rows")
    rng = random.Random(1729)
    pool = sorted(set(original) - required)
    rng.shuffle(pool)
    common = sorted(required) + pool[:DRAW_COUNT - len(required)]
    if len(common) != DRAW_COUNT:
        raise ValueError("Common replay must contain exactly 1572 draws")
    with TemporaryDirectory(dir=directory) as temporary:
        temporary = Path(temporary)
        tokenized = temporary / "paired.jsonl"
        pair_ids, positions, offsets, groups = [], defaultdict(lambda: defaultdict(list)), {}, defaultdict(list)
        control_iter = rows(directory / collection["control"]["path"])
        treatment_iter = rows(directory / collection["treatment"]["path"])
        with tokenized.open("wb") as stream:
            for index, treatment in enumerate(treatment_iter):
                control = next(control_iter, None)
                if (control is None or treatment["family_id"] != control["family_id"]
                        or treatment["group"] != control["group"] or control["paired_case_id"] != treatment["case_id"]):
                    raise ValueError("Paired control/treatment histories do not align")
                pair = []
                for name, raw in (("control", control), ("treatment", treatment)):
                    encoded = encode(raw, tokenizer, name)
                    case_id = encoded["case_id"]
                    if case_id in offsets:
                        raise ValueError("Duplicate paired TRAIN case")
                    offsets[case_id] = stream.tell()
                    stream.write((compact(encoded) + "\n").encode())
                    metadata[case_id] = {"prompt": encoded["prompt_tokens"], "target": len(encoded["input_ids"]) - encoded["prompt_tokens"]}
                    pair.append(case_id)
                pair_ids.append(pair)
                groups[treatment["family_id"]].append(index)
            if next(control_iter, None) is not None:
                raise ValueError("Unpaired control histories remain")
        families = sorted(groups)
        if len(families) != 192 or len(pair_ids) != 480:
            raise ValueError("The frozen TRAIN must preserve all 192 families and 480 contrast rows")
        draws = []
        while len(draws) < DRAW_COUNT:
            cycle = len(draws) // len(families)
            for family in families:
                choices = groups[family]
                draws.append(choices[cycle % len(choices)])
                if len(draws) == DRAW_COUNT:
                    break
        if set(draws) != set(range(len(pair_ids))):
            raise ValueError("Every authored contrast must appear in the TRAIN schedule")
        schedules, balance = {}, {}
        for seed in SEEDS:
            randomizer = random.Random(seed)
            common_order, pair_order = common.copy(), draws.copy()
            randomizer.shuffle(common_order)
            randomizer.shuffle(pair_order)
            for offset in range(0, len(pair_order), 2):
                if pair_order[offset] != pair_order[offset + 1]:
                    continue
                swap = next((i for i in range(len(pair_order)) if i not in (offset, offset + 1)
                             and pair_order[i] != pair_order[offset]
                             and pair_order[i ^ 1] != pair_order[offset + 1]), None)
                if swap is None:
                    raise ValueError("Cannot form distinct paired examples within each update")
                pair_order[offset + 1], pair_order[swap] = pair_order[swap], pair_order[offset + 1]
            for arm_index, arm in enumerate(("control", "treatment")):
                ordered = []
                for start in range(0, DRAW_COUNT, 2):
                    ordered.extend(common_order[start:start + 2])
                    ordered.extend(pair_ids[index][arm_index] for index in pair_order[start:start + 2])
                key = f"{arm}-{seed}"
                for position, case_id in enumerate(ordered):
                    positions[case_id][key].append(position)
                schedules[key] = {"rows": len(ordered), "sha256": sha256(compact(ordered).encode()).hexdigest()}
                balance[key] = {field: sum(metadata[case_id][field] for case_id in ordered) for field in ("prompt", "target")}
            for field in ("prompt", "target"):
                a, b = balance[f"control-{seed}"][field], balance[f"treatment-{seed}"][field]
                if abs(a - b) / max(a, b) > .05:
                    raise ValueError(f"{field} token totals differ by more than 5%; no cases were truncated or removed")
        candidates = list(positions)
        longest = sorted(candidates, key=lambda case: (-metadata[case]["prompt"], case))[:4]
        largest_targets = sorted(candidates, key=lambda case: (-metadata[case]["target"], case))[:4]
        for index, case_id in enumerate(longest + largest_targets):
            positions[case_id]["pilot"].append(index)
        destination = directory / "train-frozen.jsonl"
        temporary_output = temporary / "train-frozen.jsonl"
        with temporary_output.open("wb") as output, Path(assets["history_replay"]).open("rb") as old, tokenized.open("rb") as new:
            for case_id in sorted(positions):
                source = old if case_id in original else new
                source.seek(original[case_id] if case_id in original else offsets[case_id])
                row = json.loads(source.readline())
                row["positions"] = dict(positions[case_id])
                output.write((compact(row) + "\n").encode())
            output.flush()
            os.fsync(output.fileno())
        if destination.exists():
            if file_record(destination)["sha256"] != file_record(temporary_output)["sha256"]:
                raise ValueError("Prepared TRAIN differs from its existing frozen copy")
        else:
            os.replace(temporary_output, destination)
    return {"train": {**file_record(destination), "path": str(destination.resolve())}, "dataset_id": "actual-history-v3-position-matched",
            "schedules": schedules, "pilot": {"rows": 8}, "token_balance": balance,
            "common_draws": DRAW_COUNT, "paired_draws": DRAW_COUNT, "families": len(families),
            "contrasts": len(pair_ids), "common_domain_coverage": len(coverage),
            "prospective_amendment": "Matched first-biased reconstruction versus original actual ordering, not old-only replay",
            "collection_receipt": collection, "reviewed": True, "complete": True}


def prepare(config, assets, directory, checkpoint, deadline, budget):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / "prepared.json"
    if manifest.exists():
        return read(manifest)
    budget.check()
    collection = collect(assets, directory, Path(checkpoint), deadline, budget)
    budget.check()
    result = freeze_schedules(assets, directory, collection)
    result["independent_review"] = file_record(assets["history_train_review"])
    publish_json(manifest, result)
    return result
