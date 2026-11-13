"""Train, evaluate, and inspect frozen GEM-pRF cluster experiments."""

import argparse
from contextlib import ExitStack
from hashlib import sha256
from importlib.metadata import version
import json
from math import isfinite
import os
from pathlib import Path
import platform
import re
import signal
from shutil import disk_usage
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import monotonic, sleep, time, time_ns
from types import SimpleNamespace

from ledger import SAFE_ID, code_record, execution_lease, file_record, open_run, publish_json, read_result, report


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "experiments.json"
VOLUME_CONFIG = ROOT / "experiments-volume.json"
ARCHITECTURE = "functiongemma270m-lora16-volta-fp32-v2"
PILOT_EXPERIMENT = "pilot-4847-fp32"
VOCABULARY = 262144
CONTEXT = 32768
TARGET_LIMIT = 512
STOP_TOKEN = 50
EXAMPLES_PER_UPDATE = 2
MIB = 1024 * 1024
DISK_RESERVE_BYTES = 512 * MIB
DEV_LOG_BUDGET_BYTES = 32 * MIB
RUN_METADATA_BUDGET_BYTES = 8 * MIB
WORKER_PARENT_FD = None


def read_json(path):
    return json.loads(path.read_text())


def compact(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def checked_asset(reference):
    path = (ROOT / reference["path"]).resolve(strict=True)
    if not path.is_relative_to(ROOT / "assets"):
        raise ValueError("Experiment inputs must be inside assets/")
    record = file_record(path, relative_to=ROOT)
    if record["sha256"] != reference["sha256"]:
        raise ValueError(f"Input hash mismatch for {record['path']}. Check Git LFS and the experiment lock.")
    return path, record


class Schedule:
    def __init__(self, path, expected_sha, arm, count, *, longest=False, probe=False, milestones=()):
        self.path = path
        slots = {}
        probe_rows = []
        seen = set()
        digest = sha256()
        with path.open("rb") as stream:
            while True:
                offset = stream.tell()
                line = stream.readline()
                if not line:
                    break
                digest.update(line)
                row = json.loads(line)
                case_id = row["case_id"]
                if case_id in seen:
                    raise ValueError(f"Duplicate TRAIN case {case_id}")
                seen.add(case_id)
                positions = row["positions"]
                if not isinstance(positions, dict):
                    raise ValueError(f"Invalid TRAIN positions {case_id}")
                if arm not in positions and not probe:
                    continue
                indices = positions.get(arm, [])
                if type(indices) is int:
                    indices = [indices]
                ids = row["input_ids"]
                prompt = row["prompt_tokens"]
                if (not isinstance(indices, list) or (arm in positions and not indices) or not isinstance(ids, list)
                        or type(prompt) is not int
                        or not 0 < prompt < len(ids) <= CONTEXT or len(ids) - prompt > TARGET_LIMIT
                        or any(type(token) is not int or not 0 <= token < VOCABULARY for token in ids)):
                    raise ValueError(f"Invalid TRAIN record {case_id}")
                entry = (case_id, offset, len(ids), sha256(line).digest(), len(ids) - prompt)
                if probe and positions:
                    probe_rows.append(entry)
                for index in indices:
                    if type(index) is not int or not 0 <= index < count or index in slots:
                        raise ValueError(f"Invalid or duplicate TRAIN position for {case_id}")
                    slots[index] = entry
        if digest.hexdigest() != expected_sha:
            raise ValueError("TRAIN changed during schedule verification")
        if len(slots) != count or count < EXAMPLES_PER_UPDATE or count % EXAMPLES_PER_UPDATE:
            raise ValueError(f"Incomplete {arm} schedule, expected {count} positions")
        self.rows = [slots[index] for index in range(count)]
        for offset in range(0, count, EXAMPLES_PER_UPDATE):
            if len({row[0] for row in self.rows[offset:offset + EXAMPLES_PER_UPDATE]}) != EXAMPLES_PER_UPDATE:
                raise ValueError(f"Repeated case within a batch at draw {offset}")
        if milestones:
            boundary = milestones[0] * EXAMPLES_PER_UPDATE
            if count != 2 * boundary or self.rows[:boundary] != self.rows[boundary:]:
                raise ValueError(f"{arm} must repeat its exact first-pass draw order")
        if longest:
            unique = {row[0]: row for row in self.rows}
            self.rows = sorted(unique.values(), key=lambda row: (-row[2], row[0]))[:EXAMPLES_PER_UPDATE]
        if probe:
            largest = sorted(probe_rows, key=lambda row: (-row[2], row[0]))
            target = min(probe_rows, key=lambda row: (-row[4], row[0]))
            companion = next(row for row in largest if row[0] != target[0])
            self.rows = [*largest[:EXAMPLES_PER_UPDATE], target, companion]
        self.sha256 = sha256(compact([row[0] for row in self.rows]).encode()).hexdigest()

    def batch(self, stream, offset, size):
        selected = self.rows[offset:offset + size]
        if len(selected) != size:
            raise ValueError("Incomplete scheduled update")
        batch = []
        for case_id, position, _, expected_line, _ in selected:
            stream.seek(position)
            line = stream.readline()
            if sha256(line).digest() != expected_line:
                raise ValueError(f"TRAIN changed during the run at {case_id}")
            row = json.loads(line)
            ids = row["input_ids"]
            pivot = row["prompt_tokens"]
            batch.append((ids[:pivot], ids[pivot:]))
        return batch, [row[0] for row in selected]


def load_config(suite="legacy"):
    config = read_json(CONFIG)
    if suite == "volume":
        volume = read_json(VOLUME_CONFIG)
        config = {**config, **volume, "assets": {**config["assets"], **volume["assets"]}}
    architecture = config["architecture"]
    if architecture["id"] != ARCHITECTURE or architecture["context_tokens"] != CONTEXT:
        raise ValueError("This runner supports the declared Volta architecture")
    if config["recipe"]["examples_per_update"] != EXAMPLES_PER_UPDATE:
        raise ValueError("The configured batch size differs from the runtime")
    assets = {name: checked_asset(ref) for name, ref in config["assets"].items()}
    return config, assets


def select_experiment(config, assets, experiment_id):
    experiment = config["experiments"][experiment_id]
    training = experiment.get("training")
    schedule = None
    train_record = None
    if training:
        path, record = assets["train"]
        probe = experiment["kind"] == "pilot" and "volume" in config
        schedule = Schedule(path, record["sha256"], training["arm"],
                            training.get("draws", config["train_rows_per_arm"]),
                            longest=experiment["kind"] == "pilot" and not probe, probe=probe,
                            milestones=experiment.get("milestones", ()))
        if len(schedule.rows) != EXAMPLES_PER_UPDATE * training["updates"]:
            raise ValueError("Training budget must match the frozen schedule")
        train_record = dict(record, dataset_id=training["dataset_id"], arm=training["arm"],
                            schedule_sha256=schedule.sha256, rows=len(schedule.rows))
    return experiment, schedule, train_record


def volume_experiments(config, assets):
    volume = config["volume"]
    pilot_id = volume["pilot"]["id"]
    baseline_id = volume["baseline"]
    arm_ids = volume["arms"]
    if len(arm_ids) != 2 or len({pilot_id, baseline_id, *arm_ids}) != 4:
        raise ValueError("Volume comparison requires a pilot, baseline and two distinct arms")
    prepared = {name: select_experiment(config, assets, name) for name in (pilot_id, baseline_id, *arm_ids)}
    pilot, baseline, left, right = (prepared[name][0] for name in prepared)
    updates = left["training"]["updates"]
    milestones = left["milestones"]
    if (pilot["kind"] != "pilot" or baseline["kind"] != "baseline"
            or any(arm["kind"] != "comparison" for arm in (left, right))
            or [left["training"]["arm"], right["training"]["arm"]] != ["narrow", "wide"]
            or pilot["training"]["updates"] != 2 or volume["pilot"]["updates"] != 2
            or milestones != right["milestones"] or len(milestones) != 2
            or type(milestones[0]) is not int or milestones[0] <= 0
            or milestones != [updates // 2, updates] or updates % 2
            or right["training"]["updates"] != updates
            or 2 * updates + pilot["training"]["updates"] > 3246):
        raise ValueError("Volume arms must have two equal passes and fit the update budget")
    return prepared, milestones


def runtime_identity():
    if platform.system() != "Linux":
        raise RuntimeError("Use a Linux GPU allocation for train and test")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Use a Linux GPU allocation for train and test")
    if (torch.cuda.get_device_capability(0) != (7, 0) or torch.version.cuda != "11.8"
            or "sm_70" not in torch.cuda.get_arch_list()):
        raise RuntimeError("The recipe requires a Volta GPU and a CUDA 11.8 PyTorch build with sm_70")
    return {"gpu": torch.cuda.get_device_name(0), "compute_capability": [7, 0],
            "torch": torch.__version__, "cuda": torch.version.cuda, "python": platform.python_version(),
            "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                              text=True).splitlines()[0].strip(),
            "numpy": version("numpy"),
            "safetensors": version("safetensors"), "tokenizers": version("tokenizers"),
            "node": subprocess.check_output(["node", "--version"], text=True).strip()}


def run_identity(experiment_id, config, experiment, assets, train_record, *, runtime=None):
    code = ["run.py", "model.py", "ledger.py", "evaluate.mjs", "requirements.txt", "experiments.json"]
    code.extend(path.relative_to(ROOT).as_posix() for path in (ROOT / "evaluation").glob("*.mjs"))
    suite = config.get("volume")
    if suite:
        code.append(VOLUME_CONFIG.name)
    evaluation = dict(assets["dev"][1], split="DEV700", dataset_id=config["evaluation_id"],
                      registry=assets["registry"][1])
    recipe = dict(config["recipe"], kind=experiment["kind"], training=experiment.get("training"))
    if suite:
        evaluation.update(split="DEV-VOLUME", suite=suite["evaluation"])
        recipe.update(milestones=experiment.get("milestones", []), pilot=suite["pilot"])
    return {"experiment_id": experiment_id, "architecture_id": ARCHITECTURE,
            "architecture": config["architecture"], "code": code_record(ROOT, code),
            "recipe": recipe,
            "runtime": runtime_identity() if runtime is None else runtime, "train": train_record,
            "evaluation": evaluation,
            "base": assets["base"][1], "tokenizer": assets["tokenizer"][1],
            "initial_checkpoint": assets["initial_checkpoint"][1]}


def fsync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def save_latest(runtime, path, lineage):
    with TemporaryDirectory(dir=path.parent) as staging:
        candidate = Path(staging) / path.name
        runtime.save_checkpoint(candidate, lineage=lineage)
        os.replace(candidate, path)
        fsync_directory(path.parent)


def snapshot(latest, step):
    path = latest.with_name(f"checkpoint-step-{step}.safetensors")
    if path.exists():
        if file_record(latest)["sha256"] != file_record(path)["sha256"]:
            raise ValueError(f"Milestone {step} already contains a different checkpoint")
    else:
        os.link(latest, path)
        fsync_directory(path.parent)
    return path


def require_pilot(runs, pilot_run_id, identity):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,99}", pilot_run_id):
        raise ValueError("Invalid pilot run ID")
    path = runs / pilot_run_id
    manifest = read_json(path / "run.json")
    recorded = manifest["identity"]
    pilot = identity["recipe"].get("pilot", {"id": PILOT_EXPERIMENT, "updates": 1})
    keys = ("architecture_id", "base", "initial_checkpoint", "tokenizer", "code", "runtime")
    if any(identity[key] != recorded[key] for key in keys) or recorded["experiment_id"] != pilot["id"]:
        raise ValueError("The pilot must use this code, checkpoint and runtime")
    if recorded["train"]["sha256"] != identity["train"]["sha256"]:
        raise ValueError("The pilot must use this frozen TRAIN source")
    checkpoint = file_record(path / "checkpoint.safetensors")
    verified = False
    measurements = {}
    parity_steps = set()
    with (path / "events.jsonl").open() as stream:
        for line in stream:
            if not line.endswith("\n"):
                break
            event = json.loads(line)
            if event["event"] == "update":
                measurement = event["details"]
                measurements[measurement["step"]] = measurement
            if event["event"] == "parallel_parity_verified":
                parity_steps.add(event["details"]["step"])
            if event["event"] == "pilot_reload_verified":
                verified = (event["details"]["checkpoint"] == checkpoint
                            and event["details"]["step"] == identity["recipe"]["initial_step"] + pilot["updates"])
    if not verified:
        raise ValueError("Complete the pilot save and reload before training an experimental arm")
    if identity.get("execution"):
        expected = {identity["recipe"]["initial_step"] + i + 1 for i in range(pilot["updates"])}
        if recorded.get("execution") != identity["execution"] or not expected <= parity_steps:
            raise ValueError("The GPU pair must pass both sequential-versus-parallel pilot updates")
    return list(measurements.values())


def check_deadline(deadline):
    if WORKER_PARENT_FD is not None:
        try:
            if os.read(WORKER_PARENT_FD, 1) == b"":
                raise RuntimeError("Comparison coordinator disconnected; stopping its worker")
        except BlockingIOError:
            pass
    if deadline is not None and time() >= deadline:
        raise RuntimeError("Comparison time budget exhausted. Saved checkpoints and completed evaluations remain available.")


def require_run_space(path, additional_bytes):
    free = disk_usage(path).free
    required = DISK_RESERVE_BYTES + additional_bytes
    if free < required:
        raise RuntimeError(
            f"Run filesystem has {free / MIB:.0f} MiB free; "
            f"this stage needs {(required + MIB - 1) // MIB} MiB including a 512 MiB reserve."
        )


def checkpoint_budget(assets):
    return assets["initial_checkpoint"][1]["bytes"] + MIB


def verify_frozen_inputs(identity, assets, names):
    for entry in identity["code"]["files"]:
        if file_record(ROOT / entry["path"], relative_to=ROOT) != entry:
            raise ValueError(f"Experiment source changed: {entry['path']}")
    for name in names:
        path, record = assets[name]
        if file_record(path, relative_to=ROOT) != record:
            raise ValueError(f"Experiment input changed: {record['path']}")


def make_runtime(assets, checkpoint, devices=(), *, paired=False):
    from model import Runtime

    if paired:
        from model import PairedRuntime
        return PairedRuntime(assets["base"][0], checkpoint, devices=[f"cuda:{index}" for index in devices])
    options = {"device": f"cuda:{devices[0]}"} if devices else {}
    return Runtime(assets["base"][0], checkpoint, **options)


def verify_parallel_update(parallel, sequential, loss, reference_loss):
    import torch

    if parallel.step != sequential.step:
        raise ValueError("Parallel pilot step differs from sequential execution")
    torch.testing.assert_close(torch.tensor(loss), torch.tensor(reference_loss), rtol=1e-5, atol=1e-7)
    for field in ("parameters", "rows", "cols"):
        actual, expected = getattr(parallel, field), getattr(sequential, field)
        if actual.keys() != expected.keys():
            raise ValueError(f"Parallel pilot {field} names differ")
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key], rtol=1e-5,
                                       atol=1e-8 if field == "parameters" else 0,
                                       msg=lambda message: f"Parallel pilot {field}/{key}: {message}")


def train(args, identity, experiment, assets, schedule, *, until=None, lease_held=False, deadline=None):
    import torch
    from model import LAYERS, LEARNING_RATE, LOSS_SCALE

    recipe = identity["recipe"]
    if (schedule is None or recipe["learning_rate"] != LEARNING_RATE or recipe["loss_scale"] != LOSS_SCALE
            or identity["architecture"]["layers"] != LAYERS):
        raise ValueError("The experiment must declare the runtime's training recipe")
    if experiment["kind"] == "comparison":
        require_pilot(args.runs, args.pilot_run_id, identity)
    milestones = experiment.get("milestones", ())
    updates = experiment["training"]["updates"]
    limit = updates if until is None else until
    if until is not None and until not in milestones:
        raise ValueError("Training can pause only at a declared milestone")
    check_deadline(deadline)
    if lease_held:
        verify_frozen_inputs(identity, assets, ("base", "initial_checkpoint", "train"))
    with open_run(args.runs, args.run_id, identity, resume=args.resume, lease_held=lease_held) as run, ExitStack() as resources:
        initial = assets["initial_checkpoint"][0]
        latest = run.path / "checkpoint.safetensors"
        restart_pilot = experiment["kind"] == "pilot" and getattr(args, "restart_pilot", False)
        checkpoint = latest if args.resume and latest.exists() and not restart_pilot else initial
        if restart_pilot and latest.exists():
            run.event("unverified_pilot_restarted", checkpoint=file_record(latest))
        lineage = {"identity_sha256": run.identity_sha256,
                   "initial_checkpoint_sha256": assets["initial_checkpoint"][1]["sha256"]}
        devices = getattr(args, "devices", ())
        runtime = make_runtime(assets, checkpoint, devices, paired=len(devices) == 2)
        resources.callback(getattr(runtime, "close", lambda: None))
        reference = None
        if len(devices) == 2 and experiment["kind"] == "pilot":
            reference = make_runtime(assets, checkpoint, devices)
            resources.callback(reference.close)
        if runtime.tokenizer_sha256 != assets["tokenizer"][1]["sha256"]:
            raise ValueError("Checkpoint tokenizer differs from the experiment")
        if checkpoint == latest and runtime.lineage != lineage:
            raise ValueError("The saved checkpoint belongs to another run")
        initial_step = identity["recipe"]["initial_step"]
        done = runtime.step - initial_step
        save_every = identity["recipe"]["checkpoint_interval"]
        if type(save_every) is not int or save_every < 1:
            raise ValueError("Checkpoint interval must be positive")
        if not 0 <= done <= updates or (checkpoint == initial and done != 0):
            raise ValueError("Checkpoint step is outside the scheduled interval")
        if done > limit:
            raise ValueError("Cannot retrain an earlier milestone from a later cursor")
        run.event("checkpoint_loaded", step=runtime.step, file=checkpoint.name)
        if done in milestones:
            snapshot(latest, runtime.step)
        with schedule.path.open("rb") as stream:
            for update in range(done, limit):
                check_deadline(deadline)
                batch, case_ids = schedule.batch(stream, EXAMPLES_PER_UPDATE * update, EXAMPLES_PER_UPDATE)
                if devices:
                    runtime.synchronize()
                    runtime.reset_peak_memory_stats()
                else:
                    torch.cuda.reset_peak_memory_stats()
                began = monotonic()
                try:
                    loss = runtime.train_step(batch)
                except FloatingPointError as error:
                    raise FloatingPointError(f"{', '.join(case_ids)}: {error}") from error
                if devices:
                    runtime.synchronize()
                elapsed = monotonic() - began
                peaks = runtime.peak_memory_stats() if devices else {}
                if reference is not None:
                    reference_loss = reference.train_step(batch)
                    verify_parallel_update(runtime, reference, loss, reference_loss)
                    run.event("parallel_parity_verified", step=runtime.step, rtol=1e-5,
                              parameter_atol=1e-8, optimizer_atol=0)
                if runtime.step != initial_step + update + 1:
                    raise RuntimeError("Runtime cursor did not advance by one update")
                at_milestone = update + 1 in milestones
                checkpoint_saved = (update + 1) % save_every == 0 or update + 1 == limit or at_milestone
                if checkpoint_saved:
                    if lease_held:
                        require_run_space(run.path, (2 if devices else 1) * (checkpoint_budget(assets) + RUN_METADATA_BUDGET_BYTES))
                    save_latest(runtime, latest, lineage)
                    if at_milestone:
                        snapshot(latest, runtime.step)
                run.event("update", step=runtime.step, loss=loss, seconds=elapsed, case_ids=case_ids,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          checkpoint_saved=checkpoint_saved, **({"device_peak_bytes": peaks} if peaks else {}))
                prefix = f"[{args.run_id}] " if devices else ""
                print(f"{prefix}{runtime.step}  loss={loss:.6f}  {elapsed:.2f}s", flush=True)
        if experiment["kind"] == "pilot":
            resources.close()
            del runtime, reference
            torch.cuda.empty_cache()
            restored = make_runtime(assets, latest, devices)
            resources.callback(getattr(restored, "close", lambda: None))
            if restored.step != initial_step + updates or restored.lineage != lineage:
                raise ValueError("Pilot checkpoint reload differs from the saved run")
            run.event("pilot_reload_verified", step=restored.step, checkpoint=file_record(latest))


def evaluate_model(runtime, tokenizer_path, registry_path, dev_path, observations, *, suite=None, deadline=None):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    if tokenizer.decode([STOP_TOKEN], skip_special_tokens=False) != "<start_function_response>":
        raise ValueError("Tokenizer terminal token differs")
    command = ["node", str(ROOT / "evaluate.mjs"), str(registry_path), str(dev_path), str(observations)]
    expected = 700 if suite is None else suite["cases"]
    if suite is not None:
        command.extend((compact(suite), str(DISK_RESERVE_BYTES + RUN_METADATA_BUDGET_BYTES)))
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          text=True, encoding="utf-8") as process:
        score = None
        try:
            for line in process.stdout:
                message = json.loads(line)
                if message["kind"] == "result":
                    score = message["score"]
                    continue
                if score is not None or message["kind"] != "generate":
                    raise ValueError("Unexpected evaluation request")
                check_deadline(deadline)
                tokens = tokenizer.encode(message["prompt"], add_special_tokens=False).ids
                if not 0 < len(tokens) <= CONTEXT - TARGET_LIMIT:
                    raise ValueError(f"Complete prompt does not fit for {message['case_id']}")
                raw_ids = runtime.generate(tokens, max_new_tokens=TARGET_LIMIT, stop_ids={STOP_TOKEN})
                raw_text = tokenizer.decode(raw_ids, skip_special_tokens=False)
                process.stdin.write(compact({"raw_ids": raw_ids, "raw_text": raw_text}) + "\n")
                process.stdin.flush()
            if process.wait() != 0 or score is None or score["aborted"] or score["observed_records"] != expected:
                raise RuntimeError("The development evaluation did not complete. Inspect the observations and run log.")
            return score
        except BaseException:
            process.kill()
            raise


def test(args, identity, experiment, assets, *, at=None, lease_held=False, deadline=None):
    baseline = experiment["kind"] == "baseline"
    if experiment["kind"] == "pilot" or (at is not None and at not in experiment.get("milestones", ())):
        raise ValueError("Evaluation requires a reference or declared training checkpoint")
    suite = identity["evaluation"].get("suite")
    initial_step = identity["recipe"]["initial_step"]
    expected_step = initial_step + (0 if baseline else at if at is not None else experiment["training"]["updates"])
    checkpoint_name = f"checkpoint-step-{expected_step}.safetensors" if at is not None else "checkpoint.safetensors"
    result_name = f"result-step-{expected_step}.json" if at is not None else "result.json"
    resume = not baseline or (args.runs / args.run_id / "run.json").exists()
    check_deadline(deadline)
    if suite is not None:
        require_run_space(args.runs, DEV_LOG_BUDGET_BYTES + RUN_METADATA_BUDGET_BYTES)
        verify_frozen_inputs(identity, assets, ("base", "initial_checkpoint", "tokenizer", "registry", "dev"))
    stage = f"evaluate-{expected_step}" if at is not None else "evaluate"
    with open_run(args.runs, args.run_id, identity, resume=resume, stage=stage, lease_held=lease_held) as run:
        if (run.path / result_name).exists():
            raise ValueError("This checkpoint already has a completed evaluation")
        checkpoint = assets["initial_checkpoint"][0] if baseline else run.path / checkpoint_name
        checkpoint_record = file_record(checkpoint)
        runtime = make_runtime(assets, checkpoint, getattr(args, "devices", ()))
        if runtime.tokenizer_sha256 != assets["tokenizer"][1]["sha256"]:
            raise ValueError("Checkpoint tokenizer differs from the experiment")
        if not baseline:
            if runtime.lineage.get("identity_sha256") != run.identity_sha256:
                raise ValueError("The saved checkpoint belongs to another run")
        if runtime.step != expected_step:
            raise ValueError(f"Expected completed checkpoint at step {expected_step}, found {runtime.step}")
        prefix = "dev700" if suite is None else f"dev-volume-{expected_step}"
        observations = run.path / f"{prefix}-{time_ns()}.jsonl"
        run.event("evaluation_started", step=runtime.step, observations=observations.name)
        began = monotonic()
        score = evaluate_model(runtime, assets["tokenizer"][0], assets["registry"][0], assets["dev"][0],
                               observations, suite=suite, deadline=deadline)
        if suite is not None:
            verify_frozen_inputs(identity, assets, ("tokenizer", "registry", "dev"))
            require_run_space(run.path, RUN_METADATA_BUDGET_BYTES)
        if file_record(checkpoint) != checkpoint_record:
            raise ValueError("Checkpoint changed during evaluation")
        run.write_result(score["summary"], checkpoint_record, name=result_name, step=runtime.step,
                         generated_cases=score["generated_cases"], observed_records=score["observed_records"],
                         observations=file_record(observations),
                         **({"seconds": monotonic() - began} if suite is not None else {}))
        summary = score["summary"]
        if identity.get("execution"):
            summary = {"run_id": args.run_id, **summary}
        print(json.dumps(summary, indent=2), flush=True)


def check_evaluation(config, assets):
    command = ["node", str(ROOT / "evaluate.mjs"), str(assets["registry"][0]), str(assets["dev"][0]), "--check"]
    if "volume" in config:
        command.append(compact(config["volume"]["evaluation"]))
    subprocess.run(command, check=True)


def check(suite="legacy"):
    config, assets = load_config(suite)
    prepared = (volume_experiments(config, assets)[0] if suite == "volume"
                else {name: select_experiment(config, assets, name) for name in config["experiments"]})
    for experiment_id, (_, schedule, record) in prepared.items():
        if schedule:
            print(experiment_id, len(schedule.rows), record["schedule_sha256"], flush=True)
    check_evaluation(config, assets)


def comparison_deadline(directory, hours, identities):
    binding = sha256(compact(identities).encode()).hexdigest()
    path = directory / "comparison.json"
    if path.exists():
        record = read_json(path)
        if record["identity_sha256"] != binding or (hours is not None and hours != record["hours"]):
            raise ValueError("Comparison identity or time budget changed. Use a fresh comparison ID.")
    else:
        hours = 36 if hours is None else hours
        if not isfinite(hours) or not 0 < hours <= 36:
            raise ValueError("Comparison budget must be positive and at most 36 hours")
        record = {"identity_sha256": binding, "hours": hours, "started_at_unix": time()}
        directory.mkdir(exist_ok=True)
        publish_json(path, record)
    return record["started_at_unix"] + 3600 * record["hours"]


def volume_evaluation(job, identity, experiment, assets, *, at=None, deadline):
    initial_step = identity["recipe"]["initial_step"]
    path = job.runs / job.run_id
    step = initial_step if at is None else initial_step + at
    name = "result.json" if at is None else f"result-step-{step}.json"
    result = read_result(path, name, identity=identity)
    if result is None:
        test(job, identity, experiment, assets, at=at, lease_held=True, deadline=deadline)
        result = read_result(path, name, identity=identity)
    details = result["details"]
    checkpoint = assets["initial_checkpoint"][0] if at is None else path / f"checkpoint-step-{step}.safetensors"
    observations = path / details["observations"]["path"]
    if (observations.parent != path or result["checkpoint"] != file_record(checkpoint)
            or details["step"] != step or details["observed_records"] != identity["evaluation"]["suite"]["cases"]
            or result["metrics"]["missing"] != 0 or details["observations"] != file_record(observations)):
        raise ValueError(f"Incomplete or changed evaluation for {job.run_id} at {step}")
    return result


def observed_outcomes(runs, result):
    path = runs / result["run_id"] / result["details"]["observations"]["path"]
    outcomes = {}
    with path.open() as stream:
        for line in stream:
            record = json.loads(line)
            case = record["case_id"]
            if case in outcomes:
                raise ValueError(f"Repeated observed case {case}")
            outcome = record["result"]
            outcomes[case] = outcome["success"] and outcome["critical_status"] == "none_observed"
    if len(outcomes) != result["details"]["observed_records"]:
        raise ValueError("Observed cases do not match the evaluation receipt")
    return outcomes


def paired_outcomes(plan, narrow, wide):
    def wins(left, right):
        return {"wide_only": sum(r and not l for l, r in zip(left, right)),
                "narrow_only": sum(l and not r for l, r in zip(left, right))}

    if set(narrow) != set(wide) or set(narrow) != {row["case_id"] for row in plan}:
        raise ValueError("The comparison evaluated different development questions")
    result = {}
    dialogues = {}
    for cohort in ("supported", "outside_scope"):
        ids = [row["case_id"] for row in plan if row["cohort"] == cohort]
        result[cohort] = wins([narrow[key] for key in ids], [wide[key] for key in ids])
    for row in plan:
        if row.get("dialogue_id"):
            dialogues.setdefault(row["dialogue_id"], []).append(row["case_id"])
    result["whole_dialogues"] = wins(
        [all(narrow[key] for key in ids) for ids in dialogues.values()],
        [all(wide[key] for key in ids) for ids in dialogues.values()],
    )
    return result


def saved_updates(path, initial_step):
    events = path / "events.jsonl"
    if not (path / "checkpoint.safetensors").is_file() or not events.is_file():
        return 0
    step = initial_step
    with events.open("rb") as stream:
        for line in stream:
            if not line.endswith(b"\n"):
                break
            event = json.loads(line)
            if event["event"] == "update" and event["details"]["checkpoint_saved"]:
                step = max(step, event["details"]["step"])
    return step - initial_step


def comparison_space(runs, pilot_id, baseline_id, arm_ids, steps, assets, *, parallel=False):
    pilot_ids = [pilot_id] if isinstance(pilot_id, str) else pilot_id
    checkpoints = sum(not (runs / name / "checkpoint.safetensors").is_file() for name in pilot_ids)
    evaluations = int(not (runs / baseline_id / "result.json").is_file())
    for arm in arm_ids:
        for step in steps:
            checkpoints += not (runs / arm / f"checkpoint-step-{step}.safetensors").is_file()
            evaluations += not (runs / arm / f"result-step-{step}.json").is_file()
    return ((checkpoints + (2 if parallel else 1)) * checkpoint_budget(assets)
            + evaluations * DEV_LOG_BUDGET_BYTES + RUN_METADATA_BUDGET_BYTES)


def admit_round(runs, run_ids, initial_step, milestone, pilot, baseline, deadline, *, parallel=False):
    pending = []
    for name in run_ids:
        path = runs / name
        step = initial_step + milestone
        if (path / f"result-step-{step}.json").is_file():
            continue
        updates = 0
        if not (path / f"checkpoint-step-{step}.safetensors").is_file():
            updates = max(0, milestone - saved_updates(path, initial_step))
        pending.append(updates)
    if not pending:
        return
    step_seconds = max(item["seconds"] for item in pilot)
    evaluate_seconds = baseline["details"]["seconds"]
    durations = [1.15 * (updates * step_seconds + evaluate_seconds) + 120 for updates in pending]
    estimated = max(durations) if parallel else sum(durations)
    if parallel:
        print(f"Parallel round budget estimate: {estimated / 3600:.1f} h remaining", flush=True)
    if time() + estimated >= deadline:
        raise RuntimeError(f"Remaining time cannot fit both arms and their DEV checks (estimate {estimated / 3600:.1f} h)")


def publish_comparison(runs, directory, milestone, baseline, arms, plan):
    narrow, wide = (observed_outcomes(runs, result) for result in arms)
    if narrow.keys() != observed_outcomes(runs, baseline).keys():
        raise ValueError("The comparison evaluated different development questions")
    summary = {"updates_per_arm": milestone, "baseline": baseline,
               "narrow": arms[0], "wide": arms[1], "paired": paired_outcomes(plan, narrow, wide),
               "quality_status": "automatic_screening_pending_independent_review"}
    path = directory / f"comparison-{milestone}.json"
    if path.exists():
        if read_json(path) != summary:
            raise ValueError(f"Previously saved comparison changed at {milestone} updates")
    else:
        require_run_space(directory, RUN_METADATA_BUDGET_BYTES)
        publish_json(path, summary)
    print(f"Paired DEV comparison saved to {path}", flush=True)


def volume_compare(args):
    if not SAFE_ID.fullmatch(args.run_id) or len(args.run_id) > 80:
        raise ValueError("Invalid comparison ID")
    config, assets = load_config("volume")
    prepared, milestones = volume_experiments(config, assets)
    volume = config["volume"]
    pilot_id = volume["pilot"]["id"]
    baseline_id = volume["baseline"]
    arm_ids = volume["arms"]
    pilot_run = f"{args.run_id}-{pilot_id}"
    check_evaluation(config, assets)
    runtime = runtime_identity()
    identities = {name: run_identity(name, config, experiment, assets, record, runtime=runtime)
                  for name, (experiment, _, record) in prepared.items()}
    with execution_lease(args.runs):
        directory = args.runs / args.run_id
        rounds = milestones[:args.rounds]
        initial_step = config["recipe"]["initial_step"]
        run_ids = [f"{args.run_id}-{name}" for name in arm_ids]

        def check_space(pending_rounds):
            require_run_space(args.runs, comparison_space(
                args.runs, pilot_run, f"{args.run_id}-{baseline_id}", run_ids,
                [initial_step + milestone for milestone in pending_rounds], assets,
            ))

        if not (directory / f"comparison-{rounds[-1]}.json").is_file():
            check_space(rounds)
        deadline = comparison_deadline(directory, args.hours, identities)

        def job(name):
            run_id = f"{args.run_id}-{name}"
            return SimpleNamespace(runs=args.runs, run_id=run_id, pilot_run_id=pilot_run,
                                   resume=(args.runs / run_id / "run.json").is_file())

        pilot, pilot_schedule, _ = prepared[pilot_id]
        try:
            measurements = require_pilot(args.runs, pilot_run, identities[arm_ids[0]])
        except (FileNotFoundError, ValueError):
            train(job(pilot_id), identities[pilot_id], pilot, assets, pilot_schedule,
                  lease_held=True, deadline=deadline)
            measurements = require_pilot(args.runs, pilot_run, identities[arm_ids[0]])
        if (len(measurements) != volume["pilot"]["updates"]
                or any(not isfinite(item["seconds"]) or item["seconds"] <= 0 or item["peak_allocated_bytes"] <= 0
                       for item in measurements)):
            raise ValueError("Pilot is missing update timing or memory evidence")
        baseline = volume_evaluation(job(baseline_id), identities[baseline_id], prepared[baseline_id][0],
                                     assets, deadline=deadline)
        with assets["dev"][0].open() as stream:
            plan = [json.loads(line) for line in stream]
        for milestone in rounds:
            if not (directory / f"comparison-{milestone}.json").is_file():
                check_space([milestone])
            admit_round(args.runs, run_ids, initial_step, milestone, measurements, baseline, deadline)
            arms = []
            for name in arm_ids:
                experiment, schedule, _ = prepared[name]
                run_job = job(name)
                step = initial_step + milestone
                checkpoint = args.runs / run_job.run_id / f"checkpoint-step-{step}.safetensors"
                if not checkpoint.exists():
                    train(run_job, identities[name], experiment, assets, schedule, until=milestone,
                          lease_held=True, deadline=deadline)
                arms.append(volume_evaluation(run_job, identities[name], experiment, assets, at=milestone,
                                              deadline=deadline))
            publish_comparison(args.runs, directory, milestone, baseline, arms, plan)



def parallel_context():
    import torch

    config, assets = load_config("volume")
    prepared, milestones = volume_experiments(config, assets)
    runtime = runtime_identity()
    if torch.cuda.device_count() < 4:
        raise RuntimeError("--gpus 4 needs four visible V100 GPUs in the cluster allocation")
    visible = []
    for index in range(4):
        info = torch.cuda.get_device_properties(index)
        if (info.major, info.minor) != (7, 0):
            raise RuntimeError(f"Visible GPU {index} is not Volta")
        visible.append({"index": index, "name": info.name, "memory_bytes": info.total_memory,
                        "uuid": str(getattr(info, "uuid", "unavailable"))})
    uuids = [entry["uuid"] for entry in visible if entry["uuid"] != "unavailable"]
    if len(uuids) != len(set(uuids)):
        raise RuntimeError("The four visible GPU indices must identify distinct physical devices")
    runtime.update(devices=visible, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    arms = config["volume"]["arms"]
    pairs = {arm: (2 * i, 2 * i + 1) for i, arm in enumerate(arms)}
    pilot = config["volume"]["pilot"]["id"]
    baseline = config["volume"]["baseline"]
    entries = {baseline: (baseline, (0,))}
    for arm, devices in pairs.items():
        entries.update({arm: (arm, devices), f"{pilot}-{arm}": (pilot, devices)})
    identities = {}
    for name, (experiment_id, devices) in entries.items():
        experiment, _, record = prepared[experiment_id]
        identity = run_identity(experiment_id, config, experiment, assets, record, runtime=runtime)
        identity["execution"] = {"mode": "paired-microbatches-v1", "devices": list(devices),
                                 "gradient_reduction": "sum", "examples_per_update": 2}
        identities[name] = identity
    return config, assets, prepared, milestones, identities, pairs


def parallel_job(args, name, devices):
    pilot = f"{args.run_id}-pilot-{name}" if name in ("narrow", "wide") else None
    run_id = f"{args.run_id}-{name}"
    return SimpleNamespace(runs=args.runs, run_id=run_id, devices=devices, pilot_run_id=pilot,
                           resume=(args.runs / run_id / "run.json").is_file())


def verify_worker_lease(runs, lease_fd):
    held = os.fstat(lease_fd)
    named = os.stat(runs / ".execution.lock", follow_symlinks=False)
    if (held.st_dev, held.st_ino) != (named.st_dev, named.st_ino):
        raise ValueError("Worker did not inherit this comparison's execution lease")
    try:
        with execution_lease(runs):
            pass
    except RuntimeError:
        return
    raise RuntimeError("Worker's inherited execution lease is not locked")


def parallel_worker(args):
    global WORKER_PARENT_FD
    import torch

    verify_worker_lease(args.runs, args.lease_fd)
    WORKER_PARENT_FD = args.parent_fd
    os.set_blocking(WORKER_PARENT_FD, False)
    config, assets, prepared, milestones, identities, pairs = parallel_context()
    directory = args.runs / args.run_id
    if not (directory / "comparison.json").is_file():
        raise ValueError("Worker requires an existing comparison manifest")
    deadline = comparison_deadline(directory, None, identities)
    check_deadline(deadline)
    devices = pairs[args.worker]
    torch.cuda.set_device(devices[0])
    if args.worker_stage == "pilot":
        name = f"pilot-{args.worker}"
        pilot_id = config["volume"]["pilot"]["id"]
        experiment, schedule, _ = prepared[pilot_id]
        job = parallel_job(args, name, devices)
        try:
            require_pilot(args.runs, job.run_id, identities[args.worker])
        except (FileNotFoundError, ValueError):
            job.restart_pilot = True
            train(job, identities[name], experiment, assets, schedule, lease_held=True, deadline=deadline)
        require_pilot(args.runs, job.run_id, identities[args.worker])
    else:
        if args.until not in milestones:
            raise ValueError("Worker requires a declared training milestone")
        experiment, schedule, _ = prepared[args.worker]
        job = parallel_job(args, args.worker, devices)
        step = config["recipe"]["initial_step"] + args.until
        checkpoint = args.runs / job.run_id / f"checkpoint-step-{step}.safetensors"
        if not checkpoint.exists():
            train(job, identities[args.worker], experiment, assets, schedule,
                  until=args.until, lease_held=True, deadline=deadline)
        volume_evaluation(job, identities[args.worker], experiment, assets, at=args.until, deadline=deadline)


def launch_workers(args, arms, milestone, lease_fd, deadline, *, stage="arm"):
    read_fd, write_fd = os.pipe()
    processes = []
    try:
        for name in arms:
            command = [sys.executable, "-u", str(ROOT / "run.py"), "--runs", str(args.runs),
                       "compare", "--gpus", "4", "--run-id", args.run_id,
                       "--worker", name, "--worker-stage", stage, "--until", str(milestone),
                       "--lease-fd", str(lease_fd), "--parent-fd", str(read_fd)]
            processes.append((name, subprocess.Popen(command, pass_fds=(lease_fd, read_fd),
                                                      start_new_session=True)))
        while True:
            check_deadline(deadline)
            statuses = [(name, process.poll()) for name, process in processes]
            failed = [(name, status) for name, status in statuses if status not in (None, 0)]
            if failed:
                raise RuntimeError(f"Parallel {stage} worker failed: {failed}; saved checkpoints remain available")
            if all(status == 0 for _, status in statuses):
                return
            sleep(0.2)
    finally:
        os.close(write_fd)
        os.close(read_fd)
        for _, process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for _, process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()


def volume_compare_parallel(args):
    import torch

    if not SAFE_ID.fullmatch(args.run_id) or len(args.run_id) > 75:
        raise ValueError("Invalid parallel comparison ID")
    with execution_lease(args.runs) as lease_fd:
        config, assets, prepared, milestones, identities, pairs = parallel_context()
        check_evaluation(config, assets)
        directory = args.runs / args.run_id
        rounds = milestones[:args.rounds]
        initial = config["recipe"]["initial_step"]
        pilots = [f"{args.run_id}-pilot-{arm}" for arm in pairs]
        arm_ids = [f"{args.run_id}-{arm}" for arm in pairs]
        baseline_name = config["volume"]["baseline"]
        baseline_job = parallel_job(args, baseline_name, (0,))
        if not (directory / f"comparison-{rounds[-1]}.json").is_file():
            require_run_space(args.runs, comparison_space(args.runs, pilots, baseline_job.run_id, arm_ids,
                              [initial + milestone for milestone in rounds], assets, parallel=True))
        deadline = comparison_deadline(directory, args.hours, identities)
        pending_pilots = []
        for arm, pilot in zip(pairs, pilots):
            try:
                require_pilot(args.runs, pilot, identities[arm])
            except (FileNotFoundError, ValueError):
                pending_pilots.append(arm)
        if pending_pilots:
            print("Pilot: GPUs 0+1 and 2+3; each pair must match sequential updates", flush=True)
            launch_workers(args, pending_pilots, 0, lease_fd, deadline, stage="pilot")
        measurements = [item for arm, pilot in zip(pairs, pilots)
                        for item in require_pilot(args.runs, pilot, identities[arm])]
        if (len(measurements) != 4 or any(not isfinite(item["seconds"]) or item["seconds"] <= 0
                                       or item["peak_allocated_bytes"] <= 0 for item in measurements)):
            raise ValueError("Both GPU pairs need complete pilot timing and memory evidence")
        print("Baseline: evaluating 155 DEV cases on GPU 0", flush=True)
        baseline = volume_evaluation(baseline_job, identities[baseline_name], prepared[baseline_name][0],
                                     assets, deadline=deadline)
        torch.cuda.empty_cache()
        with assets["dev"][0].open() as stream:
            plan = [json.loads(line) for line in stream]
        for milestone in rounds:
            if not (directory / f"comparison-{milestone}.json").is_file():
                require_run_space(args.runs, comparison_space(args.runs, pilots, baseline_job.run_id,
                                  arm_ids, [initial + milestone], assets, parallel=True))
            admit_round(args.runs, arm_ids, initial, milestone, measurements, baseline, deadline, parallel=True)
            pending = [arm for arm in pairs if not (args.runs / f"{args.run_id}-{arm}"
                       / f"result-step-{initial + milestone}.json").is_file()]
            if pending:
                print(f"Training arms to +{milestone}: narrow on 0+1, wide on 2+3", flush=True)
                launch_workers(args, pending, milestone, lease_fd, deadline)
            results = [volume_evaluation(parallel_job(args, arm, devices), identities[arm], prepared[arm][0],
                                        assets, at=milestone, deadline=deadline) for arm, devices in pairs.items()]
            publish_comparison(args.runs, directory, milestone, baseline, results, plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs", help="Shared run directory")
    parser.add_argument("--suite", choices=("legacy", "volume"), default="legacy")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check")
    commands.add_parser("report")
    comparison = commands.add_parser("compare")
    comparison.add_argument("--run-id")
    comparison.add_argument("--gpus", type=int, choices=(1, 4), default=1)
    comparison.add_argument("--rounds", type=int, choices=(1, 2), default=1)
    comparison.add_argument("--hours", type=float, help="Fixed time budget for the comparison, up to 36 hours")
    comparison.add_argument("--worker", choices=("narrow", "wide"), help=argparse.SUPPRESS)
    comparison.add_argument("--worker-stage", choices=("pilot", "arm"), default="arm", help=argparse.SUPPRESS)
    comparison.add_argument("--until", type=int, help=argparse.SUPPRESS)
    comparison.add_argument("--lease-fd", type=int, help=argparse.SUPPRESS)
    comparison.add_argument("--parent-fd", type=int, help=argparse.SUPPRESS)
    training = commands.add_parser("train")
    evaluation = commands.add_parser("test")
    for command in (training, evaluation):
        command.add_argument("experiment")
        command.add_argument("--run-id")
    training.add_argument("--resume", action="store_true")
    training.add_argument("--pilot-run-id", default=PILOT_EXPERIMENT)
    args = parser.parse_args()
    args.runs = args.runs.resolve()
    if args.command == "report":
        print(json.dumps(report(args.runs), indent=2))
    elif args.command == "check":
        check(args.suite)
    elif args.command == "compare":
        args.run_id = args.run_id or ("train-volume-v1-4gpu" if args.gpus == 4 else "train-volume-v1")
        if args.worker:
            if args.gpus != 4 or args.lease_fd is None or args.parent_fd is None:
                parser.error("Workers must be launched by the four-GPU coordinator")
            parallel_worker(args)
        elif args.gpus == 4:
            volume_compare_parallel(args)
        else:
            volume_compare(args)
    else:
        if args.suite == "volume":
            parser.error("Use compare for the volume suite")
        args.run_id = args.run_id or args.experiment
        config, assets = load_config()
        experiment, schedule, train_record = select_experiment(config, assets, args.experiment)
        identity = run_identity(args.experiment, config, experiment, assets, train_record)
        if args.command == "train":
            train(args, identity, experiment, assets, schedule)
        else:
            test(args, identity, experiment, assets)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, FloatingPointError, ImportError, KeyError, subprocess.CalledProcessError) as error:
        sys.exit(f"Error: {error}")
