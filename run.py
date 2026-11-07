"""Train, evaluate, and inspect frozen GEM-pRF cluster experiments."""

import argparse
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import monotonic, time_ns

from ledger import code_record, file_record, open_run, report


ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "experiments.json"
ARCHITECTURE = "functiongemma270m-lora16-volta-fp32-v2"
PILOT_EXPERIMENT = "pilot-4847-fp32"
VOCABULARY = 262144
CONTEXT = 32768
TARGET_LIMIT = 512
STOP_TOKEN = 50
EXAMPLES_PER_UPDATE = 2


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
    def __init__(self, path, expected_sha, arm, count, *, longest=False):
        self.path = path
        slots = {}
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
                index = row["positions"].get(arm)
                if index is None:
                    continue
                ids = row["input_ids"]
                prompt = row["prompt_tokens"]
                if (type(index) is not int or index in slots or type(prompt) is not int
                        or not 0 < prompt < len(ids) <= CONTEXT or len(ids) - prompt > TARGET_LIMIT
                        or any(type(token) is not int or not 0 <= token < VOCABULARY for token in ids)):
                    raise ValueError(f"Invalid TRAIN record {case_id}")
                slots[index] = (case_id, offset, len(ids), sha256(line).digest())
        if digest.hexdigest() != expected_sha:
            raise ValueError("TRAIN changed during schedule verification")
        if sorted(slots) != list(range(count)):
            raise ValueError(f"Incomplete {arm} schedule, expected {count} positions")
        self.rows = [slots[index] for index in range(count)]
        if longest:
            self.rows = sorted(self.rows, key=lambda row: (-row[2], row[0]))[:EXAMPLES_PER_UPDATE]
        self.sha256 = sha256(compact([row[0] for row in self.rows]).encode()).hexdigest()

    def batch(self, stream, offset, size):
        selected = self.rows[offset:offset + size]
        if len(selected) != size:
            raise ValueError("Incomplete scheduled update")
        batch = []
        for case_id, position, _, expected_line in selected:
            stream.seek(position)
            line = stream.readline()
            if sha256(line).digest() != expected_line:
                raise ValueError(f"TRAIN changed during the run at {case_id}")
            row = json.loads(line)
            ids = row["input_ids"]
            pivot = row["prompt_tokens"]
            batch.append((ids[:pivot], ids[pivot:]))
        return batch, [row[0] for row in selected]


def load_config():
    config = read_json(CONFIG)
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
        schedule = Schedule(path, record["sha256"], training["arm"], config["train_rows_per_arm"],
                            longest=experiment["kind"] == "pilot")
        if len(schedule.rows) != EXAMPLES_PER_UPDATE * training["updates"]:
            raise ValueError("Training budget must match the frozen schedule")
        train_record = dict(record, dataset_id=training["dataset_id"], arm=training["arm"],
                            schedule_sha256=schedule.sha256, rows=len(schedule.rows))
    return experiment, schedule, train_record


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


def run_identity(experiment_id, config, experiment, assets, train_record):
    code = ["run.py", "model.py", "ledger.py", "evaluate.mjs", "requirements.txt", "experiments.json"]
    code.extend(path.relative_to(ROOT).as_posix() for path in (ROOT / "evaluation").glob("*.mjs"))
    return {"experiment_id": experiment_id, "architecture_id": ARCHITECTURE,
            "architecture": config["architecture"], "code": code_record(ROOT, code),
            "recipe": dict(config["recipe"], kind=experiment["kind"], training=experiment.get("training")),
            "runtime": runtime_identity(), "train": train_record,
            "evaluation": dict(assets["dev"][1], split="DEV700", dataset_id=config["evaluation_id"],
                               registry=assets["registry"][1]),
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


def require_pilot(runs, pilot_run_id, identity):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]{0,99}", pilot_run_id):
        raise ValueError("Invalid pilot run ID")
    path = runs / pilot_run_id
    manifest = read_json(path / "run.json")
    recorded = manifest["identity"]
    keys = ("architecture_id", "base", "initial_checkpoint", "tokenizer", "code", "runtime")
    if any(identity[key] != recorded[key] for key in keys) or recorded["experiment_id"] != PILOT_EXPERIMENT:
        raise ValueError("The pilot must use this code, checkpoint and runtime")
    if recorded["train"]["sha256"] != identity["train"]["sha256"]:
        raise ValueError("The pilot must use this frozen TRAIN source")
    checkpoint = file_record(path / "checkpoint.safetensors")
    verified = False
    with (path / "events.jsonl").open() as stream:
        for line in stream:
            if not line.endswith("\n"):
                break
            event = json.loads(line)
            if event["event"] == "pilot_reload_verified":
                verified = (event["details"]["checkpoint"] == checkpoint
                            and event["details"]["step"] == identity["recipe"]["initial_step"] + 1)
    if not verified:
        raise ValueError("Complete the pilot save and reload before training an experimental arm")


def train(args, identity, experiment, assets, schedule):
    import torch
    from model import LAYERS, LEARNING_RATE, LOSS_SCALE, Runtime

    recipe = identity["recipe"]
    if (schedule is None or recipe["learning_rate"] != LEARNING_RATE or recipe["loss_scale"] != LOSS_SCALE
            or identity["architecture"]["layers"] != LAYERS):
        raise ValueError("The experiment must declare the runtime's training recipe")
    if experiment["kind"] == "comparison":
        require_pilot(args.runs, args.pilot_run_id, identity)
    with open_run(args.runs, args.run_id, identity, resume=args.resume) as run:
        initial = assets["initial_checkpoint"][0]
        latest = run.path / "checkpoint.safetensors"
        checkpoint = latest if args.resume and latest.exists() else initial
        lineage = {"identity_sha256": run.identity_sha256,
                   "initial_checkpoint_sha256": assets["initial_checkpoint"][1]["sha256"]}
        runtime = Runtime(assets["base"][0], checkpoint)
        if runtime.tokenizer_sha256 != assets["tokenizer"][1]["sha256"]:
            raise ValueError("Checkpoint tokenizer differs from the experiment")
        if checkpoint == latest and runtime.lineage != lineage:
            raise ValueError("The saved checkpoint belongs to another run")
        initial_step = identity["recipe"]["initial_step"]
        done = runtime.step - initial_step
        updates = experiment["training"]["updates"]
        save_every = identity["recipe"]["checkpoint_interval"]
        if type(save_every) is not int or save_every < 1:
            raise ValueError("Checkpoint interval must be positive")
        if not 0 <= done <= updates or (checkpoint == initial and done != 0):
            raise ValueError("Checkpoint step is outside the scheduled interval")
        run.event("checkpoint_loaded", step=runtime.step, file=checkpoint.name)
        with schedule.path.open("rb") as stream:
            for update in range(done, updates):
                batch, case_ids = schedule.batch(stream, EXAMPLES_PER_UPDATE * update, EXAMPLES_PER_UPDATE)
                torch.cuda.reset_peak_memory_stats()
                began = monotonic()
                try:
                    loss = runtime.train_step(batch)
                except FloatingPointError as error:
                    raise FloatingPointError(f"{', '.join(case_ids)}: {error}") from error
                elapsed = monotonic() - began
                if runtime.step != initial_step + update + 1:
                    raise RuntimeError("Runtime cursor did not advance by one update")
                checkpoint_saved = (update + 1) % save_every == 0 or update + 1 == updates
                if checkpoint_saved:
                    save_latest(runtime, latest, lineage)
                run.event("update", step=runtime.step, loss=loss, seconds=elapsed, case_ids=case_ids,
                          peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                          checkpoint_saved=checkpoint_saved)
                print(f"{runtime.step}  loss={loss:.6f}  {elapsed:.2f}s", flush=True)
        if experiment["kind"] == "pilot":
            del runtime
            torch.cuda.empty_cache()
            restored = Runtime(assets["base"][0], latest)
            if restored.step != initial_step + updates or restored.lineage != lineage:
                raise ValueError("Pilot checkpoint reload differs from the saved run")
            run.event("pilot_reload_verified", step=restored.step, checkpoint=file_record(latest))


def evaluate_model(runtime, tokenizer_path, registry_path, dev_path, observations):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    if tokenizer.decode([STOP_TOKEN], skip_special_tokens=False) != "<start_function_response>":
        raise ValueError("Tokenizer terminal token differs")
    command = ["node", str(ROOT / "evaluate.mjs"), str(registry_path), str(dev_path), str(observations)]
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
                tokens = tokenizer.encode(message["prompt"], add_special_tokens=False).ids
                if not 0 < len(tokens) <= CONTEXT - TARGET_LIMIT:
                    raise ValueError(f"Complete prompt does not fit for {message['case_id']}")
                raw_ids = runtime.generate(tokens, max_new_tokens=TARGET_LIMIT, stop_ids={STOP_TOKEN})
                raw_text = tokenizer.decode(raw_ids, skip_special_tokens=False)
                process.stdin.write(compact({"raw_ids": raw_ids, "raw_text": raw_text}) + "\n")
                process.stdin.flush()
            if process.wait() != 0 or score is None or score["aborted"] or score["observed_records"] != 700:
                raise RuntimeError("DEV700 did not complete. Inspect the observations and run log.")
            return score
        except BaseException:
            process.kill()
            raise


def test(args, identity, experiment, assets):
    from model import Runtime

    baseline = experiment["kind"] == "baseline"
    if experiment["kind"] == "pilot":
        raise ValueError("Use the reference or an experimental arm for DEV700")
    resume = not baseline or (args.runs / args.run_id / "run.json").exists()
    with open_run(args.runs, args.run_id, identity, resume=resume, stage="evaluate") as run:
        checkpoint = assets["initial_checkpoint"][0] if baseline else run.path / "checkpoint.safetensors"
        checkpoint_record = file_record(checkpoint)
        runtime = Runtime(assets["base"][0], checkpoint)
        if runtime.tokenizer_sha256 != assets["tokenizer"][1]["sha256"]:
            raise ValueError("Checkpoint tokenizer differs from the experiment")
        expected_step = identity["recipe"]["initial_step"]
        if not baseline:
            expected_step += experiment["training"]["updates"]
            if runtime.lineage.get("identity_sha256") != run.identity_sha256:
                raise ValueError("The saved checkpoint belongs to another run")
        if runtime.step != expected_step:
            raise ValueError(f"Expected completed checkpoint at step {expected_step}, found {runtime.step}")
        observations = run.path / f"dev700-{time_ns()}.jsonl"
        run.event("evaluation_started", step=runtime.step, observations=observations.name)
        score = evaluate_model(runtime, assets["tokenizer"][0], assets["registry"][0], assets["dev"][0], observations)
        if file_record(checkpoint) != checkpoint_record:
            raise ValueError("Checkpoint changed during evaluation")
        run.write_result(score["summary"], checkpoint_record, step=runtime.step,
                         generated_cases=score["generated_cases"], observed_records=score["observed_records"],
                         observations=file_record(observations))
        print(json.dumps(score["summary"], indent=2))


def check():
    config, assets = load_config()
    for experiment_id in config["experiments"]:
        _, schedule, record = select_experiment(config, assets, experiment_id)
        if schedule:
            print(experiment_id, len(schedule.rows), record["schedule_sha256"])
    subprocess.run(["node", str(ROOT / "evaluate.mjs"), str(assets["registry"][0]),
                    str(assets["dev"][0]), "--check"], check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "runs", help="Shared run directory")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check")
    commands.add_parser("report")
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
        check()
    else:
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
