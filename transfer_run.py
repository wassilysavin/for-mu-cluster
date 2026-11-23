"""Compare two matched TRAIN schedules on fresh dialogue families and frozen regressions."""

from private_runtime import python_tool

from collections import Counter
from contextlib import ExitStack, contextmanager
from copy import copy
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import argparse
import json
import os
import shutil
import subprocess
import sys

import diagnose
import history
import ledger
import run
from diagnostics import student_evidence
from private_runtime import private_directory
from source_provenance import runtime_sources, verify_sources
from training.transfer_decision import BOOTSTRAP_REPEATS, MINIMUM_GAIN, SEEDS, compare_pair
from data_tools.transfer_split import check_separation


ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs"
NAME = "semantic-transfer-v1"
PARENT_STEP = 7991
ARMS = ("ordinary", "contrast")
DEV = ROOT / "assets/transfer-dev-v1.json"
MIB = 1024**2
POLICY = {"seeds": list(SEEDS), "minimum_gain": MINIMUM_GAIN, "bootstrap_replicates": BOOTSTRAP_REPEATS,
          "checkpoint": "last scheduled update for each arm; no DEV-based epoch selection",
          "optimizer": "inherit treatment-1729 step 7991 weights, FP32 Adafactor state and update counter",
          "fresh": "closed_loop primary; challenge separate; paired scenario-family bootstrap",
          "regressions": ["actual-history-v3-dev", "dev-volume-v1"], "release_ready": False}


def record(path):
    return ledger.file_record(path, relative_to=ROOT)


def command_json(args):
    return json.loads(subprocess.check_output(list(map(str, args)), text=True))


def freeze(path, value):
    return history.immutable(path, value)


class Space:
    """Count the existing reference budget plus transfer artifacts; keep the physical reserve."""

    def __init__(self, lease):
        self.lease = lease
        prior = RUNS / "train-fit-reference-v1"
        if prior.is_dir():
            self.previous = diagnose.DiskBudget(prior)
        else:
            directory = private_directory(ROOT, Path("runs") / NAME)
            self.previous = diagnose.DiskBudget(directory, create=True)

    def available(self):
        existing = self.previous.usage()["charged_bytes"]
        distinct = {}
        patterns = ("transfer*", "tests/test_transfer*.py", "assets/transfer*", "data_tools/*.py", "eval_tools/*.py",
                    "evaluation/*.py", "training/*.py", "diagnostics/*.py")
        tracked = [p for pattern in patterns for p in ROOT.glob(pattern) if p.is_file()]
        # Allow a second full copy of the published inputs for Git/LFS object storage.
        git_allowance = sum(path.stat().st_size for path in tracked if path.is_file())
        paths = [p for p in RUNS.glob(f"{NAME}*") if p != self.previous.output]
        paths.extend(tracked)
        for path in paths:
            candidates = [path, *path.rglob("*")] if path.is_dir() else [path]
            for candidate in candidates:
                history.contained(candidate, ROOT)
                entry = diagnose.allocation(candidate)
                key = entry["device"], entry["inode"]
                distinct[key] = entry["bytes"]
        used = existing + sum(distinct.values()) + git_allowance
        physical = shutil.disk_usage(ROOT).free - diagnose.RESERVE
        return min(diagnose.DISK_LIMIT - used, physical)

    def require(self, size):
        available = self.available()
        if available < size:
            raise RuntimeError(f"Transfer needs {size:,} additional bytes; {available:,} are available "
                               "within the shared 3 GB accounting and free-space reserve. Progress is saved.")
        return available


def source(assets):
    registry, bundle, previous = student_evidence.load_saved()
    if registry != assets["registry"][0] or bundle != assets["history_dev"][0]:
        raise ValueError("Regression assets differ from the authenticated parent DEV")
    base = ROOT / student_evidence.SOURCE
    terminal = history.read(base / "completion-v1/result.json")
    parent = terminal["pairs"][0]["treatment"]["checkpoint"]
    checkpoint = student_evidence.verify(ROOT, ROOT / parent["path"], parent)
    identity = history.read(base / "identities.json")["treatment-1729"]
    core = student_evidence.inventory(identity["code"]["files"])
    verify_sources(ROOT, dict(identity["code"], files=[core[name] for name in ("model.py", "run.py")]))
    completion = history.read(base / "completion-v1/experiment.json")
    inherited = student_evidence.inventory(completion["source_snapshot"]["immutable"])
    parity_path = student_evidence.from_inventory(ROOT, base / "parity/parity.json", inherited)
    parity = history.read(parity_path)
    if parity.get("passed") is not True or parity["binding"]["runtime"] != ledger.file_record(ROOT / "model.py"):
        raise ValueError("The parent has no verified native-runtime parity receipt")
    for key in ("base", "tokenizer"):
        if identity[key] != assets[key][1]:
            raise ValueError(f"The parent {key} differs from the current recipe")
    lineage = {"identity_sha256": history.digest(identity),
               "initial_checkpoint_sha256": identity["initial_checkpoint"]["sha256"]}
    check_header(checkpoint, PARENT_STEP, lineage)
    if previous["treatment-1729"][0].get("infrastructure_failures", 0):
        raise ValueError("The source DEV contains infrastructure failures")
    origin = {"name": "treatment-1729", "step": PARENT_STEP, "checkpoint": parent,
              "lineage": lineage, "parity": record(parity_path), "source_result": record(base / "completion-v1/result.json")}
    return origin, previous


def check_header(checkpoint, expected_step, lineage):
    meta = history.checkpoint_header(checkpoint)["__metadata__"]
    if int(meta["step"]) != expected_step or json.loads(meta["lineage"]) != lineage:
        raise ValueError(f"Unexpected checkpoint step or lineage at {checkpoint}")


def bindings():
    files = {"run.py", "model.py", "history.py", "ledger.py", "diagnose.py", "finish_history.py",
             "diagnostics/student_evidence.py", "private_runtime.py", "cluster", "requirements.txt", "reference_model.py",
             "transfer_contract.py", "data_tools/transfer_contract.py", "data_tools/transfer_split.py", "training/transfer_decision.py",
             "eval_tools/diagnose_data.py", "eval_tools/diagnose_dev.py", "eval_tools/evaluate_known.py", "experiments.json",
             "experiments-history.json", "assets/history-known-review.json"}
    files.update(runtime_sources(ROOT))
    for pattern in ("transfer*.py", "assets/transfer*"):
        files.update(p.relative_to(ROOT).as_posix() for p in ROOT.glob(pattern) if p.is_file())
    return ledger.code_record(ROOT, sorted(files))


def schedules(prepared):
    path = history.contained(prepared["path"], RUNS / NAME)
    count, updates = prepared["count"], prepared["updates"]
    if type(count) is not int or count < 8 or 4 * updates != count:
        raise ValueError("TRAIN schedules must contain complete batches of four")
    result = {}
    for seed in SEEDS:
        for arm in ARMS:
            key = prepared["schedule_keys"][arm][str(seed)]
            result[f"{arm}-{seed}"] = run.Schedule(path, prepared["sha256"], key, count, batch_size=4)
        left, right = [result[f"{arm}-{seed}"].rows for arm in ARMS]
        if sum(x[0] == y[0] for x, y in zip(left, right, strict=True)) * 2 != count:
            raise ValueError("Exactly half of the paired TRAIN presentations must be identical replay")
        for offset in range(0, count, 4):
            replay = sum(x[0] == y[0] for x, y in zip(left[offset:offset+4], right[offset:offset+4], strict=True))
            if replay != 2:
                raise ValueError("Each matched update must contain two common and two variable cases")
        for index, (x, y) in enumerate(zip(left, right, strict=True)):
            if x[0] == y[0] and x != y:
                raise ValueError(f"Common replay bytes differ at draw {index}")
        for label, pos in (("target", 4), ("input", 2)):
            totals = [sum(row[pos] if label == "target" else row[2] - row[4] for row in arm)
                      for arm in (left, right)]
            if abs(totals[0] - totals[1]) > 0.05 * min(totals):
                raise ValueError(f"{label} token exposure differs by more than 5%")
    if any(result[f"{arm}-1729"].sha256 == result[f"{arm}-2718"].sha256 for arm in ARMS):
        raise ValueError("The confirmatory seed must change the order of TRAIN presentations")
    pilot = copy(result[f"ordinary-{SEEDS[0]}"])
    pilot.rows = pilot.rows[:8]
    pilot.sha256 = history.digest([row[0] for row in pilot.rows])
    result["pilot"] = pilot
    return result


def training_identity(name, plan, common, recipe):
    pilot = name == "pilot"
    updates = len(plan.rows) // 4
    experiment = {"kind": "pilot" if pilot else "comparison", "training": {"arm": name, "updates": updates},
                  "milestones": [] if pilot else [updates]}
    identity = {**common, "experiment_id": name,
                "train": dict(common["train"], schedule_sha256=plan.sha256, rows=len(plan.rows), arm=name),
                "recipe": dict(recipe, initial_step=PARENT_STEP, examples_per_update=4,
                               optimizer=POLICY["optimizer"], kind=experiment["kind"],
                               training=experiment["training"], milestones=experiment["milestones"],
                               order="paired ordinary/contrast policy; 50% position-identical common replay",
                               pilot={"id": "pilot", "updates": 2})}
    return identity, experiment


class Training(history.ClusterBackend):
    def __init__(self, assets, prepared, common, recipe, space):
        self.assets, self.space, self.runner = assets, space, run
        self.args = SimpleNamespace(runs=RUNS, run_id=NAME)
        self.receipt, self.hardware = common["origin"], common["runtime"]
        self.budget = SimpleNamespace(account=None)
        self.schedules = schedules(prepared)
        self.identities, self.experiments = {}, {}
        for name, plan in self.schedules.items():
            self.identities[name], self.experiments[name] = training_identity(name, plan, common, recipe)
        # The inherited pilot audits the first comparison identity against a four-replica reference.
        self.identities["control-1729"] = self.identities["ordinary-1729"]

    def train_arm(self, name):
        job = self.job(name)
        updates = self.experiments[name]["training"]["updates"]
        step = PARENT_STEP + updates
        checkpoint = RUNS / job.run_id / f"checkpoint-step-{step}.safetensors"
        if not checkpoint.exists():
            self.space.require(2 * run.checkpoint_budget(self.assets) + 4*MIB)
            self.runner.train(job, self.identities[name], self.experiments[name], self.assets,
                              self.schedules[name], until=updates, lease_held=True)
        self.validate_checkpoint(checkpoint, name, step)
        return checkpoint


@contextmanager
def provider(checkpoint, assets, expected_step):
    import torch
    runtime, generate = diagnose.student_provider(checkpoint, {k: v[1] for k, v in assets.items()}, 0)
    try:
        if runtime.step != expected_step or runtime.tokenizer_sha256 != assets["tokenizer"][1]["sha256"]:
            raise ValueError("Evaluation loaded the wrong step or tokenizer")
        yield generate
    finally:
        runtime.close()
        del generate, runtime
        torch.cuda.empty_cache()


def pump(command, path, generate, binding, space):
    available = space.require(2*MIB)
    limit = (path.stat().st_size if path.exists() else 0) + available - MIB
    # Set the file-size limit in a fresh process, outside the multithreaded CUDA parent.
    limiter = """import ctypes, os, resource, signal, sys
limit, parent = map(int, sys.argv[1:3])
if sys.platform == "linux":
    if ctypes.CDLL(None).prctl(1, signal.SIGKILL, 0, 0, 0) != 0 or os.getppid() != parent:
        sys.exit("Evaluator coordinator disappeared or parent-death registration failed")
_, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
resource.setrlimit(resource.RLIMIT_FSIZE, (limit if hard == resource.RLIM_INFINITY else min(limit, hard), hard))
os.execvp(sys.argv[3], sys.argv[3:])
"""
    command = [sys.executable, "-B", "-S", "-c", limiter, str(limit), str(os.getpid()), *map(str, command)]
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          text=True, encoding="utf-8", pass_fds=(() if space.lease is None else (space.lease,))) as process:
        score = None
        generated = 0
        try:
            for line in process.stdout:
                message = json.loads(line)
                if message.get("kind") == "result" and score is None:
                    score = message["score"]
                    continue
                if message.get("kind") != "generate" or score is not None:
                    raise ValueError("Unexpected DEV evaluator message")
                reply = generate(message["prompt"])
                reply["diagnostic_worker_sha256"] = binding
                process.stdin.write(history.canonical(reply) + "\n")
                process.stdin.flush()
                generated += 1
                if generated == 1 or generated % 25 == 0:
                    print(f"[{path.parent.name}/{path.stem}] generated {generated} new cases", flush=True)
            process.stdin.close()
            if process.wait() or score is None or score.get("aborted"):
                raise RuntimeError(f"Incomplete DEV evaluator at {path}")
            return score
        except BaseException:
            process.kill()
            process.wait()
            raise


def legacy_panel(directory, assets, generate, binding, space):
    prefix = [*python_tool("eval_tools/diagnose_dev.py"), assets["registry"][0], assets["history_dev"][0]]
    shards = []
    for shard in range(4):
        path = directory / f"history-{shard}.jsonl"
        summary = pump([*prefix, path, shard, 4, diagnose.RESERVE, binding], path, generate, binding, space)
        if summary["shard"] != shard or summary["shards"] != 4:
            raise ValueError("History regression shard identity differs")
        shards.append(path)
    bundle = history.read(assets["history_dev"][0])
    ids = [row["case_id"] for panel in ("fixed_prefix", "closed_loop", "challenge") for row in bundle["panels"][panel]]
    rows = diagnose.merged_rows(shards, ids)
    with TemporaryDirectory(dir=directory) as tmp:
        path = Path(tmp) / "merged.jsonl"
        data = "".join(history.canonical(row) + "\n" for row in rows).encode()
        space.require(len(data) + MIB)
        path.write_bytes(data)
        score = command_json([*prefix, "--summarize", path])["score"]
    failed = history.infrastructure_failures(rows)
    score.update(infrastructure_failures=len(failed), infrastructure_failure_case_ids=failed)
    return score, shards


def enrich_known(score, rows, data):
    plan = [json.loads(line) for line in data.read_text().splitlines()]
    if [row["case_id"] for row in plan] != [r["case_id"] for r in rows]:
        raise ValueError("Legacy DEV155 rows do not match the frozen order")
    standalone = [r["result"] for example, r in zip(plan, rows, strict=True)
                  if example["cohort"] == "supported" and not example.get("dialogue_id")]
    passed = sum(row["success"] is True and row["critical_status"] == "none_observed" for row in standalone)
    score["summary"]["standalone"] = {"passed": passed, "planned": len(standalone),
                                      "accuracy": passed / len(standalone) if standalone else None}
    detected = [r["result"] for r in rows if r["result"]["critical_status"] == "detected"]
    if score["summary"]["critical_detected"] != len(detected):
        raise ValueError("Recorded critical count differs from the legacy evaluator")
    score["summary"]["critical_types"] = dict(Counter(
        r.get("critical_category") or "+".join(sorted(r.get("reasons", []))) or "unspecified" for r in detected))


def evaluate(directory, name, checkpoint, step, assets, config, manifest, fresh_spec, space, inherited,
             *, deadline=None):
    output = private_directory(ROOT, (directory / name).relative_to(ROOT))
    model = record(checkpoint)
    binding = history.digest({"manifest": history.digest(manifest), "checkpoint": model, "model": name})
    space.require(MIB)
    freeze(output / "input.json", {"binding": binding, "checkpoint": model})
    result = {}
    generator = None
    with ExitStack() as stack:
        def generate(prompt):
            nonlocal generator
            run.check_deadline(deadline)
            if generator is None:
                generator = stack.enter_context(provider(checkpoint, assets, step))
            return generator(prompt)
        for key, count in (("fresh", fresh_spec["cases"]), ("legacy155", 155), ("legacy1250", 1250)):
            receipt = output / f"{key}.json"
            if receipt.exists():
                value = history.read(receipt)
                if (value["binding"] != binding or value["receipt_sha256"] != history.digest(
                        {k: v for k, v in value.items() if k != "receipt_sha256"})):
                    raise ValueError("Saved DEV stage has another checkpoint or configuration")
                for entry in value["observations"]:
                    student_evidence.verify(ROOT, ROOT / entry["path"], entry)
            else:
                if key == "legacy1250" and inherited is not None:
                    score, path = inherited
                    paths = [path]
                elif key == "legacy1250":
                    score, paths = legacy_panel(output, assets, generate, binding, space)
                else:
                    fresh = key == "fresh"
                    evaluator = "data_tools/transfer_dev.py" if fresh else "eval_tools/evaluate_known.py"
                    data = DEV if fresh else assets["regression_dev"][0]
                    spec = ({"bundle_sha256": fresh_spec["bundle_sha256"], "diagnostic_worker_sha256": binding}
                            if fresh else config["regression"])
                    path = output / f"{key}.jsonl"
                    score = pump([*python_tool(evaluator), assets["registry"][0], data, path,
                                  history.canonical(spec), diagnose.RESERVE], path, generate, binding, space)
                    rows = history.read_observations(path)
                    paths = [path]
                    if len(rows) != count:
                        raise ValueError(f"Incomplete {key} evaluation")
                    if fresh and score.get("bundle_sha256") != fresh_spec["bundle_sha256"]:
                        raise ValueError("Fresh DEV scored a different bundle")
                    if not fresh:
                        enrich_known(score, rows, data)
                    failures = history.infrastructure_failures(rows)
                    score["infrastructure_failures"] = max(score.get("infrastructure_failures", 0), len(failures))
                    score.pop("results", None)
                run.check_deadline(deadline)
                value = {"binding": binding, "observations": [record(path) for path in paths], "score": score}
                value["receipt_sha256"] = history.digest(value)
                space.require(len(history.canonical(value).encode()) + 65536)
                freeze(receipt, value)
            score = value["score"]
            if score.get("observed_records") != count or score["summary"]["missing"]:
                raise ValueError(f"Invalid {key} denominator or missing observations")
            result[key] = score
    summary = result["fresh"]["summary"]
    metrics = []
    for key in ("supported", "whole_dialogues", "outside_scope"):
        metric = summary.get(key)
        if metric:
            metrics.append(f"{key} {metric['passed']}/{metric['planned']}")
    print(f"{name}: fresh closed-loop " + ", ".join(metrics) +
          f", critical {summary['critical_detected']}, unresolved {summary['critical_review_required']}", flush=True)
    return {"checkpoint": model, "scores": result}


def check():
    import transfer_train
    training = transfer_train.check()
    spec = command_json([*python_tool("data_tools/transfer_dev.py"), ROOT / "assets/registry.json", DEV, "--check"])
    if spec.get("kind") != "check" or not spec.get("cases"):
        raise ValueError("Invalid fresh DEV preflight")
    return {"train": training, "dev": spec, "separation": check_separation()}


def execute():
    import transfer_train
    with ledger.execution_lease(RUNS) as lease:
        if (RUNS / NAME / "result.json").is_file():
            from diagnostics.saved_transfer import load_saved_transfer
            _, result, _, _ = load_saved_transfer(ROOT)
            print(f"{NAME}: {result['status']}; completed report {RUNS / NAME / 'result.json'}", flush=True)
            return result
        preflight = check()
        description, fresh_spec = preflight["train"], preflight["dev"]
        config, original, assets = history.load(run)
        origin, previous = source(assets)
        checkpoint = ROOT / origin["checkpoint"]["path"]
        assets["initial_checkpoint"] = checkpoint, origin["checkpoint"]
        space = Space(lease)
        directory = private_directory(ROOT, Path("runs") / NAME)
        prepared_dir = private_directory(ROOT, Path("runs") / NAME / "train")
        prepared = transfer_train.prepare(prepared_dir, available_bytes=space.require(MIB) - 65536)
        space.require(MIB)
        assets["train"] = Path(prepared["path"]), record(prepared["path"])
        hardware = run.four_gpu_identity()
        code = bindings()
        manifest_path = directory / "experiment.json"
        if manifest_path.exists():
            previous_code = history.read(manifest_path)["common"]["code"]
            if previous_code["files"] != code["files"]:
                raise ValueError("Transfer code or frozen assets changed since the run began")
            code = previous_code
        common = {"architecture_id": original["architecture"]["id"], "architecture": original["architecture"],
                  "initial_checkpoint": origin["checkpoint"], "origin": origin, "runtime": hardware, "code": code,
                  "base": assets["base"][1], "tokenizer": assets["tokenizer"][1], "train": assets["train"][1],
                  "evaluation": {**record(DEV), "split": "DEV-HISTORY", "dataset_id": NAME},
                  "execution": {"mode": "four-replica-batch4-v1", "devices": [0, 1, 2, 3],
                                "gradient_reduction": "sum", "examples_per_update": 4}}
        backend = Training(assets, prepared, common, original["recipe"], space)
        manifest = {"protocol": NAME, "policy": POLICY, "common": common, "training": description,
                    "token_balance": prepared["token_balance"], "train_receipt": record(prepared_dir / "train-receipt.json"),
                    "fresh": fresh_spec, "separation": preflight["separation"],
                    "regressions": [assets[k][1] for k in ("history_dev", "regression_dev")],
                    "identities": {k: history.digest(v) for k, v in backend.identities.items() if k != "control-1729"}}
        freeze(directory / "experiment.json", manifest)
        pilot_path = RUNS / f"{NAME}-pilot/checkpoint.safetensors"
        if not pilot_path.exists():
            space.require(2 * run.checkpoint_budget(assets) + 8*MIB)
        backend.pilot(None)
        baseline = evaluate(directory, "baseline", checkpoint, PARENT_STEP, assets, config, manifest,
                            fresh_spec, space, previous["treatment-1729"])
        pairs = []
        for seed in SEEDS:
            arms, observations = {}, {}
            for arm in ARMS:
                name = f"{arm}-{seed}"
                trained = backend.train_arm(name)
                arms[arm] = evaluate(directory, name, trained, PARENT_STEP + prepared["updates"],
                                     assets, config, manifest, fresh_spec, space, None)
                observations[arm] = {r["case_id"]: r for r in history.read_observations(directory / name / "fresh.jsonl")}
            decision = compare_pair(baseline, arms["ordinary"], arms["contrast"],
                                    observations["ordinary"], observations["contrast"], seed)
            pair = {"seed": seed, "arms": arms, "decision": decision}
            freeze(directory / f"pair-{seed}.json", pair)
            pairs.append(pair)
            interval = decision["paired_95pct"]
            print(f"Seed {seed}: contrast minus ordinary {100*decision['supported_gain']:+.1f} pp, "
                  f"family-bootstrap 95% CI [{100*interval['lower']:+.1f}, {100*interval['upper']:+.1f}] pp; "
                  f"positive signal {decision['positive_signal']}", flush=True)
            if not decision["positive_signal"]:
                break
        confirmed = len(pairs) == 2 and all(p["decision"]["positive_signal"] for p in pairs)
        reviews = any(any(n for n in panel.values()) for pair in pairs for panel in pair["decision"]["unresolved"].values())
        status = "confirmed_diagnostic_gain" if confirmed else "not_promoted"
        if confirmed and reviews:
            status = "inconclusive_pending_review"
        result = {"status": status, "baseline": baseline, "historical_7205_legacy1250": previous["baseline-7205"][0],
                  "pairs": pairs, "identity_sha256": history.digest(manifest), "release_ready": False}
        path = directory / "result.json"
        freeze(path, result)
        print(f"{status}; {len(pairs)} complete pair(s). Report: {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", choices=[4], type=int, default=4)
    parser.add_argument("--check", action="store_true", help="validate frozen assets without loading a model")
    args = parser.parse_args()
    if args.check:
        print(history.canonical(check()))
    else:
        execute()


if __name__ == "__main__":
    main()
