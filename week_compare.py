"""Compare the two weekly TRAIN sizes on four V100s from the same checkpoint."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from math import isfinite
from time import time
from types import SimpleNamespace

import diagnose
from diagnostics import new_fit_workers, saved_transfer, student_evidence
import extended_train
import history
import ledger
from private_runtime import private_directory
import run
from source_provenance import runtime_sources
from training.transfer_decision import panel_summaries, verify_primary
import transfer_run as transfer
import week_train


ROOT, RUNS = transfer.ROOT, transfer.RUNS
NAME = "week-train-volume-v2"
ARMS = ("matched", "full")
SEED = 1729
PARENT_STEP = 7991
UPDATES = 4044
BATCH_SIZE = 4
CHECKPOINT_INTERVAL = 10
PILOT_UPDATES = 2
HOURS = 168
MIB = 1024**2
TRAIN = {
    "path": "assets/week-train-v2.jsonl",
    "sha256": "0d0c5e28ef54f598b3f72ebd09de6dd79a0f299d46b09a5d726536fa582cd198",
}
SIZES = {"matched": 5169, "full": 16174}
# Each audited pilot update charges the parallel step and two sequential checks.
# Leave room for two interruptions between periodic checkpoints.
MAX_CHARGED_UPDATES = UPDATES * len(ARMS) + 3 * PILOT_UPDATES + 2 * CHECKPOINT_INTERVAL


class WeeklySchedule:
    """Read the packed dataset directly through the training runner's batch interface."""

    def __init__(self, path, rows, prefixes):
        self.path = path
        self.rows = rows
        self.prefixes = prefixes
        self.sha256 = history.digest([row[0] for row in rows])
        self.tokenizer = None
        self.space_check = None

    def batch(self, stream, offset, size):
        selected = self.rows[offset : offset + size]
        if size != BATCH_SIZE or len(selected) != size:
            raise ValueError("Incomplete weekly TRAIN batch")
        if self.space_check and (
            offset % (BATCH_SIZE * CHECKPOINT_INTERVAL) == 0 or offset + size == len(self.rows)
        ):
            self.space_check()
        if self.tokenizer is None:
            self.tokenizer = week_train.load_tokenizer()
        batch, cases = [], []
        for case_id, position, _, line_hash, _ in selected:
            stream.seek(position)
            line = stream.readline()
            if sha256(line).digest() != line_hash:
                raise ValueError(f"Frozen weekly TRAIN changed at {case_id}")
            tokens = week_train.unpack(json.loads(line), self.prefixes, self.tokenizer)
            ids, pivot = tokens["input_ids"], tokens["prompt_tokens"]
            batch.append((ids[:pivot], ids[pivot:]))
            cases.append(case_id)
        return batch, cases


def prepare_schedules(path):
    prefixes, offsets, strata, _ = extended_train.inspect(path)
    by_id = {cid: key for key, cases in strata.items() for cid in cases}
    index = {}
    with path.open("rb") as stream:
        for cid, offset in offsets.items():
            stream.seek(offset)
            line = stream.readline()
            row = json.loads(line)
            length = row["prompt_tokens"] + row["target_tokens"]
            if row["case_id"] != cid or not 0 < row["prompt_tokens"] < length <= run.CONTEXT:
                raise ValueError(f"Unexpected weekly TRAIN row: {cid}")
            index[cid] = (cid, offset, length, sha256(line).digest(), row["target_tokens"])

    draws = {arm: list(extended_train.draws(strata, arm=arm, updates=UPDATES, seed=SEED)) for arm in ARMS}
    if any(len(cases) != BATCH_SIZE * UPDATES or len(set(cases)) != SIZES[arm] for arm, cases in draws.items()):
        raise ValueError("Unexpected weekly TRAIN exposure")
    if any(by_id[left] != by_id[right] for left, right in zip(*(draws[arm] for arm in ARMS), strict=True)):
        raise ValueError("The weekly arms must match strata at every draw")
    plans = {arm: WeeklySchedule(path, [index[cid] for cid in cases], prefixes) for arm, cases in draws.items()}
    exposure = {}
    for arm, plan in plans.items():
        exposure[arm] = {
            "distinct_cases": SIZES[arm],
            "presentations": len(plan.rows),
            "schedule_sha256": plan.sha256,
            "prompt_tokens": sum(row[2] - row[4] for row in plan.rows),
            "target_tokens": sum(row[4] for row in plan.rows),
            "batches_with_repeats": sum(
                len(set(batch)) != BATCH_SIZE
                for batch in (draws[arm][i : i + BATCH_SIZE] for i in range(0, len(plan.rows), BATCH_SIZE))
            ),
        }
    for key in ("prompt_tokens", "target_tokens"):
        a, b = (exposure[arm][key] for arm in ARMS)
        if abs(a - b) > 0.05 * min(a, b):
            raise ValueError(f"Weekly arms exceed the 5% {key} exposure difference")
    # Exercise the longest inputs and the longest supervised target.
    hardest = sorted(index.values(), key=lambda row: (-row[2], row[0]))[:BATCH_SIZE]
    target = min(index.values(), key=lambda row: (-row[4], row[0]))
    companions = [row for row in plans["full"].rows if row != target and row not in hardest][: BATCH_SIZE - 1]
    plans["pilot"] = WeeklySchedule(path, [*hardest, target, *companions], prefixes)
    return plans, exposure


def completed_fit(transfer_result):
    """Require the completed measurement the user asked to finish first."""
    root = new_fit_workers.OUTPUT
    source = saved_transfer.RUN / "result.json"
    expected = {
        f"{arm}-{pair['seed']}": model["checkpoint"]
        for pair in transfer_result["pairs"]
        for arm, model in pair["arms"].items()
    }
    for directory in (root / new_fit_workers.PYTHON_REVISION, root):
        path = directory / "result.json"
        if not path.is_file():
            continue
        manifest = student_evidence.read(ROOT, directory / "experiment.json")
        result = student_evidence.read(ROOT, path)
        if (
            result["status"] != "complete"
            or result["step"] != 8151
            or result["updates"] != 160
            or manifest["protocol"] != "new-train-fit-v1"
            or manifest["source_result_sha256"] != transfer.record(ROOT / source)["sha256"]
            or result["experiment_sha256"] != history.digest(manifest)
            or {arm: value["checkpoint"] for arm, value in manifest["arms"].items()} != expected
            or {arm: value["checkpoint"] for arm, value in result["arms"].items()} != expected
            or any(value["subsets"]["all"]["total"] != 640 for value in result["arms"].values())
        ):
            raise ValueError("TRAIN-fit completion differs from the saved step-8151 transfer pair")
        return transfer.record(path)
    raise RuntimeError("Finish check_new_train_fit.py before starting the weekly comparison")


class Space:
    def __init__(self, lease):
        self.lease = lease
        self.previous = new_fit_workers.FitBudget(new_fit_workers.OUTPUT)

    def require(self, size):
        files = [ROOT / TRAIN["path"], week_train.DATA, Path(__file__)]
        entries = {path: diagnose.allocation(path) for path in files}
        for directory in RUNS.glob(f"{NAME}*"):
            student_evidence.checked_path(ROOT, directory)
            entries.update(diagnose.inventory(directory))
        distinct = {(entry["device"], entry["inode"]): entry["bytes"] for entry in entries.values()}
        git_copy = sum(path.stat().st_size for path in files)
        available = self.previous.require(0) - sum(distinct.values()) - git_copy
        if available < size:
            raise RuntimeError(
                f"Weekly comparison needs {size:,} additional bytes; {available:,} available "
                "under the shared 3 GB budget and physical reserve. Saved progress is retained."
            )
        return available


def code_identity(directory):
    names = {
        *runtime_sources(ROOT),
        "week_compare.py",
        "extended_train.py",
        "week_train.py",
        "transfer_run.py",
        "transfer_contract.py",
        "transfer_train.py",
        "run.py",
        "model.py",
        "ledger.py",
        "history.py",
        "cluster",
        "diagnose.py",
        "reference_model.py",
        "diagnostics/new_fit_workers.py",
        "diagnostics/student_evidence.py",
        "diagnostics/saved_transfer.py",
        "training/transfer_decision.py",
        "requirements.txt",
        "experiments.json",
        "experiments-history.json",
    }
    current = ledger.code_record(ROOT, sorted(names))
    receipt = directory / "experiment.json"
    if not receipt.exists():
        return current
    saved = student_evidence.read(ROOT, receipt)["code"]
    if current["files"] != saved["files"]:
        raise ValueError("Weekly experiment implementation changed since the launch")
    return saved


class Training(history.ClusterBackend):
    """Use the existing four-replica numerical pilot and checkpoint validation."""

    def __init__(self, plans, common, recipe, assets, space):
        self.args = SimpleNamespace(runs=RUNS, run_id=NAME)
        self.runner, self.assets = run, assets
        self.receipt, self.hardware = common["origin"], common["runtime"]
        self.schedules, self.space = plans, space
        self.identities, self.experiments = {}, {}
        for arm, plan in plans.items():
            pilot = arm == "pilot"
            kind = "pilot" if pilot else "comparison"
            updates = PILOT_UPDATES if pilot else UPDATES
            experiment = {
                "kind": kind,
                "training": {"arm": arm, "updates": updates},
                "milestones": [] if pilot else [UPDATES],
            }
            self.experiments[arm] = experiment
            self.identities[arm] = {
                **common,
                "experiment_id": arm,
                "train": dict(common["train"], arm=arm, rows=len(plan.rows), schedule_sha256=plan.sha256),
                "recipe": dict(
                    recipe,
                    initial_step=PARENT_STEP,
                    examples_per_update=BATCH_SIZE,
                    kind=kind,
                    training=experiment["training"],
                    milestones=experiment["milestones"],
                    checkpoint_interval=CHECKPOINT_INTERVAL,
                    order=f"weekly strata, seed {SEED}",
                    optimizer="inherit FP32 Adafactor state and update counter at step 7991",
                    pilot={"id": "pilot", "updates": PILOT_UPDATES},
                ),
            }
            plan.space_check = self.require_checkpoint_space
        # The common backend uses this entry to verify the pilot against an experimental arm.
        self.identities["control-1729"] = self.identities["matched"]

    def require_checkpoint_space(self):
        self.space.require(2 * run.checkpoint_budget(self.assets) + 4 * MIB)

    def train_arm(self, arm, deadline):
        job = self.job(arm)
        step = PARENT_STEP + UPDATES
        checkpoint = RUNS / job.run_id / f"checkpoint-step-{step}.safetensors"
        if not checkpoint.is_file():
            self.require_checkpoint_space()
            experiment = self.experiments[arm]
            run.train(
                job,
                self.identities[arm],
                experiment,
                self.assets,
                self.schedules[arm],
                until=UPDATES,
                lease_held=True,
                deadline=deadline,
            )
        student_evidence.checked_path(ROOT, checkpoint)
        self.validate_checkpoint(checkpoint, arm, step)
        return checkpoint


def open_budget(directory, binding, started):
    path = directory / "budget.json"
    maximum = MAX_CHARGED_UPDATES
    if path.exists():
        value = student_evidence.read(ROOT, path)
        if (
            value["identity_sha256"] != binding
            or value["maximum_updates"] != maximum
            or value["hours"] != HOURS
            or type(value["charged_updates"]) is not int
            or not 0 <= value["charged_updates"] <= maximum
            or not isfinite(value["started_at_unix"])
            or not 0 < value["started_at_unix"] <= started
        ):
            raise ValueError("Weekly comparison budget changed")
    else:
        if any((RUNS / f"{NAME}-{arm}").exists() for arm in ("pilot", *ARMS)):
            raise ValueError("Existing weekly jobs are missing their optimizer budget")
        value = {
            "identity_sha256": binding,
            "maximum_updates": maximum,
            "charged_updates": 0,
            "hours": HOURS,
            "started_at_unix": started,
        }
        ledger.publish_json(path, value)
    return SimpleNamespace(
        account={"path": path, "record": value}, deadline=value["started_at_unix"] + HOURS * 3600
    )


def comparisons(models, observations):
    panels = {name: panel_summaries(model) for name, model in models.items()}
    key = ("fresh", "closed_loop")
    for name, rows in observations.items():
        verify_primary(rows, panels[name][key])
    primary = {name: history.metric(values[key], "supported") for name, values in panels.items()}
    return {
        "full_minus_matched": primary["full"] - primary["matched"],
        "paired_95pct": history.paired_bootstrap(observations["matched"], observations["full"], seed=SEED),
        "change_from_baseline": {arm: primary[arm] - primary["baseline"] for arm in ARMS},
        "infrastructure_clean": not any(
            score["infrastructure_failures"] for model in models.values() for score in model["scores"].values()
        ),
    }


def print_report(result, path):
    for name, model in result["models"].items():
        summary = model["scores"]["fresh"]["summary"]
        counts = ", ".join(
            f"{key} {summary[key]['passed']}/{summary[key]['planned']}"
            for key in ("supported", "whole_dialogues", "outside_scope")
        )
        print(
            f"{name}: DEV233 closed loop, {counts}, critical {summary['critical_detected']}, "
            f"unresolved {summary['critical_review_required']}",
            flush=True,
        )
    comparison = result["comparison"]
    interval = comparison["paired_95pct"]
    print(
        f"Full minus matched: {100 * comparison['full_minus_matched']:+.1f} pp "
        f"(paired 95% CI {100 * interval['lower']:+.1f} to {100 * interval['upper']:+.1f} pp)",
        flush=True,
    )
    print(f"Report: {path}", flush=True)


def execute(lease):
    started = time()
    print("Verifying the completed TRAIN-fit and preparing the paired weekly schedules", flush=True)
    source_manifest, source_result, plan, baseline_rows = saved_transfer.load_saved_transfer(ROOT)
    fit = completed_fit(source_result)
    config, original, assets = history.load(run)
    origin, _ = transfer.source(assets)
    if origin != source_manifest["common"]["origin"]:
        raise ValueError("Weekly comparison must use the authenticated step-7991 transfer parent")
    assets["initial_checkpoint"] = ROOT / origin["checkpoint"]["path"], origin["checkpoint"]
    assets["train"] = run.checked_asset(TRAIN)
    schedules, exposure = prepare_schedules(assets["train"][0])
    dev = run.checked_asset({"path": str(saved_transfer.DEV), "sha256": saved_transfer.DEV_SHA256})
    space = Space(lease)
    directory = RUNS / NAME
    common = {
        "code": code_identity(directory),
        "architecture_id": original["architecture"]["id"],
        "architecture": original["architecture"],
        "runtime": run.four_gpu_identity(),
        "origin": origin,
        "initial_checkpoint": origin["checkpoint"],
        "base": assets["base"][1],
        "tokenizer": assets["tokenizer"][1],
        "train": assets["train"][1],
        "evaluation": {**dev[1], "split": "DEV-HISTORY", "dataset_id": "transfer-dev-v1"},
        "execution": {
            "mode": "four-replica-batch4-v1",
            "devices": [0, 1, 2, 3],
            "gradient_reduction": "sum",
            "examples_per_update": BATCH_SIZE,
        },
    }
    backend = Training(schedules, common, original["recipe"], assets, space)
    manifest = {
        **common,
        "protocol": NAME,
        "training_fit": fit,
        "seed": SEED,
        "updates_per_arm": UPDATES,
        "exposure": exposure,
        "regressions": [assets[key][1] for key in ("regression_dev", "history_dev")],
        "baseline_source": transfer.record(ROOT / saved_transfer.RUN / "result.json"),
        "identities": {arm: history.digest(backend.identities[arm]) for arm in ("pilot", *ARMS)},
        "release_ready": False,
        "TEST_opened": False,
    }
    space.require(MIB)
    private_directory(ROOT, directory.relative_to(ROOT))
    transfer.freeze(directory / "experiment.json", manifest)
    binding = history.digest(manifest)
    backend.budget = open_budget(directory, binding, started)
    path = directory / "result.json"
    print("Numerical pilot on four V100s", flush=True)
    pilot = backend.pilot(backend.budget.deadline)
    models, observations = {"baseline": source_result["baseline"]}, {"baseline": baseline_rows["baseline"]}
    for arm in ARMS:
        checkpoint = backend.train_arm(arm, backend.budget.deadline)
        models[arm] = transfer.evaluate(
            directory,
            arm,
            checkpoint,
            PARENT_STEP + UPDATES,
            assets,
            config,
            manifest,
            source_manifest["fresh"],
            space,
            None,
            deadline=backend.budget.deadline,
        )
        score = models[arm]["scores"]["fresh"]
        arm_binding = history.digest({"manifest": binding, "checkpoint": models[arm]["checkpoint"], "model": arm})
        observations[arm] = saved_transfer.load_fresh(
            directory / arm / "fresh.jsonl", plan, source_manifest["fresh"]["bundle_sha256"], arm_binding, score
        )
    result = {
        "status": "complete",
        "identity_sha256": binding,
        "models": models,
        "pilot": pilot,
        "comparison": comparisons(models, observations),
        "budget": backend.budget.account["record"],
        "release_ready": False,
        "TEST_opened": False,
    }
    space.require(MIB)
    transfer.freeze(path, result)
    print_report(result, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", choices=(4,), type=int, default=4)
    parser.add_argument("--check", action="store_true", help="Verify the published TRAIN schedules on CPU")
    args = parser.parse_args()
    if args.check:
        corpus, _ = run.checked_asset(TRAIN)
        schedules, exposure = prepare_schedules(corpus)
        for schedule in schedules.values():
            with schedule.path.open("rb") as stream:
                for offset in sorted({0, len(schedule.rows) - BATCH_SIZE}):
                    schedule.batch(stream, offset, BATCH_SIZE)
        print(history.canonical(exposure))
        return
    private_directory(ROOT, RUNS.relative_to(ROOT))
    with ledger.execution_lease(RUNS) as lease:
        execute(lease)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Weekly comparison: {error}")
