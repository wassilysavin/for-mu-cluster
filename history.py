"""Run the frozen actual-history experiment, preserving every completed stage."""

from contextlib import contextmanager
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import struct
import subprocess
import sys
from tempfile import TemporaryDirectory
from time import monotonic, time
from types import SimpleNamespace

from ledger import SAFE_ID, code_record, execution_lease, file_record, publish_json, read_result

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "experiments-history.json"
INITIAL_STEP = 7205
UPDATES = 786
MIDPOINT = 393
SEEDS = (1729, 2718)
MAX_UPDATES = 3246
MAX_SECONDS = 36 * 3600
RESERVE = 512 * 1024 ** 2
EVALUATION_BYTES = 192 * 1024 ** 2


class BudgetLimit(RuntimeError):
    pass


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def digest(value):
    return sha256(canonical(value).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def contained(path, parent=ROOT):
    result, parent = Path(path).resolve(), Path(parent).resolve()
    if result == parent or not result.is_relative_to(parent):
        raise ValueError(f"Path must stay inside {parent}: {result}")
    return result


def replace_json(path, value):
    path = Path(path)
    with TemporaryDirectory(dir=path.parent) as staging:
        temporary = Path(staging) / path.name
        publish_json(temporary, value)
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def immutable(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError(f"Frozen record changed: {path}")
    else:
        publish_json(path, value)
    return value


class Budget:
    """Charge work before updates; interrupted phases retain a conservative time charge."""

    def __init__(self, path, identity, hours=36, *, clock=time):
        if not 0 < hours <= 36:
            raise ValueError("History model-worker budget must be positive and at most 36 hours")
        self.clock, self.path = clock, Path(path)
        self.identity = identity
        record = {"identity_sha256": identity, "maximum_updates": MAX_UPDATES, "charged_updates": 0,
                  "maximum_seconds": hours * 3600, "elapsed_seconds": 0.0, "inflight": None, "stages": []}
        if self.path.exists():
            record = read(self.path)
            if (record.get("identity_sha256") != identity or record.get("maximum_updates") != MAX_UPDATES
                    or record.get("maximum_seconds") != hours * 3600):
                raise ValueError("Frozen history budget changed; do not reset it on resume")
            if record.get("inflight"):
                phase = record["inflight"]
                charged = max(0.0, self.clock() - phase["started_at"])
                record["elapsed_seconds"] += charged
                record["stages"].append({"name": phase["name"], "seconds": charged, "interrupted": True,
                                         "conservative_recovery_charge": True})
                record["inflight"] = None
        self.account = {"path": self.path, "record": record}
        replace_json(self.path, record)

    @property
    def remaining_seconds(self):
        value = self.account["record"]
        active = max(0.0, self.clock() - value["inflight"]["started_at"]) if value["inflight"] else 0.0
        return max(0.0, value["maximum_seconds"] - value["elapsed_seconds"] - active)

    @property
    def deadline(self):
        return self.clock() + self.remaining_seconds

    def check(self):
        if self.remaining_seconds <= 0:
            raise BudgetLimit("History model-worker time budget exhausted")
        value = self.account["record"]
        if value["charged_updates"] > value["maximum_updates"]:
            raise BudgetLimit("History optimizer budget exceeded")

    def claim_updates(self, updates):
        self.check()
        record = dict(self.account["record"])
        if type(updates) is not int or updates < 0 or record["charged_updates"] + updates > MAX_UPDATES:
            raise BudgetLimit("History optimizer budget cannot fit the requested work")
        record["charged_updates"] += updates
        replace_json(self.path, record)
        self.account["record"] = record

    @contextmanager
    def phase(self, name):
        self.check()
        record = self.account["record"]
        if record["inflight"]:
            raise RuntimeError("Model phases must execute sequentially under one lease")
        record["inflight"] = {"name": name, "started_at": self.clock()}
        replace_json(self.path, record)
        try:
            yield self.deadline
        finally:
            record = self.account["record"]
            phase = record["inflight"]
            elapsed = max(0.0, self.clock() - phase["started_at"])
            record["elapsed_seconds"] += elapsed
            record["stages"].append({"name": name, "seconds": elapsed, "interrupted": sys.exc_info()[0] is not None})
            record["inflight"] = None
            replace_json(self.path, record)


def checkpoint_header(path):
    with Path(path).open("rb") as source:
        raw = source.read(8)
        if len(raw) != 8:
            raise ValueError("Truncated checkpoint header")
        length = struct.unpack("<Q", raw)[0]
        if length > 8 * 1024 ** 2:
            raise ValueError("Checkpoint header exceeds the declared bound")
        return json.loads(source.read(length))


def source_checkpoint(runs, source_id, assets):
    if not SAFE_ID.fullmatch(source_id):
        raise ValueError("Invalid source course ID")
    outer = contained(runs / source_id, runs)
    course = contained(runs / f"{source_id}-full-course", runs)
    report = read(outer / "result.json")
    if report.get("status") != "complete" or report.get("completed_epochs") != 3 or len(report.get("epochs", [])) != 3:
        raise ValueError("The source full course must have all three completed evaluations")
    if [item.get("details", {}).get("step") for item in report["epochs"]] != [5633, 6419, 7205]:
        raise ValueError("The completed source course has unexpected epoch checkpoints")
    evaluated = read_result(course, name=f"result-step-{INITIAL_STEP}.json")
    if evaluated is None:
        raise FileNotFoundError("The actual checkpoint 7205 evaluation receipt is required; no fallback is allowed")
    path = contained(course / f"checkpoint-step-{INITIAL_STEP}.safetensors", runs)
    actual = file_record(path)
    if evaluated["checkpoint"] != actual or evaluated["details"].get("step") != INITIAL_STEP:
        raise ValueError("Checkpoint 7205 differs from its completed evaluation receipt")
    if evaluated["details"].get("observed_records") != 155 or evaluated["metrics"].get("missing") != 0:
        raise ValueError("Checkpoint 7205 needs its complete original DEV155 evaluation")
    observation_record = evaluated["details"]["observations"]
    observations = contained(course / observation_record["path"], course)
    if file_record(observations) != observation_record:
        raise ValueError("Original checkpoint 7205 observations differ from their receipt")
    if not any(item.get("checkpoint") == actual and item.get("details", {}).get("step") == INITIAL_STEP
               for item in report.get("epochs", [])):
        raise ValueError("The source course did not publish the actual final checkpoint 7205")
    manifest = read(course / "run.json")
    identity = manifest["identity"]
    if digest(identity) != manifest["identity_sha256"]:
        raise ValueError("Source run identity digest differs")
    for key in ("base", "tokenizer"):
        if identity[key]["sha256"] != assets[key][1]["sha256"]:
            raise ValueError(f"Source checkpoint uses a different {key}")
    if identity["recipe"]["initial_step"] != 4847 or identity["recipe"]["examples_per_update"] != 4:
        raise ValueError("Unexpected source full-course recipe")
    if (identity["architecture_id"] != "functiongemma270m-lora16-volta-fp32-v2"
            or identity["recipe"]["training"]["updates"] != 2358):
        raise ValueError("The source must be the original completed three-epoch full course")
    header = checkpoint_header(path)
    metadata = header.get("__metadata__", {})
    lineage = json.loads(metadata.get("lineage", "null"))
    expected = {"identity_sha256": manifest["identity_sha256"],
                "initial_checkpoint_sha256": identity["initial_checkpoint"]["sha256"]}
    if int(metadata.get("step", -1)) != INITIAL_STEP or lineage != expected:
        raise ValueError("Checkpoint 7205 cursor or original lineage differs")
    names = set(header) - {"__metadata__"}
    adapters = {name.removeprefix("lora.") for name in names if name.startswith("lora.")}
    if len(adapters) != 252 or names != {f"{prefix}.{name}" for name in adapters
                                      for prefix in ("lora", "adafactor.row", "adafactor.col")}:
        raise ValueError("Checkpoint 7205 must preserve all LoRA tensors and optimizer row/column state")
    if any(header[name]["dtype"] != "F32" for name in names):
        raise ValueError("Checkpoint 7205 state must remain FP32")
    receipt = {"step": INITIAL_STEP, "checkpoint": file_record(path, relative_to=ROOT), "lineage": lineage,
               "source_manifest": file_record(course / "run.json", relative_to=ROOT),
               "source_evaluation": file_record(course / f"result-step-{INITIAL_STEP}.json", relative_to=ROOT),
               "source_observations": file_record(observations, relative_to=ROOT),
               "source_course": file_record(outer / "result.json", relative_to=ROOT)}
    return path, receipt


def load(runner):
    config = read(CONFIG)
    if (config["id"] != "actual-history-v3" or config["initial_step"] != INITIAL_STEP
            or config["updates_per_arm"] != UPDATES or config["midpoint"] != MIDPOINT
            or config["seeds"] != list(SEEDS) or config["examples_per_update"] != 4
            or config["max_optimizer_updates"] != MAX_UPDATES or config["max_elapsed_hours"] != 36):
        raise ValueError("The frozen actual-history protocol differs")
    original = read(ROOT / "experiments.json")
    declarations = {name: original["assets"][name] for name in ("base", "tokenizer", "registry")}
    declarations.update(config["assets"])
    assets = {name: runner.checked_asset(reference) for name, reference in declarations.items()}
    return config, original, assets


def check(runner):
    import history_prepare
    config, _, assets = load(runner)
    history_prepare.check(config, {key: value[0] for key, value in assets.items()})
    subprocess.run(["node", str(ROOT / "evaluate-actual.mjs"), str(assets["registry"][0]),
                    str(assets["history_dev"][0]), "--check", canonical(config["evaluation"])], check=True)
    print("Inputs validated. The history command will verify checkpoint 7205 and runtime parity before training.", flush=True)


def validate_prepared(prepared, directory, runner):
    from history_prepare import verify_collection
    if prepared.get("complete") is not True or prepared.get("reviewed") is not True:
        raise ValueError("TRAIN collection and independent label review must be complete before freezing")
    declaration = prepared["train"]
    path = Path(declaration["path"])
    if not path.is_absolute():
        path = directory / path
    path = contained(path, directory)
    actual = file_record(path)
    if actual["sha256"] != declaration["sha256"] or actual["bytes"] != declaration["bytes"]:
        raise ValueError("Prepared TRAIN changed after collection")
    collection = prepared["collection_receipt"]
    if collection["fingerprint"].get("checkpoint_step") != INITIAL_STEP:
        raise ValueError("Prepared TRAIN must contain actual histories from checkpoint 7205")
    for key in ("control", "treatment", "prefix_receipts"):
        contained(path.parent / collection[key]["path"], path.parent)
    verify_collection(path.parent, collection)
    schedules = {}
    for arm in ["pilot"] + [f"{name}-{seed}" for seed in SEEDS for name in ("control", "treatment")]:
        count = 8 if arm == "pilot" else UPDATES * 4
        schedules[arm] = runner.Schedule(path, declaration["sha256"], arm, count, batch_size=4)
        if arm != "pilot" and prepared["schedules"][arm]["sha256"] != schedules[arm].sha256:
            raise ValueError(f"Prepared schedule changed: {arm}")
    return (path, file_record(path, relative_to=ROOT)), schedules


def make_identities(runner, original, config, assets, schedules, runtime, receipt, prepared):
    paths = ["run.py", "history.py", "history_prepare.py", "history_parity.py", "model.py", "ledger.py",
             "evaluate.mjs", "evaluate-actual.mjs", "evaluate-known.mjs", "requirements.txt", "experiments.json", "experiments-history.json"]
    paths.extend(str(path.relative_to(ROOT)) for path in sorted((ROOT / "evaluation").rglob("*.mjs")))
    paths.extend(config.get("code_files", []))
    code = code_record(ROOT, sorted(set(paths)))
    identities, experiments = {}, {}
    for name in ["pilot", "baseline"] + [f"{arm}-{seed}" for seed in SEEDS for arm in ("control", "treatment")]:
        kind = "baseline" if name == "baseline" else "pilot" if name == "pilot" else "comparison"
        updates = 2 if name == "pilot" else UPDATES
        experiment = {"kind": kind}
        record = None
        if kind != "baseline":
            experiment.update(training={"arm": name, "dataset_id": prepared["dataset_id"], "updates": updates},
                              milestones=[] if kind == "pilot" else [MIDPOINT, UPDATES])
            record = dict(assets["train"][1], dataset_id=prepared["dataset_id"], arm=name,
                          schedule_sha256=schedules[name].sha256, rows=len(schedules[name].rows))
        recipe = dict(original["recipe"], initial_step=INITIAL_STEP, examples_per_update=4,
                      kind=kind, training=experiment.get("training"), milestones=experiment.get("milestones", []),
                      pilot={"id": "pilot", "updates": 2},
                      order="50% identical common replay; history-training package versus first-biased reconstructed matched control; frozen seeds")
        identity = {"experiment_id": name, "architecture_id": original["architecture"]["id"],
                    "architecture": original["architecture"], "code": code, "recipe": recipe, "runtime": runtime,
                    "train": record, "evaluation": dict(assets["history_dev"][1], split="DEV-VOLUME",
                                                         dataset_id=config["evaluation"]["id"]),
                    "base": assets["base"][1], "tokenizer": assets["tokenizer"][1],
                    "initial_checkpoint": assets["initial_checkpoint"][1], "origin": receipt,
                    "prepared_sha256": digest(prepared), "execution": {"mode": "four-replica-batch4-v1",
                    "devices": [0, 1, 2, 3], "gradient_reduction": "sum", "examples_per_update": 4}}
        identities[name], experiments[name] = identity, experiment
    return identities, experiments


def read_observations(path):
    values = [json.loads(line) for line in Path(path).read_text().splitlines()]
    if len({row["case_id"] for row in values}) != len(values):
        raise ValueError("Evaluation contains duplicate case IDs")
    return values


def infrastructure_failures(rows):
    failed = []
    for row in rows:
        generation = row.get("generation") or {}
        exchanges = (row.get("after") or {}).get("exchanges", [])
        last = exchanges[-1] if exchanges else {}
        error = generation.get("error") or row.get("provider_failure")
        if (error and error != "context_overflow") or (
                last.get("outcome") == "provider_error" and last.get("error_code") != "context_overflow"):
            failed.append(row["case_id"])
    return failed


def metric(summary, name):
    value = summary[name]
    if (type(value.get("planned")) is not int or value["planned"] <= 0
            or type(value.get("passed")) is not int or not 0 <= value["passed"] <= value["planned"]):
        raise ValueError(f"Invalid complete metric: {name}")
    return value["passed"] / value["planned"]


def paired_bootstrap(control, treatment, *, repeats=10000, seed=1729):
    groups, original_families = {}, {}
    if set(control) != set(treatment):
        raise ValueError("Paired evaluation case IDs differ")
    for case_id, before in control.items():
        after = treatment[case_id]
        for key in ("family_id", "cohort", "panel"):
            if before[key] != after[key]:
                raise ValueError(f"Paired evaluation changed {key}: {case_id}")
        if before["panel"] != "closed_loop" or before["cohort"] != "supported":
            continue
        family = before.get("bootstrap_family_id", before["family_id"])
        if family != after.get("bootstrap_family_id", after["family_id"]):
            raise ValueError("Paired bootstrap blocks differ")
        previous = original_families.setdefault(before["family_id"], family)
        if previous != family:
            raise ValueError("A complete scenario family cannot be split across bootstrap blocks")
        difference = int(after["success"] is True and after["critical_status"] == "none_observed")
        difference -= int(before["success"] is True and before["critical_status"] == "none_observed")
        total = groups.setdefault(family, [0, 0])
        total[0] += difference
        total[1] += 1
    if not groups:
        raise ValueError("No complete supported families for bootstrap")
    families, rng, samples = list(groups.values()), random.Random(seed), []
    for _ in range(repeats):
        selected = rng.choices(families, k=len(families))
        samples.append(sum(item[0] for item in selected) / sum(item[1] for item in selected))
    samples.sort()
    return {"lower": samples[int(0.025 * (repeats - 1))], "upper": samples[int(0.975 * (repeats - 1))],
            "families": len(families), "replicates": repeats, "seed": seed,
            "unit": "whole family; turn-weighted complete supported accuracy"}


def decision(baseline, control, treatment, observations, *, confirmatory=False):
    summaries = []
    for item in (baseline, control, treatment):
        panels = item["score"]["panels"]
        summary = dict(item["score"]["summary"], exact_reference=panels["fixed_prefix"]["exact_reference"])
        summary["critical_by_kind"] = {}
        for panel in panels.values():
            for category, count in panel["critical_types"].items():
                kind = "history" if category in ("wrong_reference", "omitted_reference", "history_integrity") else category
                summary["critical_by_kind"][kind] = summary["critical_by_kind"].get(kind, 0) + count
        summary["critical_review_required"] = sum(panel["critical_review_required"] for panel in panels.values())
        summary["missing"] = sum(panel["missing"] for panel in panels.values())
        summaries.append(summary)
    b, a, t = summaries
    names = ("supported", "whole_dialogues", "outside_scope", "standalone", "exact_reference")
    for name in names:
        if len({value[name]["planned"] for value in (b, a, t)}) != 1:
            raise ValueError(f"Comparison denominators differ: {name}")
        for value in (b, a, t):
            metric(value, name)
    interval = paired_bootstrap(observations[0], observations[1])
    supported_delta = metric(t, "supported") - metric(a, "supported")
    reference_delta = metric(t, "exact_reference") - metric(a, "exact_reference")
    all_kinds = set().union(*(value["critical_by_kind"] for value in (b, a, t)))
    other_safe = all(t["critical_by_kind"].get(kind, 0) <= min(a["critical_by_kind"].get(kind, 0),
                         b["critical_by_kind"].get(kind, 0)) for kind in all_kinds - {"history"})
    gates = {
        "complete": all(value["missing"] == 0 for value in (b, a, t)),
        "infrastructure_clean": all(item["score"].get("infrastructure_failures", 0) == 0
                                    for item in (baseline, control, treatment)),
        "reviews_resolved": all(value["critical_review_required"] == 0 for value in (b, a, t)),
        "supported_gain": supported_delta >= (0 if confirmatory else min(.05, .5 * (1 - metric(a, "supported"))))
                          and supported_delta > 0,
        "reference_gain": reference_delta >= (0 if confirmatory else min(.15, .5 * (1 - metric(a, "exact_reference"))))
                          and reference_delta > 0,
        "whole_dialogues_gain": metric(t, "whole_dialogues") > metric(a, "whole_dialogues"),
        "paired_interval": confirmatory or interval["lower"] > 0,
        "history_critical": t["critical_by_kind"].get("history", 0) <= min(
            a["critical_by_kind"].get("history", 0) / 2, b["critical_by_kind"].get("history", 0)),
        "other_critical": other_safe,
        "baseline_supported": metric(t, "supported") >= metric(b, "supported"),
        "baseline_dialogues": metric(t, "whole_dialogues") >= metric(b, "whole_dialogues"),
        "standalone_guard": metric(t, "standalone") >= max(metric(a, "standalone"), metric(b, "standalone")),
        "outside_scope_guard": metric(t, "outside_scope") >= max(metric(a, "outside_scope"), metric(b, "outside_scope")),
    }
    return {"passed": all(gates.values()), "gates": gates, "supported_delta": supported_delta,
            "reference_delta": reference_delta, "bootstrap": interval,
            "status": "diagnostic_gain" if all(gates.values()) else "inconclusive" if (
                not gates["reviews_resolved"] or not gates["infrastructure_clean"]) else "not_promoted",
            "product_acceptance": False}


def timing_records(path, *, repair=False):
    if not Path(path).exists():
        return []
    records, offset = [], 0
    with Path(path).open("rb+" if repair else "rb") as stream:
        for line in stream:
            if not line.endswith(b"\n"):
                if repair:
                    stream.truncate(offset)
                    stream.flush()
                    os.fsync(stream.fileno())
                    break
                raise ValueError("Evaluation timing receipt has an incomplete tail")
            value = json.loads(line)
            if value.get("kind") not in ("load", "generation", "overhead") or not math.isfinite(value["seconds"]) or value["seconds"] < 0:
                raise ValueError("Invalid evaluation timing receipt")
            records.append(value)
            offset += len(line)
    return records


def append_timing(path, kind, seconds, *, case_id=None):
    if path is None:
        return
    value = {"kind": kind, "seconds": seconds, **({"case_id": case_id} if case_id is not None else {})}
    with Path(path).open("ab") as stream:
        stream.write((canonical(value) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


def evaluate_bridge(runtime, assets, panel, observations, spec, deadline, *, script="evaluate-actual.mjs", timings=None):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(assets["tokenizer"][0]))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    if tokenizer.decode([50], skip_special_tokens=False) != "<start_function_response>":
        raise ValueError("The tokenizer stop token changed")
    command = ["node", str(ROOT / script), str(assets["registry"][0]), str(panel), str(observations),
               canonical(spec), str(RESERVE)]
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding="utf-8") as process:
        score, generated, after_reply = None, 0, None
        try:
            for line in process.stdout:
                if after_reply is not None:
                    append_timing(timings, "overhead", monotonic() - after_reply)
                    after_reply = None
                message = json.loads(line)
                if message["kind"] == "result":
                    if score is not None:
                        raise ValueError("Evaluator emitted multiple final results")
                    score = message["score"]
                    continue
                if message["kind"] != "generate" or score is not None:
                    raise ValueError("Unexpected evaluator message")
                if time() >= deadline:
                    raise BudgetLimit("Evaluation exhausted the shared time budget; saved prefix is retained")
                began = monotonic()
                tokens = tokenizer.encode(message["prompt"], add_special_tokens=False).ids
                if not 0 < len(tokens) <= 32768 - 512:
                    if script != "evaluate-actual.mjs":
                        raise ValueError("Known regression prompt overflow")
                    reply = {"raw_ids": [], "raw_text": "", "error": "context_overflow"}
                else:
                    raw = runtime.generate(tokens, max_new_tokens=512, stop_ids={50})
                    reply = {"raw_ids": raw, "raw_text": tokenizer.decode(raw, skip_special_tokens=False)}
                after_reply = monotonic()
                append_timing(timings, "generation", after_reply - began, case_id=message["case_id"])
                process.stdin.write(canonical(reply) + "\n")
                process.stdin.flush()
                generated += 1
                if generated == 1 or generated % 25 == 0:
                    print(f"[{observations.parent.name}] DEV generated {generated} new cases", flush=True)
            process.stdin.close()
            return_code = process.wait()
            if return_code or score is None or score.get("aborted"):
                raise RuntimeError(f"History evaluator did not complete (exit {return_code})")
            return score
        except BaseException:
            process.kill()
            process.wait()
            raise


def validate_score(score, observations, config, *, known=False):
    rows = read_observations(observations)
    if score.get("infrastructure_failures", 0) != len(infrastructure_failures(rows)):
        raise ValueError("Infrastructure failure count differs from the saved observations")
    if known:
        expected = config["regression"]["cases"]
        if score.get("observed_records") != expected or len(rows) != expected:
            raise ValueError("Known regression evaluation did not preserve its full denominator")
        return
    spec = config["evaluation"]
    expected = spec["fixed_prefix"] + spec["closed_loop"] + spec.get("challenge", 0)
    if score.get("observed_records") != expected or len(rows) != expected:
        raise ValueError("Fresh evaluation did not preserve all planned observations")
    panels = score["panels"]
    bundle = read(ROOT / config["assets"]["history_dev"]["path"])
    plan = {row["case_id"]: row for panel in ("fixed_prefix", "closed_loop", "challenge") if spec.get(panel)
            for row in bundle["panels"][panel]}
    if set(plan) != {row["case_id"] for row in rows}:
        raise ValueError("Evaluation case IDs differ from the frozen panel")
    for row in rows:
        original = plan[row["case_id"]]
        for key in ("panel", "cohort", "family_id", "bootstrap_family_id", "reference_eligible", "dialogue_id"):
            if row.get(key) != original.get(key):
                raise ValueError(f"Evaluation metadata differs from the frozen panel: {key}")
    for panel, count in ((key, spec[key]) for key in ("fixed_prefix", "closed_loop", "challenge") if spec.get(key)):
        observed = [row for row in rows if row["panel"] == panel]
        if len(observed) != count or sum(panels[panel][key]["planned"] for key in ("supported", "outside_scope")) != count:
            raise ValueError(f"Fresh {panel} evaluation is incomplete")
        for key, cohort in (("supported", "supported"), ("outside_scope", "outside_scope")):
            passed = sum(row.get("success") is True and row.get("critical_status") == "none_observed"
                         for row in observed if row["cohort"] == cohort)
            if panels[panel][key]["passed"] != passed:
                raise ValueError("Metric PASS counts differ from the saved observations")
        unresolved = sum(row["critical_status"] == "review_required" for row in observed)
        critical = [row for row in observed if row["critical_status"] == "detected"]
        kinds = {}
        for row in critical:
            key = row.get("critical_category") or "other"
            kinds[key] = kinds.get(key, 0) + 1
        if (panels[panel]["critical_review_required"] != unresolved
                or panels[panel]["critical_detected"] != len(critical)
                or panels[panel]["critical_types"] != kinds):
            raise ValueError("Critical counts differ from the saved observations")
        eligible = [row for row in observed if row["reference_eligible"]]
        if panels[panel]["exact_reference"]["planned"] != len(eligible) or panels[panel]["exact_reference"]["passed"] != sum(
                row.get("reference_success") is True for row in eligible):
            raise ValueError("Reference denominator or successes changed")
    summary = score["summary"]
    if summary != panels["closed_loop"]:
        raise ValueError("Primary summary must contain only the closed-loop panel")
    for metric_name, expected_count in (("supported", spec["supported"]), ("outside_scope", spec["outside_scope"]),
                                        ("whole_dialogues", spec["dialogues"]), ("standalone", spec["standalone"])):
        metric(summary, metric_name)
        if summary[metric_name]["planned"] != expected_count:
            raise ValueError(f"Changed fresh DEV denominator: {metric_name}")
    for row in rows:
        if not row.get("family_id") or row.get("critical_status") not in ("none_observed", "detected", "review_required"):
            raise ValueError("Every observation requires a frozen family and explicit critical status")
        if row.get("success") is True and row["critical_status"] != "none_observed":
            raise ValueError("Unresolved or critical observations cannot count as PASS")
    if summary["missing"] != 0:
        raise ValueError("The evaluator must record failed attempts, not remove planned cases")
    if panels["fixed_prefix"]["exact_reference"]["planned"] != spec["exact_reference"]:
        raise ValueError("Frozen fixed-prefix reference denominator changed")


class Stages:
    def __init__(self, directory, binding, budget, *, verify=lambda: None):
        self.directory, self.binding, self.budget = directory, binding, budget
        self.verify = verify
        (directory / "stages").mkdir(exist_ok=True)

    def run(self, name, action, validate=lambda value: None):
        self.verify()
        path = self.directory / "stages" / f"{name}.json"
        if path.exists():
            record = read(path)
            if record.get("identity_sha256") != self.binding:
                raise ValueError(f"Stage belongs to another experiment: {name}")
            validate(record["result"])
            return record["result"]
        with self.budget.phase(name) as deadline:
            result = action(deadline)
            validate(result)
        self.budget.check()
        publish_json(path, {"identity_sha256": self.binding, "result": result})
        return result


class ClusterBackend:
    def __init__(self, runner, args, config, original, assets, directory, receipt, hardware, budget, stages):
        self.runner, self.args, self.config, self.original = runner, args, config, original
        self.assets, self.directory, self.receipt, self.hardware = assets, directory, receipt, hardware
        self.budget, self.stages = budget, stages
        self.identities, self.experiments, self.schedules = {}, {}, {}

    def prepared(self, value):
        fingerprint = value["collection_receipt"]["fingerprint"]
        for key, asset in (("checkpoint_sha256", "initial_checkpoint"), ("registry_sha256", "registry"),
                           ("tokenizer_sha256", "tokenizer"), ("specs_sha256", "history_train_specs")):
            if fingerprint.get(key) != self.assets[asset][1]["sha256"]:
                raise ValueError(f"Collected TRAIN used a different {asset}")
        train, self.schedules = validate_prepared(value, self.directory, self.runner)
        self.assets["train"] = train
        totals = {}
        for seed in SEEDS:
            for arm in ("control", "treatment"):
                name = f"{arm}-{seed}"
                rows = self.schedules[name].rows
                totals[name] = {"prompt": sum(row[2] - row[4] for row in rows),
                                "target": sum(row[4] for row in rows)}
            for field in ("prompt", "target"):
                left, right = (totals[f"{arm}-{seed}"][field] for arm in ("control", "treatment"))
                if abs(right - left) > .05 * max(left, right):
                    raise ValueError(f"Matched arms exceed the frozen 5% {field}-token difference")
            left, right = (self.schedules[f"{arm}-{seed}"].rows for arm in ("control", "treatment"))
            if sum(a[0] == b[0] for a, b in zip(left, right)) != UPDATES * 2:
                raise ValueError("Exactly 50% of the paired draws must be identical common replay at identical positions")
        if any(self.schedules[f"{arm}-1729"].sha256 == self.schedules[f"{arm}-2718"].sha256
               for arm in ("control", "treatment")):
            raise ValueError("The second seed must use a separately frozen common order")
        self.identities, self.experiments = make_identities(self.runner, self.original, self.config, self.assets,
                  self.schedules, self.hardware, self.receipt, value)
        immutable(self.directory / "identities.json", self.identities)
        immutable(self.directory / "token-budget.json", totals)

    def job(self, name):
        run_id = f"{self.args.run_id}-{name}"
        return SimpleNamespace(runs=self.args.runs, run_id=run_id, devices=(0, 1, 2, 3),
                               pilot_run_id=f"{self.args.run_id}-pilot", progress=True,
                               resume=(self.args.runs / run_id / "run.json").is_file(),
                               fork_lineage=self.receipt["lineage"], optimizer_budget=self.budget.account)

    def pilot(self, deadline):
        identity = self.identities["control-1729"]
        job = self.job("pilot")
        try:
            measurements = self.runner.require_pilot(self.args.runs, job.run_id, identity)
        except (FileNotFoundError, ValueError):
            events = self.args.runs / job.run_id / "events.jsonl"
            if events.exists():
                for line in events.read_text().splitlines():
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event["event"] == "pilot_reload_verified":
                        raise ValueError("A previously completed pilot no longer matches its immutable inputs or checkpoint")
                    if event["event"] == "pilot_numerical_audit" and any(
                            event["details"][name].get("passed") is not True for name in ("parallel", "sequential_repeat")):
                        raise RuntimeError("The prior pilot failed numerical parity; automatic retries are forbidden")
            job.restart_pilot = True
            self.runner.train(job, self.identities["pilot"], self.experiments["pilot"], self.assets,
                              self.schedules["pilot"], lease_held=True, deadline=deadline)
            measurements = self.runner.require_pilot(self.args.runs, job.run_id, identity)
        capacities = {f"cuda:{item['index']}": item["memory_bytes"] for item in self.hardware["devices"]}
        if len(measurements) != 2:
            raise ValueError("The four-GPU pilot requires two complete audited updates")
        for index, item in enumerate(measurements):
            expected_cases = [row[0] for row in self.schedules["pilot"].rows[4 * index:4 * (index + 1)]]
            peaks = item.get("device_peak_bytes", {})
            if (item["case_ids"] != expected_cases or not math.isfinite(item["seconds"]) or item["seconds"] <= 0
                    or set(peaks) != set(capacities) or any(not 0 < peaks[key] < capacities[key] for key in peaks)):
                raise ValueError("Pilot timing, cases or physical memory evidence is incomplete")
        return {"measurements": measurements, "checkpoint": file_record(self.args.runs / job.run_id / "checkpoint.safetensors",
                                                                         relative_to=ROOT)}

    def train(self, name, milestone, deadline):
        job = self.job(name)
        path = self.args.runs / job.run_id / f"checkpoint-step-{INITIAL_STEP + milestone}.safetensors"
        if not path.exists():
            self.runner.train(job, self.identities[name], self.experiments[name], self.assets, self.schedules[name],
                              until=milestone, lease_held=True, deadline=deadline)
        self.validate_checkpoint(path, name, INITIAL_STEP + milestone)
        return {"checkpoint": file_record(path, relative_to=ROOT), "step": INITIAL_STEP + milestone}

    def validate_checkpoint(self, path, name, step):
        metadata = checkpoint_header(path).get("__metadata__", {})
        lineage = json.loads(metadata.get("lineage", "null"))
        if int(metadata.get("step", -1)) != step or lineage != {
                "identity_sha256": digest(self.identities[name]),
                "initial_checkpoint_sha256": self.assets["initial_checkpoint"][1]["sha256"]}:
            raise ValueError("Saved history checkpoint step or lineage differs")

    def validate_artifact(self, record):
        if file_record(contained(ROOT / record["path"]), relative_to=ROOT) != record:
            raise ValueError(f"Saved artifact changed: {record['path']}")

    def evaluate(self, name, checkpoint, deadline, *, known=False):
        import torch
        key = f"{name}-regression" if known else name
        directory = self.directory / "evaluations" / key
        directory.mkdir(parents=True, exist_ok=True)
        identity = {"run": self.identities[name], "checkpoint": file_record(checkpoint, relative_to=ROOT), "known": known}
        immutable(directory / "input.json", identity)
        observations = directory / "observations.jsonl"
        timings = directory / "timings.jsonl"
        timing_records(timings, repair=True)
        began = monotonic()
        runtime = self.runner.make_runtime(self.assets, checkpoint, (0,), batch_size=4)
        try:
            append_timing(timings, "load", monotonic() - began)
            if runtime.step != (INITIAL_STEP if name == "baseline" else INITIAL_STEP + (MIDPOINT if known else UPDATES)):
                raise ValueError("Evaluation checkpoint cursor differs")
            if runtime.tokenizer_sha256 != self.assets["tokenizer"][1]["sha256"]:
                raise ValueError("Evaluation tokenizer differs")
            score = evaluate_bridge(runtime, self.assets,
                self.assets["regression_dev" if known else "history_dev"][0], observations,
                self.config["regression" if known else "evaluation"], deadline,
                script="evaluate-known.mjs" if known else "evaluate-actual.mjs", timings=timings)
        finally:
            runtime.close()
            del runtime
            torch.cuda.empty_cache()
        failed = infrastructure_failures(read_observations(observations))
        score.update(infrastructure_failures=len(failed), infrastructure_failure_case_ids=failed,
                     quality_valid=not failed)
        validate_score(score, observations, self.config, known=known)
        return {"score": score, "seconds": sum(row["seconds"] for row in timing_records(timings)),
                "timings": file_record(timings, relative_to=ROOT), "checkpoint": identity["checkpoint"],
                "observations": file_record(observations, relative_to=ROOT)}

    def validate_evaluation(self, record, *, known=False):
        self.validate_artifact(record["checkpoint"])
        self.validate_artifact(record["observations"])
        self.validate_artifact(record["timings"])
        if sum(row["seconds"] for row in timing_records(ROOT / record["timings"]["path"])) != record["seconds"]:
            raise ValueError("Evaluation time differs from its measured work receipts")
        validate_score(record["score"], ROOT / record["observations"]["path"], self.config, known=known)

    def outcomes(self, record):
        return {row["case_id"]: row for row in read_observations(ROOT / record["observations"]["path"])}

    def remaining_updates(self, name):
        path = self.args.runs / self.job(name).run_id / "checkpoint.safetensors"
        if not path.exists():
            return UPDATES
        step = int(checkpoint_header(path).get("__metadata__", {}).get("step", -1))
        if not INITIAL_STEP <= step <= INITIAL_STEP + UPDATES:
            raise ValueError("Saved training cursor is outside this experiment")
        self.validate_checkpoint(path, name, step)
        return INITIAL_STEP + UPDATES - step

    def space(self, evaluations=5):
        required = evaluations * EVALUATION_BYTES + 14 * (self.assets["initial_checkpoint"][1]["bytes"] + 1024 ** 2)
        self.runner.require_run_space(self.args.runs, required)


def estimate_package(measurements, baseline, backend, stages, seeds):
    seconds = max(item["seconds"] for item in measurements)
    evaluation = baseline["seconds"]
    if not math.isfinite(evaluation) or evaluation <= 0:
        raise ValueError("Baseline timing is invalid")
    updates, evaluation_units = 0, 0.0
    for seed in seeds:
        for arm in ("control", "treatment"):
            name = f"{arm}-{seed}"
            updates += backend.remaining_updates(name)
            evaluation_units += not (stages.directory / "stages" / f"{name}-final.json").exists()
            evaluation_units += .2 * (not (stages.directory / "stages" / f"{name}-regression.json").exists())
    seconds = 1.20 * (updates * seconds + (evaluation_units + 1) * evaluation) + 300
    return seconds, updates


def orchestrate(backend, stages, budget, initial):
    pilot = stages.run("pilot", backend.pilot, lambda value: backend.validate_artifact(value["checkpoint"]))
    baseline = stages.run("baseline", lambda deadline: backend.evaluate("baseline", initial, deadline),
                          backend.validate_evaluation)
    both, _ = estimate_package(pilot["measurements"], baseline, backend, stages, SEEDS)
    pairs, second_fits = [], both < budget.remaining_seconds
    for seed in SEEDS:
        if seed == SEEDS[1] and (not pairs[0]["decision"]["passed"] or not second_fits):
            break
        existing = all((stages.directory / "stages" / f"{arm}-{seed}-final.json").exists()
                       for arm in ("control", "treatment"))
        if not existing:
            needed, pending = estimate_package(pilot["measurements"], baseline, backend, stages, [seed])
            if budget.remaining_seconds < needed or budget.account["record"]["charged_updates"] + pending > MAX_UPDATES:
                return {"status": "budget_stopped", "baseline": baseline, "pairs": pairs,
                        "reason": "The remaining budget cannot fit both complete arms and their evaluations",
                        "exploratory": True, "product_acceptance": False}
            backend.space(evaluations=4 if seed == SEEDS[0] and second_fits else 2)
        arm_results = {}
        for arm in ("control", "treatment"):
            name = f"{arm}-{seed}"
            midpoint = stages.run(f"{name}-midpoint", lambda deadline: backend.train(name, MIDPOINT, deadline),
                                  lambda value: backend.validate_artifact(value["checkpoint"]))
            stages.run(f"{name}-regression", lambda deadline: backend.evaluate(name,
                       ROOT / midpoint["checkpoint"]["path"], deadline, known=True),
                       lambda value: backend.validate_evaluation(value, known=True))
            trained = stages.run(f"{name}-trained", lambda deadline: backend.train(name, UPDATES, deadline),
                                 lambda value: backend.validate_artifact(value["checkpoint"]))
            arm_results[arm] = stages.run(f"{name}-final", lambda deadline: backend.evaluate(name,
                       ROOT / trained["checkpoint"]["path"], deadline), backend.validate_evaluation)
        outcome = decision(baseline, arm_results["control"], arm_results["treatment"],
                           (backend.outcomes(arm_results["control"]), backend.outcomes(arm_results["treatment"])),
                           confirmatory=seed == SEEDS[1])
        pair = {"seed": seed, **arm_results, "decision": outcome}
        immutable(stages.directory / f"comparison-{seed}.json", pair)
        pairs.append(pair)
        estimate, pending = estimate_package(pilot["measurements"], baseline, backend, stages, [SEEDS[1]])
        second_fits = (estimate < budget.remaining_seconds
                       and budget.account["record"]["charged_updates"] + pending <= MAX_UPDATES)
    confirmed = len(pairs) == 2 and all(pair["decision"]["passed"] for pair in pairs)
    status = "confirmed_diagnostic_gain" if confirmed else pairs[0]["decision"]["status"]
    if len(pairs) == 1 and pairs[0]["decision"]["passed"]:
        status = "exploratory_gain_second_seed_not_run"
    return {"status": status, "baseline": baseline, "pairs": pairs, "exploratory": len(pairs) < 2,
            "product_acceptance": False, "selection": {
                "status": "recipe_confirmed" if confirmed else "retain_7205_as_diagnostic_reference",
                "checkpoint": pairs[0]["treatment"]["checkpoint"] if confirmed else baseline["checkpoint"],
                "rule": "Seed 1729 is predeclared; never select the luckier seed; no product acceptance"}}


def print_summary(result, path):
    print(f"Actual-history: {result['status']}. Diagnostic experiment; product acceptance: false.", flush=True)
    evaluated = [("baseline 7205", result["baseline"])] if result.get("baseline") else []
    evaluated.extend((f"{arm} seed {pair['seed']}", pair[arm])
                     for pair in result.get("pairs", []) for arm in ("control", "treatment"))
    for label, candidate in evaluated:
        score = candidate["score"]
        summary, panels = score["summary"], score["panels"]
        fields = []
        for name in ("supported", "whole_dialogues", "outside_scope"):
            value = summary[name]
            fields.append(f"{name} {value['passed']}/{value['planned']}")
        reference = panels["fixed_prefix"]["exact_reference"]
        fields.append(f"fixed_reference {reference['passed']}/{reference['planned']}")
        fields.append(f"critical {sum(panel['critical_detected'] for panel in panels.values())}")
        fields.append(f"unresolved {sum(panel['critical_review_required'] for panel in panels.values())}")
        fields.append(f"infrastructure_failures {score.get('infrastructure_failures', 0)}")
        print(f"{label}: " + "; ".join(fields), flush=True)
    selection = result.get("selection")
    if selection and selection.get("checkpoint"):
        print(f"Diagnostic checkpoint: {ROOT / selection['checkpoint']['path']}", flush=True)
    print(f"Report: {Path(path).resolve()}", flush=True)


def compare(args, runner):
    import history_parity
    import history_prepare
    if not SAFE_ID.fullmatch(args.run_id) or len(args.run_id) > 70:
        raise ValueError("History run ID must be a safe name of at most 70 characters")
    args.runs = contained(args.runs)
    config, original, assets = load(runner)
    hook_assets = {key: value[0] for key, value in assets.items()}
    history_prepare.check(config, hook_assets)
    subprocess.run(["node", str(ROOT / "evaluate-actual.mjs"), str(assets["registry"][0]),
                    str(assets["history_dev"][0]), "--check", canonical(config["evaluation"])], check=True)
    with execution_lease(args.runs):
        initial, receipt = source_checkpoint(args.runs, args.source_run_id, assets)
        assets["initial_checkpoint"] = (initial, file_record(initial, relative_to=ROOT))
        hardware = runner.four_gpu_identity()
        directory = contained(args.runs / args.run_id, args.runs)
        if directory.exists() and any(directory.iterdir()) and not (directory / "experiment.json").is_file():
            raise ValueError("The output directory already belongs to another run; it will not be changed")
        directory.mkdir(exist_ok=True)
        code_paths = ["run.py", "history.py", "history_prepare.py", "history_parity.py", "model.py", "ledger.py", "experiments.json",
                      "evaluate-actual.mjs", "evaluate-known.mjs", "evaluate.mjs", "experiments-history.json", "requirements.txt"]
        code_paths.extend(str(path.relative_to(ROOT)) for path in sorted((ROOT / "evaluation").rglob("*.mjs")))
        code_paths.extend(config.get("code_files", []))
        manifest = {"config": config, "original": original, "assets": {key: value[1] for key, value in assets.items()}, "origin": receipt,
                    "runtime": hardware, "code": code_record(ROOT, sorted(set(code_paths)))}
        immutable(directory / "experiment.json", manifest)
        binding = digest(manifest)
        def verify_inputs():
            for record in list(manifest["assets"].values()) + manifest["code"]["files"]:
                if file_record(contained(ROOT / record["path"]), relative_to=ROOT) != record:
                    raise ValueError(f"Frozen experiment input changed: {record['path']}")
        final = directory / "result.json"
        if final.exists():
            result = read(final)
            if result.get("identity_sha256") != binding:
                raise ValueError("Closed history result belongs to another experiment")
            verify_inputs()
            evaluations = ([result["baseline"]] if result.get("baseline") else [])
            evaluations.extend(pair[arm] for pair in result.get("pairs", []) for arm in ("control", "treatment"))
            for evaluated in evaluations:
                for key in ("checkpoint", "observations", "timings"):
                    record = evaluated[key]
                    if file_record(contained(ROOT / record["path"]), relative_to=ROOT) != record:
                        raise ValueError("A closed experiment artifact changed")
                validate_score(evaluated["score"], ROOT / evaluated["observations"]["path"], config)
                if sum(row["seconds"] for row in timing_records(ROOT / evaluated["timings"]["path"])) != evaluated["seconds"]:
                    raise ValueError("Closed evaluation timing differs from its measured receipts")
            print_summary(result, final)
            return result
        budget = Budget(directory / "budget.json", binding, 36 if args.hours is None else args.hours)
        stages = Stages(directory, binding, budget, verify=verify_inputs)
        runner.require_run_space(args.runs, EVALUATION_BYTES + 512 * 1024 ** 2)
        hook_assets["train"] = hook_assets["history_replay"]
        parity_dir = directory / "parity"
        parity_dir.mkdir(exist_ok=True)
        def verify_parity(value):
            if value.get("passed") is not True:
                raise RuntimeError("Independent official-runtime parity did not pass; training is blocked")
        try:
            stages.run("official-parity", lambda deadline: history_parity.verify(
                hook_assets, initial, parity_dir, deadline, budget), verify_parity)
            collection = directory / "collection"
            collection.mkdir(exist_ok=True)
            prepared = stages.run("prepare", lambda deadline: history_prepare.prepare(
                config, hook_assets, collection, initial, deadline, budget),
                lambda value: validate_prepared(value, directory, runner))
            backend = ClusterBackend(runner, args, config, original, assets, directory, receipt, hardware, budget, stages)
            backend.prepared(prepared)
            result = orchestrate(backend, stages, budget, initial)
        except (BudgetLimit, runner.BudgetExhausted) as error:
            baseline_path = directory / "stages" / "baseline.json"
            result = {"status": "budget_stopped", "reason": str(error), "exploratory": True,
                      "baseline": read(baseline_path)["result"] if baseline_path.exists() else None,
                      "pairs": [read(path) for path in sorted(directory.glob("comparison-*.json"))],
                      "product_acceptance": False, "selection": None}
        result.update(identity_sha256=binding, budget=budget.account["record"], TEST_opened=False)
        publish_json(final, result)
        print_summary(result, final)
        return result
