"""Measure what each completed semantic-transfer checkpoint learned on its own TRAIN."""

from source_provenance import runtime_sources
import argparse
from array import array
from collections import Counter
from hashlib import sha256
import json

from diagnose import EvaluatorRPC
from ledger import execution_lease, file_record, publish_json
from diagnostics.new_fit_inputs import iter_cases, load_inputs
from diagnostics.new_fit_workers import FitBudget, LIMIT, OUTPUT, PYTHON_REVISION, ROOT, RUNS, generate_arm
from private_runtime import private_directory
from diagnostics.student_evidence import digest, read

READER_FILES = (
    "check_new_train_fit.py", "diagnostics/new_fit_inputs.py", "diagnostics/new_fit_workers.py", "eval_tools/new_train_fit.py",
    "diagnostics/saved_transfer.py", "diagnostics/student_evidence.py", "transfer_train.py", "data_tools/transfer_contract.py",
    "data_tools/transfer_train.py", "eval_tools/diagnose_data.py", "data_tools/history_data.py", "eval_tools/evaluate_actual.py",
    "diagnose.py", "run.py", "model.py", "ledger.py", "history.py", "private_runtime.py",
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def freeze(path, value, budget):
    if path.is_symlink():
        raise ValueError(f"Unexpected symlink: {path}")
    if path.exists():
        require(read(ROOT, path) == value, f"Frozen TRAIN-fit record differs: {path}")
        return
    size_bound = len(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()) + 4096
    with budget.writing(size_bound):
        publish_json(path, value)


def choose_output(manifest):
    """Keep a pre-migration receipt intact when no generations were recorded."""
    receipt = OUTPUT / "experiment.json"
    if not receipt.exists() and not receipt.is_symlink():
        return OUTPUT
    require(receipt.is_file() and not receipt.is_symlink(), f"Invalid TRAIN-fit receipt: {receipt}")
    saved = read(ROOT, receipt)
    if saved == manifest:
        return OUTPUT

    require(isinstance(saved, dict), f"Invalid TRAIN-fit receipt: {receipt}")
    code = saved.get("code")
    same_inputs = {k: v for k, v in saved.items() if k != "code"} == {
        k: v for k, v in manifest.items() if k != "code"
    }
    require(same_inputs and isinstance(code, list) and any(
        isinstance(entry, dict) and entry.get("path") == "new_train_fit.mjs" for entry in code
    ), f"Frozen TRAIN-fit record differs: {receipt}")
    for path in OUTPUT.iterdir():
        if path == receipt:
            continue
        require(path.is_dir() and not path.is_symlink(), f"Previous TRAIN-fit run contains results: {path}")
        require(path.name == PYTHON_REVISION.parts[0] or (path.name in saved["arms"] and not any(path.iterdir())),
                f"Previous TRAIN-fit run contains results: {path}")
    return OUTPUT / PYTHON_REVISION


def text_sha(value):
    return sha256(value.encode("utf-8")).hexdigest()


def prepare(inputs, tokenizer, rpc):
    """Bind the frozen semantic case to the exact token sequence seen by the trainer."""
    prepared = {}
    for arm in inputs["arms"]:
        for row in iter_cases(inputs, arm):
            cid = row["case_id"]
            if cid in prepared:
                require(prepared[cid]["row_sha256"] == row["row_sha256"], f"Conflicting TRAIN rows: {cid}")
                continue
            ids, cut = row["input_ids"], row["prompt_tokens"]
            prompt = tokenizer.decode(ids[:cut], skip_special_tokens=False)
            target = tokenizer.decode(ids[cut:], skip_special_tokens=False)
            require(tokenizer.encode(prompt + target, add_special_tokens=False).ids == ids,
                    f"TRAIN input and target do not round-trip: {cid}")
            evidence = rpc.request({"action": "prepare", "case_id": cid, "prompt": prompt, "target": target},
                                   "prepared")
            require(evidence["case_id"] == cid and evidence["section"] == row["section"]
                    and evidence["group"] == row["group"] and evidence["prompt_sha256"] == text_sha(prompt)
                    and evidence["target_sha256"] == text_sha(target), f"Semantic TRAIN identity differs: {cid}")
            prepared[cid] = {**evidence, "row_sha256": row["row_sha256"], "input_tokens": cut}
    return prepared


def content_keys(arm, prepared):
    return {(prepared[r["case_id"]]["prompt_sha256"], prepared[r["case_id"]]["target_sha256"])
            for r in arm["cases"] if r["section"] == "variable"}


def metric(rows):
    return {"total": len(rows), "passed": sum(row["passed"] for row in rows),
            "canonical_calls": sum(row["canonical_call_exact"] for row in rows),
            "literal_exact": sum(row["native_literal_exact"] for row in rows),
            "valid_transport": sum(row["transport_valid"] for row in rows),
            "critical": sum(row["critical_status"] == "detected" for row in rows),
            "unresolved": sum(row["critical_status"] == "review_required" for row in rows),
            "length_stops": sum(row["finish_reason"] == "length" for row in rows),
            "failures": dict(sorted(Counter(row["failure_category"] for row in rows if not row["passed"]).items()))}


def assess(fit):
    if not fit["total"]:
        return "no_cases"
    if fit["passed"] * 10 >= 9 * fit["total"]:
        return "high_seen_fit"
    if (fit["passed"] + fit["unresolved"]) * 10 >= 9 * fit["total"]:
        return "resolve_reviews_first"
    return "inspect_misses_then_test_longer_training"


def score_arm(name, arm, other_keys, prepared, raw, rpc):
    require(set(raw) == {r["case_id"] for r in arm["cases"]}, f"Incomplete TRAIN generation: {name}")
    scores = []
    for meta in arm["cases"]:
        cid = meta["case_id"]
        case, generation = prepared[cid], raw[cid]
        require(generation["input_tokens"] == case["input_tokens"], f"Generation input length differs: {cid}")
        evidence = {**generation, "input_sha256": case["prompt_sha256"]}
        obs = rpc.request({"action": "score", "case_id": cid, "generation": evidence}, "scored")["observation"]
        require(all(obs[k] == case[k] for k in ("case_id", "section", "cohort", "group", "prompt_sha256",
                                                "target_sha256")), f"Scoring identity differs: {cid}")
        require(all(type(obs[k]) is bool for k in ("success", "canonical_call_exact", "native_literal_exact",
                                                    "transport_valid"))
                and obs["critical_status"] in {"none_observed", "review_required", "detected"}
                and (not obs["success"] or obs["critical_status"] == "none_observed"),
                f"Invalid evaluator verdict: {cid}")
        scope = "shared_replay"
        if case["section"] == "variable":
            key = (case["prompt_sha256"], case["target_sha256"])
            scope = "variable_common" if key in other_keys else "variable_unique"
        scores.append({**obs, "position": meta["position"], "subset": scope, "passed": obs["success"],
                       "finish_reason": generation["finish_reason"]})
    subsets = {scope: metric([r for r in scores if r["subset"] == scope])
               for scope in ("variable_unique", "variable_common", "shared_replay")}
    subsets["variable"] = metric([r for r in scores if r["section"] == "variable"])
    subsets["variable_supported"] = metric([r for r in scores if r["section"] == "variable" and r["cohort"] == "supported"])
    subsets["all"] = metric(scores)
    groups = {g: metric([r for r in scores if r["group"] == g]) for g in sorted({r["group"] for r in scores})}
    return {"checkpoint": arm["checkpoint"], "fresh_dev_saved": arm["fresh_dev"], "subsets": subsets,
            "groups": groups, "seen_variable_fit": assess(subsets["variable"]),
            "seen_unique_fit": assess(subsets["variable_unique"]), "scores": scores}


def evaluate(inputs, prepared, rpc, binding, output, lease, budget):
    groups = {name: content_keys(arm, prepared) for name, arm in inputs["arms"].items()}
    results = {}
    for name, arm in inputs["arms"].items():
        branch, seed = name.split("-")
        peer = ("ordinary" if branch == "contrast" else "contrast") + "-" + seed
        prompts = {row["case_id"]: array("I", row["input_ids"][:row["prompt_tokens"]])
                   for row in iter_cases(inputs, name)}
        print(f"{name}: evaluating {len(prompts)} saved TRAIN prompts on four GPUs", flush=True)
        raw = generate_arm(output, name, digest([binding, name]), arm["checkpoint"], inputs["base"],
                           inputs["tokenizer"], prompts, lease, budget)
        del prompts
        results[name] = score_arm(name, arm, groups[peer], prepared, raw, rpc)
        del raw
    return {"status": "complete", "experiment_sha256": binding, "step": 8151, "updates": 160, "arms": results}


def fraction(m):
    return f"{m['passed']}/{m['total']}"


def print_summary(result, path):
    for name, arm in result["arms"].items():
        subsets = arm["subsets"]
        dev = arm["fresh_dev_saved"]["supported"]
        dialogues = arm["fresh_dev_saved"]["whole_dialogues"]
        counts = ", ".join(f"{k} {fraction(subsets[k])}" for k in
                           ("variable", "variable_unique", "variable_common", "shared_replay"))
        print(f"{name}: {counts}. Saved fresh DEV: supported {dev['passed']}/{dev['planned']}, "
              f"whole dialogues {dialogues['passed']}/{dialogues['planned']}.")
        if subsets["variable_supported"]["total"] != subsets["variable"]["total"]:
            print("  New TRAIN supported: " + fraction(subsets["variable_supported"]))
        v = subsets["variable"]
        print(f"  New TRAIN: exact calls {v['canonical_calls']}/{v['total']}, "
              f"critical {v['critical']}, unresolved {v['unresolved']}, length stops {v['length_stops']}.")
        print("  By request: " + ", ".join(f"{g} {fraction(m)}" for g, m in arm["groups"].items()
                                           if g != "shared_replay"))
        print(f"  Fit at 90%: all new={arm['seen_variable_fit']}, unique={arm['seen_unique_fit']}.")
    print(f"Report: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, choices=(4,), default=4)
    parser.parse_args()
    from tokenizers import Tokenizer

    private_directory(ROOT, RUNS.relative_to(ROOT))
    with execution_lease(RUNS) as lease:
        print("Verifying the saved transfer result, checkpoints and exact TRAIN schedules...", flush=True)
        inputs = load_inputs()
        files = [ROOT / n for n in sorted({*READER_FILES, "source_provenance.py", *runtime_sources(ROOT)})]
        manifest = {"protocol": "new-train-fit-v1", "inputs_sha256": digest(inputs),
                    "source_result_sha256": inputs["source_result_sha256"], "train": inputs["train"],
                    "provenance": inputs["train_provenance"],
                    "arms": {name: {k: arm[k] for k in ("checkpoint", "identity_sha256", "schedule_sha256")}
                             for name, arm in inputs["arms"].items()},
                    "code": [file_record(p, relative_to=ROOT) for p in files],
                    "generation": {"max_new_tokens": LIMIT, "stop_token_id": 50, "mode": "native_greedy_fp32"},
                    "primary": "delivered answer and resulting dialogue state, excluding unresolved reviews"}
        output = choose_output(manifest)
        if output != OUTPUT:
            print(f"Preserving previous TRAIN-fit receipt. Python run: {output}", flush=True)
        budget = FitBudget(output)
        if not output.exists():
            new_directories = max(1, len(output.relative_to(OUTPUT).parts))
            budget.require(8192 * new_directories)
        private_directory(ROOT, output.relative_to(ROOT))
        freeze(output / "experiment.json", manifest, budget)
        decoder = Tokenizer.from_file(str(ROOT / inputs["tokenizer"]["path"]))
        decoder.no_padding()
        decoder.no_truncation()
        with EvaluatorRPC("eval_tools/new_train_fit.py", ROOT / inputs["registry"]["path"]) as rpc:
            rpc.request({"action": "init", "receipts": str(ROOT / inputs["prefix_receipts"]["path"]),
                         "contexts": str(ROOT / inputs["contexts"]["path"])}, "initialized")
            prepared = prepare(inputs, decoder, rpc)
            result = evaluate(inputs, prepared, rpc, digest(manifest), output, lease, budget)
        path = output / "result.json"
        freeze(path, result, budget)
        print_summary(result, path)


if __name__ == "__main__":
    main()
