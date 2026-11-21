"""Validate, collect and materialize frozen prefixes, preserving interrupted receipts."""

import argparse
import hashlib
import json
from importlib.util import module_from_spec, spec_from_file_location
import os
import sys
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

import evaluation
from evaluation.history_control import matchedControl
from evaluation.history_train import applyObservation, materializeScenario, renderPrompt
from .common import ROOT, check, compact, rows, send, sha


def load_runtime(path):
    if path.resolve() == (ROOT / "evaluation").resolve():
        return evaluation.api()
    source = path.resolve() / "__init__.py"
    spec = spec_from_file_location(
        "_gemprf_history_runtime", source, submodule_search_locations=[str(source.parent)]
    )
    if spec is None or spec.loader is None:
        raise ValueError(f"Invalid Python evaluation runtime: {path}")
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.api()


def validate_observation(item):
    ids = item["raw_ids"]
    check(item["kind"] in ("native", "incomplete") and isinstance(item["raw_output"], str))
    check(isinstance(ids, list) and 0 < len(ids) <= 512 and all(type(i) is int and 0 <= i < 262144 for i in ids))
    check(50 not in ids[:-1])
    if item["kind"] == "native":
        check(ids[-1] == 50)
    else:
        check(len(ids) == 512 and 50 not in ids)


def line(value):
    return (compact(value) + "\n").encode("utf-8")


def durable(handle, data):
    handle.write(data)
    handle.flush()
    os.fsync(handle.fileno())


def read_receipts(path, immutable, prefixes, reg, api):
    receipts = {}
    if not path.exists():
        return receipts
    raw = path.read_bytes()
    last = raw.rfind(b"\n") + 1
    if last < len(raw):
        tail = raw[last:]
        preserved = path.with_name(f"{path.name}.interrupted-tail-{sha(tail)}.bin")
        if preserved.exists():
            check(preserved.read_bytes() == tail, "Interrupted tail evidence differs")
        else:
            with preserved.open("xb") as out:
                durable(out, tail)
        with path.open("r+b") as out:
            out.truncate(last)
            os.fsync(out.fileno())
        send({"kind": "receipt_tail_recovered", "truncated_bytes": len(tail), "preserved": str(preserved)})
    for record in rows(path):
        key = record["prefix_id"]
        check(all(record[k] == v for k, v in immutable.items()) and key not in receipts and key in prefixes)
        spec = prefixes[key]
        observations = record["observations"]
        check(len(observations) == len(spec["prefix"]), "Partial prefix stored as complete")
        before = api.createSession(reg, {"sessionId": spec["session_id"], "dialogueVersion": 2})
        for step, item in zip(spec["prefix"], observations):
            validate_observation(item)
            question = step["question"]
            check(
                question == item["question"]
                and sha(renderPrompt(reg, api, before, question)) == item["model_input_sha256"]
            )
            before = applyObservation(reg, api, before, question, item)
        check(before == record["actual_session"], "Stored actual session differs from replayed observations")
        receipts[key] = record
    return receipts


def output_rows(args, specs, reg, api, immutable, receipts=None):
    hashes = [hashlib.sha256(), hashlib.sha256()]
    count = identity = 0
    branches, target_ids, outcomes = Counter(), Counter(), Counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        out = stack.enter_context(args.output.open("xb"))
        control_out = stack.enter_context(args.control_output.open("xb")) if args.control_output else None
        for spec in specs:
            receipt = receipts[spec["prefix_id"]] if receipts is not None else None
            row = materializeScenario(reg, api, spec, receipt["observations"] if receipt else None)
            row["model_input"] = renderPrompt(reg, api, row["before"], row["question"])
            row["provenance"].update(immutable)
            if receipt:
                row["provenance"].update(
                    model_fingerprint=receipt["model_fingerprint"],
                    observed_receipt_sha256=sha(line(receipt)),
                    reconstructed_prefix=False,
                )
            raw = line(row)
            out.write(raw)
            hashes[0].update(raw)
            count += 1
            if control_out:
                control = matchedControl(reg, api, spec, row)
                control["model_input"] = renderPrompt(reg, api, control["before"], control["question"])
                raw = line(control)
                control_out.write(raw)
                hashes[1].update(raw)
                identity += bool(control["match"]["identity_order_pair"])
            branches[row["branch"]] += 1
            target_ids.update(row["reference_targets"])
            outcomes.update(e["outcome"] for e in row["before"]["exchanges"])
        for handle in (out, control_out):
            if handle:
                handle.flush()
                os.fsync(handle.fileno())
    return {
        **immutable,
        "rows": count,
        "sha256": hashes[0].hexdigest(),
        "branches": dict(branches),
        "target_ids": dict(target_ids),
        **(
            {
                "control_sha256": hashes[1].hexdigest(),
                "control_rows": count,
                "identity_order_pairs": identity,
                "changed_order_pairs": count - identity,
            }
            if args.control_output
            else {}
        ),
        "input_exchange_occurrences": dict(outcomes),
        "actual_observed_rows": count if receipts is not None else 0,
        "reconstructed_rows": count if receipts is None else 0,
        "model_output_used_as_gold": False,
        "tokenizer_invoked": False,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("check", "collect", "materialize"))
    for name in ("specs", "registry", "runtime", "output", "control-output", "receipts", "model-fingerprint"):
        parser.add_argument("--" + name, type=Path, required=name in ("specs", "registry", "runtime"))
    args = parser.parse_args()
    api = load_runtime(args.runtime)
    bundle = json.loads(args.registry.read_bytes())
    reg = api.createRegistry(bundle["knowledge"], bundle["contracts"])
    raw = args.specs.read_bytes()
    specs = [json.loads(s) for s in raw.splitlines()]
    check(len({s["case_id"] for s in specs}) == len(specs))
    prefixes = {s["prefix_id"]: s for s in specs}
    immutable = {"specs_sha256": sha(raw), "registry_sha256": reg.fingerprint()}
    if args.command == "check":
        for spec in specs:
            materializeScenario(reg, api, spec)
        send(
            {
                "kind": "check",
                **immutable,
                "cases": len(specs),
                "prefixes": len(prefixes),
                "requests": sum(len(s["prefix"]) for s in prefixes.values()),
                "tokenizer_invoked": False,
            }
        )
        return
    check(args.output, "--output is required")
    if args.command == "materialize":
        send({"kind": "result", **output_rows(args, specs, reg, api, immutable)})
        return
    check(args.receipts and args.model_fingerprint, "Collection requires --receipts and --model-fingerprint")
    fingerprint = json.loads(args.model_fingerprint.read_bytes())
    check(fingerprint.get("checkpoint_sha256") and fingerprint.get("checkpoint_step") == 7205)
    receipts = read_receipts(args.receipts, immutable, prefixes, reg, api)
    check(all(r["model_fingerprint"] == fingerprint for r in receipts.values()), "Changed collection model")
    args.receipts.parent.mkdir(parents=True, exist_ok=True)
    for key, spec in prefixes.items():
        if key in receipts:
            continue
        before = api.createSession(reg, {"sessionId": spec["session_id"], "dialogueVersion": 2})
        observations = []
        for i, step in enumerate(spec["prefix"]):
            question = step["question"]
            prompt = renderPrompt(reg, api, before, question)
            cid = f"{key}-e{i + 1}"
            send(
                {
                    "kind": "generate",
                    "case_id": cid,
                    "scenario_id": spec["case_id"],
                    "prefix_id": key,
                    "prefix_index": i,
                    "prompt": prompt,
                }
            )
            reply = sys.stdin.readline()
            check(reply, "Model provider disconnected")
            reply = json.loads(reply)
            check(not reply.get("case_id") or reply["case_id"] == cid)
            ids = reply["raw_ids"]
            check(isinstance(ids, list) and ids)
            item = {
                "kind": "native" if ids[-1] == 50 else "incomplete",
                "question": question,
                "raw_output": reply["raw_text"],
                "raw_ids": ids,
                "model_input_sha256": sha(prompt),
            }
            validate_observation(item)
            before = applyObservation(reg, api, before, question, item)
            observations.append(item)
        record = {
            **immutable,
            "prefix_id": key,
            "model_fingerprint": fingerprint,
            "observations": observations,
            "actual_session": before,
            "completed_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        }
        with args.receipts.open("ab") as out:
            durable(out, line(record))
        receipts[key] = record
        send({"kind": "prefix_complete", "prefix_id": key, "completed": len(receipts), "planned": len(prefixes)})
    result = output_rows(args, specs, reg, api, immutable, receipts)
    manifest = args.output.with_name(args.output.name + ".manifest.json")
    with manifest.open("x", encoding="utf-8") as out:
        json.dump({**result, "model_fingerprint": fingerprint}, out, ensure_ascii=False, indent=2)
        out.write("\n")
    send({"kind": "result", **result})


if __name__ == "__main__":
    main()
