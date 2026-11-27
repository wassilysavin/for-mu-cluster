"""Trace shared DEV1250 critical cases through verified saved dialogue prefixes."""

from itertools import chain

from eval_tools.diagnose_dev import expandRecords, full_observation, loadPanel
from eval_tools.evaluate_actual import _materialize
from eval_tools.io import records
from evaluation import api
from evaluation.actual_history import createActualHistoryHarness
from evaluation.history_train import renderPrompt
from evaluation.schema import canonical

from .inspect_week_comparison import MODELS
from .student_errors import _decode


def shared_critical_ids(saved):
    return {
        cid
        for cid, base in saved["baseline"].items()
        if base["critical_status"] != "detected"
        and all(saved[name][cid]["critical_status"] == "detected" for name in MODELS[1:])
    }


def prefixes(context, target_ids):
    last_turn = {}
    for row in context.plan:
        if row["case_id"] in target_ids and row.get("dialogue_id"):
            key = row["panel"], row["dialogue_id"]
            last_turn[key] = max(last_turn.get(key, 0), row["turn_index"])
    return [
        row
        for row in context.plan
        if row["case_id"] in target_ids
        or row["turn_index"] <= last_turn.get((row["panel"], row.get("dialogue_id")), -1)
    ]


def selected_records(paths, prefix):
    ids = [row["case_id"] for row in prefix]
    expected, selected = set(ids), {}
    for record in chain.from_iterable(map(records, paths)):
        cid = record["case_id"]
        if cid not in expected:
            continue
        if cid in selected:
            raise ValueError(f"Repeated dialogue observation: {cid}")
        selected[cid] = record
    if selected.keys() != expected:
        raise ValueError("The selected dialogue prefixes are incomplete")
    return [selected[cid] for cid in ids]


def replay_baseline(context, prefix, saved):
    core = api()
    harness = createActualHistoryHarness(core)
    sessions = {}
    for row, observation in zip(prefix, saved, strict=True):
        key = row["panel"], row.get("dialogue_id")
        before = sessions.get(key) or harness.initial(context.registry, row)
        resolved = harness.resolve(row, before)
        prompt = renderPrompt(context.registry, core, before, resolved["question"])
        replay = full_observation(
            context, harness, row, before, resolved, prompt, observation["generation"], _materialize
        )
        if canonical(replay) != canonical(observation):
            raise ValueError(f"Baseline replay differs at {row['case_id']}")
        if row.get("dialogue_id"):
            sessions[key] = replay["after"]
        yield replay


def replay_prefixes(context, prefix, paths, saved):
    models = {}
    for name in MODELS:
        rows = selected_records(paths[name], prefix)
        if name == "baseline":
            replay = replay_baseline(context, prefix, rows)
        else:
            workers = {row["generation"].get("diagnostic_worker_sha256") for row in rows}
            if len(workers) != 1 or not next(iter(workers)):
                raise ValueError(f"Dialogue observations for {name} have different saved bindings")
            replay = expandRecords(context, rows, {"binding": workers.pop()})
        models[name] = {row["case_id"]: row for row in replay}
        for cid, row in models[name].items():
            expected = saved[name][cid]
            if any(row[key] != expected[key] for key in ("success", "critical_status")):
                raise ValueError(f"Replayed judgment differs for {name}/{cid}")
    return models


def exchange(row):
    return row["after"]["exchanges"][-1]


def differences(base, arm):
    left, right = exchange(base), exchange(arm)
    changes = [
        label
        for label, key in (("outcome", "outcome"), ("answer", "assistant"), ("error", "error_code"))
        if left.get(key) != right.get(key)
    ]
    if canonical(base["after"]["semantic"]) != canonical(arm["after"]["semantic"]):
        changes.append("state")
    return changes


def clipped(value, limit):
    text = " ".join(value.split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def call_summary(row):
    turn = exchange(row)
    calls = _decode(turn)
    if calls is None:
        return turn.get("error_code") or turn["outcome"]
    pieces = [clipped(f"{call['name']} {canonical(call['arguments'])}", 100) for call in calls[:3]]
    return " + ".join([*pieces, *(["…"] if len(calls) > 3 else [])]) or turn["outcome"]


def verdict(row):
    if row["success"] and row["critical_status"] == "none_observed":
        return "pass"
    return row.get("critical_category") or row.get("failure_category") or row["critical_status"]


def reference_summary(row):
    return (
        f"{verdict(row)} expected={row['expected_reference_targets'] or '-'}"
        f" selected={row['selected_reference_targets'] or '-'}"
    )


def trace_arm(rows, base, arm, name):
    first = None
    for row in rows:
        cid = row["case_id"]
        changed = differences(base[cid], arm[cid])
        if changed:
            first = (row, changed)
            break
    first_miss = next((arm[r["case_id"]] for r in rows if verdict(arm[r["case_id"]]) != "pass"), None)
    miss = f"t{first_miss['turn_index']} {verdict(first_miss)}" if first_miss else "none in prefix"
    target = arm[rows[-1]["case_id"]]
    lines = [
        f"  {name} | first state/answer difference: "
        + (f"t{first[0]['turn_index']} ({', '.join(first[1])})" if first else "none in prefix")
        + f" | first non-pass: {miss} | target {reference_summary(target)}"
    ]
    if target["question"] != base[rows[-1]["case_id"]]["question"]:
        lines.append(f"    {name} target question: {clipped(target['question'], 130)}")
    if first:
        cid = first[0]["case_id"]
        left, right = base[cid], arm[cid]
        lines.append(f"    t{first[0]['turn_index']} question: {clipped(right['question'], 105)}")
        if left["question"] != right["question"]:
            lines.append(f"    7991 question at that turn: {clipped(left['question'], 105)}")
        lines.append(
            f"    7991 [{verdict(left)}]: {call_summary(left)} | {name} [{verdict(right)}]: {call_summary(right)}"
        )
        if "answer" in first[1]:
            old, new = (clipped(exchange(item)["assistant"] or "", 75) for item in (left, right))
            lines.append(f"    answers 7991: {old or '-'} | {name}: {new or '-'}")
    return lines


def render_dialogues(registry_path, bundle_path, plan, saved, paths):
    """Reconstruct affected dialogues up to the marked turn using saved generations."""
    target_ids = shared_critical_ids(saved)
    lines = [f"DEV1250 closed loop | shared new critical flags: {len(target_ids)}"]
    if not target_ids:
        return lines
    context = loadPanel(registry_path, bundle_path)
    prefix = prefixes(context, target_ids)
    if any(row != plan[row["case_id"]] or row["panel"] != "closed_loop" for row in prefix):
        raise ValueError("Dialogue trace differs from the authenticated DEV plan")
    if not target_ids <= {row["case_id"] for row in prefix}:
        raise ValueError("A shared critical case is missing from the DEV plan")
    models = replay_prefixes(context, prefix, paths, saved)
    for target in prefix:
        cid = target["case_id"]
        if cid not in target_ids:
            continue
        dialogue = target.get("dialogue_id")
        rows = [
            row
            for row in prefix
            if row["case_id"] == cid
            or (dialogue and row.get("dialogue_id") == dialogue and row["turn_index"] < target["turn_index"])
        ]
        baseline = models["baseline"][cid]
        lines.append(f"{cid} | {target['policy']['kind']} | 7991 target {reference_summary(baseline)}")
        lines.append(f"  7991 target question: {clipped(baseline['question'], 130)}")
        for name in MODELS[1:]:
            lines.extend(trace_arm(rows, models["baseline"], models[name], name))
    return lines
