"""Compare the stored weekly regressions against step 7991 on the same cases."""

from collections import Counter
import json
import sys

from evaluation.evaluation import summarizeEvaluation

from .inspect_week_comparison import MODELS, RUN_ID, load_report
from .saved_transfer import RUN as TRANSFER, stage_paths
from .student_evidence import ROOT, digest, inventory, read, verify


DEV155 = "assets/dev-volume-v1.jsonl"
DEV1250 = "assets/actual-history-v3-dev.json"
SPLITS = ("legacy155", "legacy1250")
MAX_EXAMPLES = 2


def json_lines(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def stage_files(root, directory, name, split, model, manifest_sha):
    baseline = name == "baseline"
    folder = root / TRANSFER / name if baseline else directory / name
    binding = digest({"manifest": manifest_sha, "checkpoint": model["checkpoint"], "model": name})
    if read(root, folder / "input.json") != {"binding": binding, "checkpoint": model["checkpoint"]}:
        raise ValueError(f"Different saved input for {name}")
    receipt = read(root, folder / f"{split}.json")
    checksum = receipt.pop("receipt_sha256")
    if digest(receipt) != checksum or receipt["binding"] != binding or receipt["score"] != model["scores"][split]:
        raise ValueError(f"Different saved evaluation receipt for {name}/{split}")
    if baseline or split == "legacy155":
        paths = stage_paths(root, folder, name, split)
    else:
        paths = [folder / f"history-{i}.jsonl" for i in range(4)]
    return [verify(root, path, entry) for path, entry in zip(paths, receipt["observations"], strict=True)]


def observations(paths, plan, split, expected_summary):
    found = {}
    known = split == "legacy155"
    for path in paths:
        for record in json_lines(path):
            case_id = record["case_id"]
            if case_id in found or case_id not in plan:
                raise ValueError(f"Unknown or repeated {split} case: {case_id}")
            case = plan[case_id]
            grade = record["result"] if known else record
            verdict = grade["critical_status"]
            if (
                grade["case_id"] != case_id
                or type(grade["success"]) is not bool
                or verdict not in {"none_observed", "review_required", "detected"}
                or (
                    not known
                    and any(
                        record.get(k) != case.get(k)
                        for k in ("panel", "dialogue_id", "turn_index", "cohort", "family_id")
                    )
                )
            ):
                raise ValueError(f"Saved judgment does not match its case: {case_id}")
            exchange = record.get("delivered")
            if "delivered" not in record and record.get("status") != "blocked":
                exchanges = record["after"]["exchanges"]
                exchange = exchanges[-1] if exchanges else None
            reason = grade.get("critical_category") or grade.get("failure_category")
            reason = reason or (record.get("known_review") or {}).get("category")
            if not reason:
                reason = (exchange or {}).get("error_code") or verdict
                if reason == "none_observed":
                    reason = (grade.get("reasons") or ["unspecified"])[0]
            found[case_id] = {
                "case_id": case_id,
                "success": grade["success"],
                "critical_status": verdict,
                "reason": reason,
                "history": record.get("before_visible_history_sha256"),
                "question": record.get("question"),
                "expected": grade.get("expected_reference_targets") or [],
                "selected": grade.get("selected_reference_targets") or [],
            }
    if found.keys() != plan.keys() or (known and list(found) != list(plan)):
        raise ValueError(f"Saved {split} observations differ from the planned cases")
    panel = [case for case in plan.values() if known or case["panel"] == "closed_loop"]
    selected = {case["case_id"]: found[case["case_id"]] for case in panel}
    summary = summarizeEvaluation(panel, selected.values())
    for name in ("supported", "whole_dialogues", "outside_scope"):
        if any(summary[name][key] != expected_summary[name][key] for key in ("passed", "planned")):
            raise ValueError(f"Saved {split} {name} does not reproduce")
    for name in ("missing", "critical_detected", "critical_review_required"):
        if summary[name] != expected_summary[name]:
            raise ValueError(f"Saved {split} {name} does not reproduce")
    return selected


def load_evidence(root):
    result = load_report(root)
    directory = root / "runs" / RUN_ID
    manifest = read(root, directory / "experiment.json")
    parent_path = verify(root, root / TRANSFER / "result.json", manifest["baseline_source"])
    parent = read(root, parent_path)
    if parent["baseline"] != result["models"]["baseline"]:
        raise ValueError("Baseline metrics differ from the authenticated source")
    assets = inventory(manifest["regressions"])
    known = verify(root, root / DEV155, assets[DEV155])
    actual = verify(root, root / DEV1250, assets[DEV1250])
    plans = {
        "legacy155": {row["case_id"]: row for row in json_lines(known)},
        "legacy1250": {row["case_id"]: row for rows in read(root, actual)["panels"].values() for row in rows},
    }
    saved = {split: {} for split in SPLITS}
    journals = {split: {} for split in SPLITS}
    for name in MODELS:
        model = result["models"][name]
        binding = parent["identity_sha256"] if name == "baseline" else result["identity_sha256"]
        for split in SPLITS:
            paths = stage_files(root, directory, name, split, model, binding)
            journals[split][name] = paths
            saved[split][name] = observations(paths, plans[split], split, model["scores"][split]["summary"])
    return plans, saved, journals


def load(root):
    plan, saved, _ = load_evidence(root)
    return plan, saved


def passed(row):
    return row["success"] and row["critical_status"] == "none_observed"


def labels(keys):
    counts = Counter(keys)
    return ", ".join(f"{key}={count}" for key, count in counts.most_common(5)) or "none"


def history_relation(before, after):
    if not before["history"] or not after["history"]:
        return "unknown"
    return "same" if before["history"] == after["history"] else "different"


def examples(ids, plan, before, after):
    # Pick one case per failure type before repeating a type.
    ordered = sorted(ids, key=lambda cid: (history_relation(before[cid], after[cid]) != "same", cid))
    chosen = list(dict.fromkeys(after[cid]["reason"] for cid in ordered))[:MAX_EXAMPLES]
    for reason in chosen:
        cid = next(cid for cid in ordered if after[cid]["reason"] == reason)
        old, row = before[cid], after[cid]
        question = " ".join((row["question"] or plan[cid]["question"]).split())
        if len(question) > 115:
            question = question[:114].rstrip() + "…"
        detail = (
            f"history={history_relation(old, row)}, baseline={old['critical_status']}, arm={row['critical_status']}"
        )
        if row["expected"] or row["selected"]:
            detail += f", expected={row['expected']}, selected={row['selected']}"
        yield f"    {cid} | {reason} | {detail} | {question}"


def render(plans, saved):
    lines = ["Saved paired judgments relative to step 7991."]
    for split in SPLITS:
        plan = plans[split]
        before = saved[split]["baseline"]
        losses = {}
        for name in MODELS[1:]:
            after = saved[split][name]
            if split == "legacy155":
                supported = [cid for cid in before if plan[cid]["cohort"] == "supported"]
                up = [cid for cid in supported if not passed(before[cid]) and passed(after[cid])]
                down = [cid for cid in supported if passed(before[cid]) and not passed(after[cid])]
                review = sum(after[cid]["critical_status"] == "review_required" for cid in down)
                lines.append(
                    f"DEV155 {name} | gained {len(up)}, lost {len(down)} supported passes"
                    f" | {review} losses await review"
                )
                lines.append("  Losses by case family: " + labels(plan[cid]["authoring_family_id"] for cid in down))
            else:
                old = {cid for cid, row in before.items() if row["critical_status"] == "detected"}
                new = {cid for cid, row in after.items() if row["critical_status"] == "detected"}
                down = [cid for cid in before if cid in new - old]
                pending = sum(before[cid]["critical_status"] == "review_required" for cid in down)
                lines.append(
                    f"DEV1250 closed loop {name} | {len(down)} newly flagged, {len(old - new)} no longer flagged"
                    f" | {pending} of the new flags were pending at baseline"
                )
                lines.append(
                    "  New flags by dialogue policy: " + labels(plan[cid]["policy"]["kind"] for cid in down)
                )
            lines.append("  Recorded categories: " + labels(after[cid]["reason"] for cid in down))
            lines.append(
                "  Visible history vs 7991: " + labels(history_relation(before[cid], after[cid]) for cid in down)
            )
            lines.extend(examples(down, plan, before, after))
            losses[name] = set(down)
        common = len(losses["matched"] & losses["full"])
        lines.append(
            f"{'DEV155 losses' if split == 'legacy155' else 'DEV1250 new flags'} shared by both arms: {common}"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    try:
        print(render(*load(ROOT)))
    except (OSError, ValueError, KeyError, TypeError) as error:
        sys.exit(f"Weekly failure report: {error}")
