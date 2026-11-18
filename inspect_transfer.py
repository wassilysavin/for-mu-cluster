"""Summarize discordant saved DEV outcomes for semantic-transfer-v1."""

from collections import Counter, defaultdict
from pathlib import Path
import sys

from saved_transfer import load_saved_transfer

ROOT = Path(__file__).resolve().parent
PAIRS = (("contrast", "ordinary"), ("ordinary", "parent"), ("contrast", "parent"))
FOCUS = ("retry", "ambiguous")


def passed(row):
    return row["success"] is True and row["critical_status"] == "none_observed"


def fraction(summary, key):
    metric = summary[key]
    return f"{metric['passed']}/{metric['planned']}"


def changes(case_ids, values, candidate, reference):
    gains, losses = [], []
    for cid in case_ids:
        current, other = values[candidate][cid], values[reference][cid]
        if current != other:
            (gains if current else losses).append(cid)
    return gains, losses


def counts(delta):
    gains, losses = delta
    return f"+{len(gains)}/-{len(losses)}"


def policy_group(case):
    kind = case["policy"]["kind"]
    if kind == "fresh":
        return "fresh/dialogue" if case.get("dialogue_id") else "fresh/standalone"
    return kind


def failure_type(row):
    return row.get("critical_category") or row.get("failure_category") or "unspecified"


def verdict(row):
    if passed(row):
        return "pass"
    reason = failure_type(row)
    if reason == "execution_error" and (row.get("delivered") or {}).get("error_code"):
        reason += "/" + row["delivered"]["error_code"]
    elif reason == "wrong_reference" and row.get("selected_reference_targets"):
        reason += "/" + ",".join(row["selected_reference_targets"])
    if row["critical_status"] == "review_required":
        return f"review({reason})"
    return reason


def failure_mix(ids, records):
    labels = Counter(failure_type(records[cid]) for cid in ids)
    return ", ".join(f"{key}={n}" for key, n in sorted(labels.items(), key=lambda item: (-item[1], item[0]))) or "none"


def example_lines(panel, delta, plan, observations):
    lines = []
    for direction, ids in zip(("gain", "loss"), delta, strict=True):
        if not ids:
            continue
        # Prefer the two motivating skills, then a like-for-like history and stable case order.
        def order(cid):
            kind = plan[cid]["policy"]["kind"]
            same = observations["contrast"][cid]["before_visible_history_sha256"] == observations["ordinary"][cid]["before_visible_history_sha256"]
            return (kind not in FOCUS, not same, cid)

        cid = min(ids, key=order)
        a, b = (observations[name][cid] for name in ("ordinary", "contrast"))
        same = a["before_visible_history_sha256"] == b["before_visible_history_sha256"]
        question = " ".join(plan[cid]["question"].split())
        if len(question) > 145:
            question = question[:144].rstrip() + "…"
        labels = "  ".join(f"{name}={verdict(rows[cid])}" for name, rows in observations.items())
        lines.append(f"  {panel} {direction} {cid} [{policy_group(plan[cid])}]  {question}")
        lines.append(f"    {labels}  history={'same' if same else 'different'} (contrast vs ordinary)")
    return lines


def render_seed(result, pair, plan, rows):
    seed = pair["seed"]
    models = {"parent": result["baseline"], **{key: pair["arms"][key] for key in ("ordinary", "contrast")}}
    observations = {"parent": rows["baseline"], **{key: rows[f"{key}-{seed}"] for key in ("ordinary", "contrast")}}
    values = {label: {cid: passed(row) for cid, row in cases.items()} for label, cases in observations.items()}
    panels = {"primary": [], "outside": [], "challenge": []}
    dialogues = defaultdict(list)
    for cid, case in plan.items():
        if case["panel"] == "challenge":
            panels["challenge"].append(cid)
        else:
            panel = "primary" if case["cohort"] == "supported" else "outside"
            panels[panel].append(cid)
            if case.get("dialogue_id"):
                dialogues[case["dialogue_id"]].append(cid)
    whole = {label: {key: all(cases[cid] for cid in ids) for key, ids in dialogues.items()} for label, cases in values.items()}

    lines = [f"Seed {seed} | fresh DEV ({len(plan)} cases)",
             "model          supported  dialogues  outside  crit/review  challenge  DEV155  DEV1250"]
    for label, model in models.items():
        scores = model["scores"]
        primary = scores["fresh"]["summary"]
        challenge = scores["fresh"]["panels"]["challenge"]
        legacy = [fraction(scores[split]["summary"], "supported") for split in ("legacy155", "legacy1250")]
        name = "parent 7991" if label == "parent" else label
        safety = f"{primary['critical_detected']}/{primary['critical_review_required']}"
        lines.append(f"{name:<15}{fraction(primary, 'supported'):<11}{fraction(primary, 'whole_dialogues'):<11}"
                     f"{fraction(primary, 'outside_scope'):<9}{safety:<13}{fraction(challenge, 'supported'):<11}"
                     f"{legacy[0]:<8}{legacy[1]}")
    infra = [f"{label}/{split}={score['infrastructure_failures']}" for label, model in models.items()
             for split, score in model["scores"].items() if score.get("infrastructure_failures", 0)]
    if infra:
        lines.append("Recorded infrastructure failures: " + ", ".join(infra))
    lines.extend(("crit/review = detected/unresolved on fresh closed-loop. DEV155 and DEV1250 show supported (DEV1250 closed-loop).",
                  "Candidate vs reference: gained/lost    primary    dialogues   outside    challenge"))
    for candidate, reference in PAIRS:
        label = f"{candidate} vs {reference}"
        primary = changes(panels["primary"], values, candidate, reference)
        dialogue = changes(dialogues, whole, candidate, reference)
        outside = changes(panels["outside"], values, candidate, reference)
        fixed = changes(panels["challenge"], values, candidate, reference)
        lines.append(f"  {label:<37}{counts(primary):<11}{counts(dialogue):<12}{counts(outside):<11}{counts(fixed)}")

    interval = pair["decision"]["paired_95pct"]
    lines.append(f"Primary contrast minus ordinary: {pair['decision']['supported_gain']*100:+.1f} pp "
                 f"(95% family-bootstrap CI {100*interval['lower']:+.1f} to {100*interval['upper']:+.1f} pp).")
    lines.append("Policy (supported turns)           n   contrast/ordinary  ordinary/parent  contrast/parent")
    examples = []
    primary_delta = changes(panels["primary"], values, *PAIRS[0])
    for panel in ("primary", "challenge"):
        subset = panels[panel]
        groups = defaultdict(list)
        for cid in subset:
            groups[policy_group(plan[cid])].append(cid)
        for kind, ids in sorted(groups.items()):
            deltas = [changes(ids, values, *pair_names) for pair_names in PAIRS]
            if kind in FOCUS or any(gain or loss for gain, loss in deltas):
                numbers = [counts(d) for d in deltas]
                lines.append(f"  {panel + '/' + kind:<33}{len(ids):<4}{numbers[0]:<18}{numbers[1]:<17}{numbers[2]}")
        delta = changes(subset, values, *PAIRS[0])
        examples.extend(example_lines(panel, delta, plan, observations))
    changed = [[cid for cid in ids if observations["ordinary"][cid]["before_visible_history_sha256"] !=
                observations["contrast"][cid]["before_visible_history_sha256"]] for ids in primary_delta]
    lines.append(f"Primary flips on different visible histories: {counts(changed)} of {counts(primary_delta)}. "
                 "Each live dialogue follows that model's own replies.")
    gains, losses = primary_delta
    lines.append("Contrast gains, ordinary's recorded failures: " + failure_mix(gains, observations["ordinary"]))
    lines.append("Contrast losses, contrast's recorded failures: " + failure_mix(losses, observations["contrast"]))
    if examples:
        lines.append("Contrast vs ordinary examples (one per direction/panel where available):")
        lines.extend(examples)
    return lines


def report(result, plan, rows):
    lines = [f"semantic-transfer-v1 | saved result: {result['status']}"]
    for pair in result["pairs"]:
        lines.extend(render_seed(result, pair, plan, rows))
    return "\n".join(lines)


def main():
    try:
        _, result, plan, rows = load_saved_transfer(ROOT)
        print(report(result, plan, rows))
    except (OSError, ValueError, TypeError, KeyError, IndexError) as error:
        sys.exit(f"Cannot report saved transfer results: {error}")


if __name__ == "__main__":
    main()
