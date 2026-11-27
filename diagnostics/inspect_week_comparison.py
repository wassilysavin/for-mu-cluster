"""Print the saved weekly volume comparison."""

import sys

from .student_evidence import ROOT, digest, read


RUN_ID = "week-train-volume-v2"
MODELS = ("baseline", "matched", "full")
DEV_SIZES = {"fresh": 233, "legacy155": 155, "legacy1250": 1250}
PANELS = (
    ("DEV233 closed loop", "fresh", "closed_loop"),
    ("DEV233 fixed history", "fresh", "challenge"),
    ("DEV155", "legacy155", None),
    ("DEV1250 closed loop", "legacy1250", None),
)


def load_report(root):
    directory = root / "runs" / RUN_ID
    manifest = read(root, directory / "experiment.json")
    result = read(root, directory / "result.json")
    if (
        manifest["protocol"] != RUN_ID
        or result["status"] != "complete"
        or result["identity_sha256"] != digest(manifest)
        or result["TEST_opened"] is not False
        or result["models"].keys() != set(MODELS)
    ):
        raise ValueError("Expected the completed weekly volume comparison with its original manifest")
    for name in MODELS:
        scores = result["models"][name]["scores"]
        for split, size in DEV_SIZES.items():
            score = scores[split]
            if score["observed_records"] != size or score["summary"]["missing"] != 0:
                raise ValueError(f"Incomplete saved evaluation for {name}/{split}")
    return result


def fraction(summary, name):
    metric = summary[name]
    return f"{metric['passed']}/{metric['planned']}"


def render(result):
    lines = [
        "baseline = step 7991 | matched = 5,169 | full = 16,174",
        f"{'DEV panel':<23}{'model':<10}{'answer+state':>13}{'dialogues':>11}"
        f"{'outside':>9}{'critical':>10}{'review':>8}",
    ]
    for label, split, panel in PANELS:
        for name in MODELS:
            score = result["models"][name]["scores"][split]
            summary = score["panels"][panel] if panel else score["summary"]
            lines.append(
                f"{label:<23}{name:<10}{fraction(summary, 'supported'):>13}"
                f"{fraction(summary, 'whole_dialogues'):>11}{fraction(summary, 'outside_scope'):>9}"
                f"{summary['critical_detected']:>10}{summary['critical_review_required']:>8}"
            )
    comparison = result["comparison"]
    interval = comparison["paired_95pct"]
    lines.append(
        f"DEV233 closed loop, full minus matched: {100 * comparison['full_minus_matched']:+.1f} pp "
        f"(paired 95% CI {100 * interval['lower']:+.1f} to {100 * interval['upper']:+.1f} pp)"
    )
    if not comparison["infrastructure_clean"]:
        lines.append("Infrastructure failures were recorded in one or more evaluations.")
    return "\n".join(lines)


if __name__ == "__main__":
    try:
        print(render(load_report(ROOT)))
    except (OSError, ValueError, KeyError, TypeError) as error:
        sys.exit(f"Weekly report: {error}")
