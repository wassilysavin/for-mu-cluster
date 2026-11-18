"""Predeclared paired decision for the semantic-transfer experiment."""
import history

SEEDS = (1729, 2718)
MINIMUM_GAIN = 0.05
BOOTSTRAP_REPEATS = 10000


def panel_summaries(model):
    return {(split, panel): summary for split, score in model["scores"].items()
            for panel, summary in score.get("panels", {"all": score["summary"]}).items()}


def verify_primary(rows, summary):
    supported = [row for row in rows.values() if row["panel"] == "closed_loop" and row["cohort"] == "supported"]
    if any(type(row["success"]) is not bool or row["critical_status"] not in
           ("none_observed", "detected", "review_required") for row in supported):
        raise ValueError("Invalid paired verdict")
    passed = sum(row["success"] and row["critical_status"] == "none_observed" for row in supported)
    if (len(supported), passed) != (summary["supported"]["planned"], summary["supported"]["passed"]):
        raise ValueError("Primary score does not reproduce from saved observations")


def compare_pair(parent, ordinary, contrast, a_rows, b_rows, seed):
    if seed not in SEEDS:
        raise ValueError("Seed was not scheduled")
    summaries = {key: m["scores"]["fresh"]["summary"] for key, m in
                 (("parent", parent), ("ordinary", ordinary), ("contrast", contrast))}
    verify_primary(a_rows, summaries["ordinary"])
    verify_primary(b_rows, summaries["contrast"])
    interval = history.paired_bootstrap(a_rows, b_rows, repeats=BOOTSTRAP_REPEATS, seed=seed)
    gain = history.metric(summaries["contrast"], "supported") - history.metric(summaries["ordinary"], "supported")
    guards = {}
    models = {"parent": parent, "ordinary": ordinary, "contrast": contrast}
    panels = {key: panel_summaries(model) for key, model in models.items()}
    for label in ("parent", "ordinary"):
        if panels[label].keys() != panels["contrast"].keys():
            raise ValueError("Paired panel sets differ")
        for (split, panel), ref in panels[label].items():
            cur = panels["contrast"][split, panel]
            stem = f"{split}/{panel}"
            if ref["missing"] or cur["missing"]:
                raise ValueError("Unfinished comparison panel")
            for key in ("standalone", "outside_scope"):
                if cur[key]["planned"] != ref[key]["planned"]:
                    raise ValueError(f"Unequal paired {key} denominator")
                guards[f"{stem}/{key}/{label}"] = cur[key]["passed"] >= ref[key]["passed"]
            guards[f"{stem}/critical/{label}"] = cur["critical_detected"] <= ref["critical_detected"]
            for kind in sorted(cur["critical_types"].keys() | ref["critical_types"].keys()):
                guards[f"{stem}/critical_{kind}/{label}"] = cur["critical_types"].get(kind, 0) <= ref["critical_types"].get(kind, 0)
        whole = summaries["contrast"]["whole_dialogues"]
        ref = summaries[label]["whole_dialogues"]
        if whole["planned"] != ref["planned"]:
            raise ValueError("Whole dialogue denominators differ")
        guards[f"fresh/whole_dialogues/{label}"] = whole["passed"] > ref["passed"]
    clean = all(score["infrastructure_failures"] == 0 for model in models.values() for score in model["scores"].values())
    unresolved = {key: {split: sum(summary["critical_review_required"] for (s, _), summary in panels[key].items()
                                   if s == split) for split in model["scores"]}
                  for key, model in models.items()}
    effect = gain + 1e-12 >= MINIMUM_GAIN and interval["lower"] > 0 if seed == SEEDS[0] else gain > 0
    return {"seed": seed, "supported_gain": gain, "paired_95pct": interval, "guards": guards,
            "infrastructure_clean": clean, "unresolved": unresolved,
            "positive_signal": effect and clean and all(guards.values())}
