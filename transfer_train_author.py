"""Rebuild the readable TRAIN declarations from frozen TRAIN inputs only."""

from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import random

ASSETS = Path(__file__).resolve().parent / "assets"
ARMS = ("ordinary", "contrast")
MODES = ("retry_or_quote_failure", "retry_or_missing_answer", "quote_or_recompute",
         "ambiguous_address", "original_or_correction", "compound_scope", "same_parameter",
         "direction_or_flag", "argument_value", "outcome_changes_reference", "condition_omission", "revise_or_new")


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def gold(row):
    return {"kind": "gold", "question": row["question"], "calls": row["calls"],
            "ancestor_case_ids": [row["case_id"]]}


def action(kind, questions, **kwargs):
    return {"kind": kind, "questions": questions, **kwargs}


def addressed(kind, pos, a, b, **kwargs):
    return action(kind, [a, b], exchange_index=pos, **kwargs)


def choose_actions(mode, i, side, anchor, other, distractor, pairs):
    ord_word = ("opening", "second", "third")[i % 3]
    pos = i % 3 + 1
    before = [gold(r) for r in distractor[:pos - 1]]
    retry = addressed("answer_again", pos,
        f"Please answer my {ord_word} question again, keeping its original conditions.",
        f"Work through the question in my {ord_word} message anew, with the same settings.", calls=anchor["calls"])
    quote = addressed("quote_exchange", pos,
        f"Copy the entire reply I saw after my {ord_word} message, as it was shown.",
        f"Give me the exact message that came back for my {ord_word} request, including any error.")
    answer = addressed("quote_answer", pos,
        f"Read back the scientific answer from the reply to my {ord_word} request.",
        f"Can you quote the substantive answer I received for my {ord_word} question, word for word?", part=1)
    if mode.startswith("retry_or") or mode in ("quote_or_recompute", "outcome_changes_reference"):
        fail = mode.startswith("retry_or") or mode == "outcome_changes_reference" and side == i % 2
        step = gold(anchor)
        if fail:
            step = ({"kind": "provider_error", "error_code": "inference_failed"} if i % 2 else
                    {"kind": "malformed", "raw_output": "The computation failed."})
            step.update(question=anchor["question"], ancestor_case_ids=[anchor["case_id"]])
        prefix = [*before, step, gold(distractor[-1])]
        return prefix, (retry, quote if mode == "retry_or_quote_failure" else answer)
    if mode == "ambiguous_address":
        left, right = pairs["subject"][i]
        topic = left["calls"][0]["arguments"]["subject"]
        prefix = [gold(left), gold(distractor[-1]), gold(right)]
        return prefix, (action("clarify_reference", [
            f"I need one of your two replies about {topic} for my notes, but my note does not say which one. Can you retrieve the one I mean?",
            f"Could you read back the answer I meant on {topic}? I have two answers on that topic in mind and have not specified one."], reference_kind="answer"),
            addressed("quote_answer", 1 if i % 2 == 0 else 3,
                f"Quote the {'first' if i % 2 == 0 else 'last'} of the two answers about {topic}.",
                f"Read back the {'earlier' if i % 2 == 0 else 'later'} answer you gave about {topic}, exactly.", part=1))
    if mode == "original_or_correction":
        left, right = pairs["replace"][i]
        old, new = (r["calls"][0]["arguments"]["subject"] for r in (left, right))
        question = f"Please revise the first answer by changing the parameter from {old} to {new}, keeping the operation the same."
        step = {"kind": "gold", "question": question, "ancestor_case_ids": [left["case_id"], right["case_id"]],
                "policy": {"kind": "revise_answer", "exchange_index": 1, "part": 1, "question": question,
                           "changes": {"subject": new}, "expected_original": left["calls"][0], "replacement": right["calls"][0]}}
        return [gold(left), step, gold(distractor[-1])], (
            addressed("quote_answer", 1, "Read back your original answer, before I corrected the parameter.",
                      "I'd like the first version verbatim, as it stood before the correction.", part=1),
            addressed("quote_answer", 2, "Read back the revised answer you gave after I corrected the parameter.",
                      "Please quote the corrected version, exactly as you supplied it after my change.", part=1))
    if mode == "compound_scope":
        prefix = [gold(distractor[-1])]
        def obligations(rows):
            questions = [r["question"] for r in rows]
            return action("fresh", ["Please address each of the following in order. " + " ".join(questions),
                "My request has these parts, in this order. " + " ".join(questions)], calls=[c for r in rows for c in r["calls"]], source_case_ids=[r["case_id"] for r in rows])
        return prefix, (obligations([anchor]), obligations([anchor, other] if i % 2 == 0 else [anchor, other, distractor[0]]))
    if mode == "condition_omission":
        row = pairs["flag"][i]
        call = row["calls"][0]
        topic = call["arguments"]["subject"]
        value = str(call["arguments"]["value"]).lower()
        return [gold(distractor[-1])], (
            action("fresh", [f"What effect does my setting for {topic} have? I haven't told you whether I set it true or false. Ask me for the setting.",
                             f"I need the effect of my particular {topic} setting. Check which value I'm using before explaining."],
                   calls=[{"name": call["name"], "arguments": {"subject": topic, "_clarify": True, "_missing": ["value"]}}],
                   source_case_ids=[row["case_id"]]),
            action("fresh", [f"What effect does my setting for {topic} have? I've set it to {value}.",
                             f"I need the effect of my particular {topic} setting. I'm using {value}; what follows from that?"],
                   calls=row["calls"], source_case_ids=[row["case_id"]]))
    if mode == "revise_or_new":
        left, right = pairs["delta"][i]
        original, replacement = left["calls"][0], right["calls"][0]
        old, new = (v["arguments"]["subject"] for v in (original,replacement))
        return [gold(left), gold(distractor[-1])], (
            addressed("revise_answer", 1, f"In the first answer, substitute {new} for {old}; keep every other argument.",
                      f"Revise the opening answer by changing just its parameter to {new}, leaving the original conditions in place.",
                      part=1, expected_original=original, replacement=replacement, changes={"subject": new},
                      source_case_ids=[left["case_id"],right["case_id"]]),
            action("fresh", [f"Separately, without changing that earlier answer: {right['question']}",
                             f"Leave our first answer as it stands. For a new question, {right['question']}"],
                   calls=right["calls"], source_case_ids=[right["case_id"]]))
    left, right = pairs[{"same_parameter": "subject", "direction_or_flag": "switch", "argument_value": "value"}[mode]][i]
    def scientific(row):
        question = row["question"]
        call = row["calls"][0]
        args = call["arguments"]
        topic = args["subject"]
        if call["name"] == "explain_change_effect":
            verb = "increasing" if args["direction"] == "increase" else "decreasing"
            alt = f"What effect does the documented implementation predict from {verb} {topic}, and which assumptions qualify that prediction?"
        elif call["name"] == "explain_boolean_effect":
            alt = f"How does GEM-pRF behave with {topic} set to {str(args['value']).lower()}?"
        elif call["name"] == "explain_item_count_limits":
            alt = f"What {args['bound']} count of {topic} items is documented, including parser caveats?"
        else:
            alt = {"explain_purpose": f"What role does {topic} play in a GEM-pRF analysis?",
                   "get_value_guidance": f"What should guide my choice of a value for {topic}?",
                   "get_default_value": f"What default for {topic} do the documentation and example configuration support, and does the runtime supply it?",
                   "get_units": f"In which units is {topic} expressed, and what qualifications apply?",
                   "explain_version_format": "Which string representation does the loader accept for the XML version, and how is that attribute compared?"}[call["name"]]
        return action("fresh", [question, alt], calls=row["calls"], source_case_ids=[row["case_id"]])
    return [gold(distractor[-1])], (scientific(left), scientific(right))


def scientific_pairs(rows, rng):
    by_topic = defaultdict(list)
    for r in rows:
        call = r["calls"][0]
        if (set(call["arguments"]) == {"subject"} and call["name"] in
                {"explain_purpose", "get_units", "get_default_value", "explain_version_format", "get_value_guidance"}):
            by_topic[call["arguments"]["subject"]].append(r)
    eligible = [group for group in by_topic.values() if len(group) >= 2]
    rng.shuffle(eligible)
    subject = [rng.sample(group, 2) for group in eligible if group is not by_topic["version"]][:5]
    subject += [[next(r for r in by_topic["version"] if r["calls"][0]["name"] == name)
                 for name in ("explain_purpose", "explain_version_format")]]
    by_function = defaultdict(list)
    for row in rows:
        by_function[row["calls"][0]["name"]].append(row)
    replace = [rng.sample(by_function[name], 2) for name in
               ("get_units", "get_default_value", "explain_purpose", "get_value_guidance", "get_units", "get_default_value")]
    changes = by_function["explain_change_effect"]
    flags = by_function["explain_boolean_effect"]
    value = []
    for name, key in (("explain_change_effect", "direction"), ("explain_boolean_effect", "value"),
                      ("explain_item_count_limits", "bound")) * 2:
        candidates = [(a, b) for a in by_function[name] for b in by_function[name]
                      if a["calls"][0]["arguments"]["subject"] == b["calls"][0]["arguments"]["subject"]
                      and a["calls"][0]["arguments"][key] != b["calls"][0]["arguments"][key]]
        value.append(rng.choice(candidates))
    delta = []
    for name in ("explain_change_effect", "explain_conditional_applicability", "explain_boolean_effect") * 2:
        matches = [(a,b) for a in by_function[name] for b in by_function[name]
                   if a["calls"][0]["arguments"]["subject"] != b["calls"][0]["arguments"]["subject"]
                   and {k:v for k,v in a["calls"][0]["arguments"].items() if k != "subject"}
                   == {k:v for k,v in b["calls"][0]["arguments"].items() if k != "subject"}]
        delta.append(rng.choice(matches))
    return {"flag": rng.sample(flags, 6), "delta": delta, "subject": subject, "replace": replace, "switch": list(zip(rng.sample(changes, 6), rng.sample(flags, 6))), "value": value}


def write_lines(name, rows):
    (ASSETS / f"transfer-train-{name}.jsonl").write_text("".join(compact(row) + "\n" for row in rows))


def build():
    broad = ASSETS / "train-full-v2.jsonl"
    observed_specs = ASSETS / "history-train-specs.jsonl"
    for path, expected in ((broad, "024d21fba1d2fe73dc9f8d795552303acaeb099675d9d78df179c082c7e7afcb"),
                           (observed_specs, "beb6443c43aac3b47acec4a39ce31fcebe53d9aa0de47efbb471a17b899f89fc")):
        checksum = sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                checksum.update(chunk)
        if checksum.hexdigest() != expected:
            raise ValueError(f"Frozen TRAIN source changed: {path.name}")
    sources = {}
    for text in broad.open():
        row = json.loads(text)
        if row["case_id"].startswith("train-intent-") and len(row["calls"]) == 1:
            sources[row["case_id"]] = {key: row[key] for key in ("case_id", "question", "calls")}
    rows = list(sources.values())
    rng = random.Random(64381)
    choices = scientific_pairs(rows, rng)
    rng.shuffle(rows)
    # Retain the reviewed assignments when replacing overlapping legacy HRF direction targets.
    replacements = {key: sources[clean] for key, clean in (("train-intent-0181", "train-intent-0195"),
                                                            ("train-intent-0182", "train-intent-0196"))}
    def reviewed(row):
        return replacements.get(row["case_id"], row)
    rows = [reviewed(row) for row in rows]
    choices["flag"] = [reviewed(row) for row in choices["flag"]]
    for name in ("subject", "replace", "delta", "switch", "value"):
        choices[name] = [(reviewed(left), reviewed(right)) for left, right in choices[name]]
    roots, arms = [], {name: [] for name in ARMS}
    def add(pair_id, group, side, prefix, plans, observed=None):
        root_id = f"st-tr-{pair_id}-{side}"
        root = {"root_id": root_id, "lineage": f"st-tr-{pair_id}", "group": group, "prefix": prefix}
        if observed:
            root["observed_prefix_id"] = observed["prefix_id"]
            root["source_session_id"] = observed["session_id"]
        roots.append(root)
        for arm in ARMS:
            for form in (0, 1):
                label = (side if arm == "ordinary" or form == 0 else 1 - side)
                policy = dict(plans[label])
                policy["question"] = policy.pop("questions")[form]
                arms[arm].append({"case_id": f"{root_id}-{arm}-{form}", "root_id": root_id, "slot": form,
                                  "meaning": label, "continuation": policy})
    for index, mode in enumerate(MODES):
        for i in range(6):
            anchor, other = rows[index * 12 + i * 2:index * 12 + i * 2 + 2]
            for side in (0, 1):
                offset = 128 + index * 24 + i * 4
                distractors = [rows[offset], rows[offset + 1], rows[offset + 2 + side]]
                prefix, plans = choose_actions(mode, i, side, anchor, other, distractors, choices)
                add(f"{mode}-{i:02d}", mode, side, prefix, plans)
    groups = defaultdict(dict)
    for text in observed_specs.open():
        spec = json.loads(text)
        step = spec["prefix"][0]
        ids = step.get("ancestor_case_ids", [])
        if len(ids) == 1 and ids[0] in sources and step["question"] == sources[ids[0]]["question"]:
            groups[ids[0]][spec["prefix_id"]] = spec
    available = [(key, list(values.values())) for key, values in groups.items() if len(values) >= 2]
    rng.shuffle(available)
    for i, (key, options) in enumerate(available[:8]):
        for side, spec in enumerate(rng.sample(options, 2)):
            prefix = [{"question": step["question"], "ancestor_case_ids": step.get("ancestor_case_ids", [])}
                      for step in spec["prefix"]]
            plans = (addressed("quote_exchange", 1, "Copy the exact message you gave me at the very beginning.",
                      "Please show the complete reply from our opening exchange again, exactly as it appeared."),
                     addressed("answer_again", 1, "Take up my opening question again and work it out using the original assumptions.",
                      "Go back to my first question and give me a fresh answer with those same conditions.", calls=sources[key]["calls"]))
            add(f"observed-{i:02d}", "observed_original", side, prefix, plans, spec)
    write_lines("roots", roots)
    for arm in ARMS:
        write_lines(arm, arms[arm])
    print("Authored", len(roots), "roots and", len(arms["ordinary"]), "variable presentations per arm.")


if __name__ == "__main__":
    build()
