"""Single scientific obligations under changed conditions and competing requests."""

import re
import unicodedata
from collections import Counter, defaultdict
from copy import deepcopy
from itertools import combinations

from evaluation.schema import canonical
from .common import (
    ASSETS,
    SOURCE_SHA,
    check,
    compact,
    registry,
    render as render_prompt,
    rows,
    run_generator,
    runtime,
    session,
    sha,
)
from .week_compositions import FRESH, OPERATIONS


def normalized(text):
    text = unicodedata.normalize("NFKC", text).lower()
    return " ".join("".join(ch if ch.isalnum() else " " for ch in text).split())


def state(flag):
    return "enabled" if flag else "disabled"


def directQuestions(call):
    name, a = call["name"], call["arguments"]
    subject = a.get("subject")
    what = "the HRF setting TR" if subject == "TR" else subject
    fields = set(a)
    if fields == {"subject"}:
        if name == "explain_purpose":
            return [
                f"I'm reading a GEM-pRF configuration. What does {what} represent in this analysis?",
                f"How would you explain the job of {what} to someone seeing this parameter for the first time?",
            ]
        if name == "get_default_value":
            if subject == "version":
                return [
                    "For this GEM-pRF revision, which XML version value should I use as the default?",
                    "I need the default XML version for the pinned GEM-pRF source. What value does this revision require?",
                ]
            return [
                f"For {what}, what default, if any, can I actually cite from GEM-pRF? Distinguish an example configuration value from a fallback.",
                f"A colleague labeled a value for {what} “the default”. What do the configuration example and the runtime evidence establish?",
            ]
        if name == "explain_code_usage":
            return [
                f"I'm following {what} through GEM-pRF. Which code passages read or use it? Please point me to those passages.",
                f"Where should I look in the GEM-pRF source to verify how {what} is used?",
            ]
        if name == "get_value_guidance":
            return [
                f"I need to fill in {what}. Which selection criteria for this setting can I justify from GEM-pRF's sources?",
                f"What should someone go by when choosing a value for {what} in this package?",
            ]
        if name == "get_enable_guidance":
            return [
                f"Under what circumstances should I switch {subject} on in GEM-pRF, and when would I leave it off?",
                f"I'm deciding whether to enable {subject}. What evidence can guide that choice?",
            ]
        if name == "get_units":
            return [
                f"What unit should I attach to {what} when reporting a GEM-pRF setup? Include any source qualifications.",
                f"How does GEM-pRF specify the units of {what}?",
            ]
        if name == "get_allowed_values":
            return [
                f"Which entries does GEM-pRF support for {subject}, including any qualifications about accepted values?",
                f"I need the set of supported values for {subject}. What can I enter according to GEM-pRF?",
            ]
        if name == "explain_case_sensitivity":
            return [
                f"If capitalization changes in the value of {subject}, does GEM-pRF treat the spelling as equivalent?",
                f"For the {subject} entry, is capitalization significant when GEM-pRF reads its value?",
            ]
        if name == "explain_default_rationale":
            return [
                f"What evidence explains why the listed default for {subject} was chosen? Mention any uncertainty in the rationale.",
                f"Can the source justify the choice of the {subject} default, beyond showing which value was used?",
            ]
        if name == "explain_path_portability":
            return [f"If I take the configuration to another machine, how portable is the path for {subject}?"]
        if name == "describe_path_target":
            return [
                f"I'm checking a GEM-pRF configuration. What exactly is the path in {subject} supposed to point to?"
            ]
        if name == "explain_version_format":
            return ["How must the XML version string be written for GEM-pRF's loader to compare it correctly?"]
    if name == "explain_boolean_effect" and fields == {"subject", "value"} and isinstance(a["value"], bool):
        return [
            f"What behavior should I expect from GEM-pRF with {subject} {state(a['value'])}?",
            f"Assume the Boolean value of {subject} is {compact(a['value'])}. What effect does that have during the analysis?",
        ]
    if name == "explain_flag_literal_effect" and fields == {"subject", "value"} and isinstance(a["value"], str):
        return [
            f"The raw XML contains {compact(a['value'])} as the value of {subject}. How does GEM-pRF interpret that exact text?",
            f"How does the loader treat {subject} if I spell its XML value {compact(a['value'])} with that exact capitalization?",
        ]
    if (
        name == "explain_change_effect"
        and fields == {"subject", "direction"}
        and a["direction"] in ("decrease", "increase")
    ):
        higher = a["direction"] == "increase"
        return [
            f"What consequences does the implementation support if I make {what} {'larger' if higher else 'smaller'} in GEM-pRF? State the relevant assumptions.",
            f"Suppose {what} is {'increased' if higher else 'reduced'}. How would that change affect the analysis, within the conditions covered by the code?",
        ]
    if (
        name == "explain_conditional_applicability"
        and fields == {"condition_subject", "condition_value", "subject"}
        and isinstance(a["condition_value"], bool)
    ):
        return [
            f"With {a['condition_subject']} switched {'on' if a['condition_value'] else 'off'}, does GEM-pRF use {subject}?",
            f"I need to know if {subject} takes effect in a configuration in which {a['condition_subject']} is {state(a['condition_value'])}.",
        ]
    if name == "get_value_guidance":
        conditions = {
            "slice_time_corrected": (True, "the data have undergone slice-time correction"),
            "run_labels_present": (False, "the input filenames have no run labels"),
            "gpu_out_of_memory": (True, "GPU memory is exhausted"),
            "preprocessing": ("fMRIPrep", "the preprocessing pipeline is fMRIPrep"),
            "individualize": (True, "I want an individualized setting"),
        }
        condition = None
        if fields == {"subject", "refinement_enabled"} and isinstance(a["refinement_enabled"], bool):
            condition = f"refinement is {state(a['refinement_enabled'])}"
        for key, (value, wording) in conditions.items():
            if fields == {"subject", key} and type(a[key]) is type(value) and a[key] == value:
                condition = wording
        if condition:
            return [
                f"How should I choose {subject} when {condition}?",
                f"Assume {condition}. Which considerations apply to choosing {subject} for GEM-pRF in that situation?",
            ]
    if (
        name == "explain_item_count_limits"
        and fields == {"subject", "bound"}
        and a["bound"] in ("minimum", "maximum")
    ):
        low = a["bound"] == "minimum"
        return [
            f"Does GEM-pRF impose a {'lower' if low else 'upper'} bound on how many {subject} entries I may supply? What are the parsing caveats?",
            f"I'm checking the {'smallest' if low else 'largest'} number of {subject} entries the implementation allows. What is documented?",
        ]
    if name == "explain_resource_impact" and fields == {"subject", "runtime", "memory"}:
        quantity = (
            "runtime and memory use"
            if a["memory"] and a["runtime"]
            else "memory use"
            if a["memory"]
            else "runtime"
            if a["runtime"]
            else None
        )
        if quantity:
            return [
                f"What does the source say about the effect of {subject} on {quantity} in GEM-pRF?",
                f"For {subject}, my question is specifically about {quantity}. Which effects can we infer from this implementation?",
            ]
    if (
        name == "get_file_requirements"
        and fields == {"subject", "requirement"}
        and a["requirement"] in ("format", "contents")
    ):
        directory = subject == "stimulus.directory"
        at = f"the input files in {subject}" if directory else f"the input file named by {subject}"
        if a["requirement"] == "format":
            return [
                f"Which file formats does GEM-pRF support for {at}?",
                f"I need the format requirements for {at}. What is supported?",
            ]
        return [
            f"What contents does GEM-pRF expect in {subject if directory else at}?",
            f"What needs to be present inside {subject if directory else at} for the loader to use it?",
        ]
    if name == "explain_parameter_data_matching" and fields == {"subject", "reference"}:
        label = {
            "fmri_acquisition_repetition_time": "the fMRI acquisition repetition time",
            "stimulus_file_height": "the height of the stimulus in the file",
            "stimulus_file_width": "the width of the stimulus in the file",
            "measured_timecourse_length": "the length of the measured time course",
            "stimulus_file_frame_count": "the frame count in the stimulus file",
        }.get(a["reference"])
        if label:
            return [
                f"Is {what} required to match {label} in GEM-pRF?",
                f"For this implementation, does setting {what} differently from {label} violate a matching requirement?",
            ]
    return []


def relation(a, b):
    x, y = a["arguments"], b["arguments"]
    if a["name"] == b["name"]:
        diff = [k for k in x.keys() | y.keys() if k not in x or k not in y or canonical(x[k]) != canonical(y[k])]
        if len(diff) == 1:
            return "correct_subject" if diff[0] == "subject" else "correct_condition"
    elif x.get("subject") and x["subject"] == y.get("subject"):
        return "select_operation"
    return None


def anchors():
    records, old, labels = [], set(), defaultdict(set)
    for row in rows(ASSETS / "train-full-v2.jsonl", SOURCE_SHA):
        question = normalized(row["question"])
        old.add(question)
        if row["history_origin"] not in FRESH:
            continue
        labels[question].add(canonical(row["calls"]))
        if len(row["calls"]) != 1:
            continue
        call = row["calls"][0]
        if (
            not call["name"].startswith(OPERATIONS)
            or any(k.startswith("_") for k in call["arguments"])
            or call["name"] == "explain_change_effect"
            and (
                call["arguments"].get("subject") == "default_hrf.t"
                or re.search("endpoint.*hrf|hrf.*endpoint", row["question"], re.I)
            )
        ):
            continue
        records.append({"id": row["case_id"], "question": row["question"], "call": call})
    unique = {}
    for a in sorted(records, key=lambda r: r["id"]):
        if len(labels[normalized(a["question"])]) == 1:
            unique.setdefault(canonical(a["call"]), a)
    return list(unique.values()), old


def selectQuestion(a, b, pick, style):
    wanted, other = ("second", "first") if pick else ("first", "second")
    choices = [
        f"I wrote down two questions. The {wanted} one is still open; I've already dealt with the {other}.\n(1) {a}\n(2) {b}\nPlease answer the open question only.",
        f"Draft A: {a}\nDraft B: {b}\nI changed my mind about which question to send. Please answer draft {'B' if pick else 'A'} alone.",
        f"For orientation, here are two possible questions about GEM-pRF:\n• {a}\n• {b}\nFor my current analysis I only need the {wanted} question answered.",
        f"I'm listing both notes so you can see the distinction.\n{'Resolved' if pick else 'Still to answer'}: {a}\n{'Still to answer' if pick else 'Resolved'}: {b}\nCould you answer the outstanding one?",
        f"My colleague and I are asking different things. {'Their' if pick else 'My'} note reads “{a}”. {'My' if pick else 'Their'} note reads “{b}”. Please answer my question; the colleague's note is background.",
        f"These were the two lines in my draft:\n1. {a}\n2. {b}\nCross out line {'1' if pick else '2'}. What is the answer to the line I kept?",
    ]
    return choices[style % len(choices)]


def expandedScience(render=False):
    source, used = anchors()
    reg, api, programs = registry(), runtime(), {}
    origin = session(reg, "verify-standalone")
    for a in source:
        after = api.applyModelOutput(reg, origin, a["question"], api.encodeCalls([a["call"]]))
        exchange = after["exchanges"][0]
        check(exchange["outcome"] == "committed" and len(after["semantic"]["turns"][0]["answers"]) == 1, a["id"])
        programs[a["id"]] = (exchange["assistant"], after["semantic"])

    def entry(question, correct, ancestors, group, family):
        key = "wx-sci-" + sha(canonical([family, question]))[:24]
        check(normalized(question) not in used, f"Repeated scientific question: {key}")
        used.add(normalized(question))
        before = session(reg, family)
        calls = [correct["call"]]
        target = api.encodeCalls(calls)
        after = api.applyModelOutput(reg, before, question, target)
        original_text, original_state = programs[correct["id"]]
        exchange = after["exchanges"][0]
        reference = deepcopy(original_state)
        reference["turns"][0]["question"] = question
        check(
            exchange["outcome"] == "committed"
            and exchange["assistant"] == original_text
            and after["semantic"] == reference,
            key,
        )
        row = {
            "case_id": key,
            "family_id": family,
            "group": group,
            "split": "train",
            "history_origin": "fresh",
            "question": question,
            "calls": calls,
            "target": target,
            "ancestor_case_ids": ancestors,
        }
        if render:
            row["model_input"] = render_prompt(reg, before, question)
        return row

    for a in source:
        family = "wx-direct-" + sha(canonical(a["call"]))[:22]
        for question in directQuestions(a["call"]):
            if normalized(question) not in used:
                yield entry(question, a, [a["id"]], "scientific_direct", family)
    groups = {k: [] for k in ("correct_condition", "select_operation", "correct_subject")}
    for a, b in combinations(source, 2):
        group = relation(a["call"], b["call"])
        if (
            group
            and programs[a["id"]][0] != programs[b["id"]][0]
            and normalized(a["question"]) != normalized(b["question"])
        ):
            rank = sha(canonical([a["id"], b["id"]]))
            groups[group].append((rank, a, b))
    for group, pairs in groups.items():
        exposures = Counter()
        for rank, a, b in sorted(pairs, key=lambda p: p[0]):
            if any(exposures[x["id"]] >= 3 for x in (a, b)):
                continue
            family = "wx-scope-" + rank[:22]
            style = int(rank[:8], 16) % 6
            ids = [a["id"], b["id"]]
            for pick, selected in enumerate((a, b)):
                yield entry(selectQuestion(a["question"], b["question"], pick, style), selected, ids, group, family)
            exposures.update(ids)


if __name__ == "__main__":
    run_generator(expandedScience)
