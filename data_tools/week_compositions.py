"""Compose ordered scientific obligations from authenticated standalone TRAIN anchors."""

import re
from collections import Counter, defaultdict
from itertools import combinations

from evaluation.schema import canonical
from .common import (
    ASSETS,
    SOURCE_SHA,
    check,
    registry,
    render as render_prompt,
    rows,
    run_generator,
    runtime,
    session,
    sha,
)

TERMS = (
    "I have separate questions about GEM-pRF. Please answer each one in order.",
    "Please take the following GEM-pRF questions one at a time, in the order given.",
    "In my notes these are separate questions. Could you answer each under its own stated conditions?",
    "For each of these questions, give a separate answer in the same order.",
)
FRESH = {"fresh", "new_standalone"}
OPERATIONS = ("explain_", "get_", "describe_", "interpret_", "compare_")
EXCLUDED = re.compile(
    r"\b(only|exclude|instead of|rather than|do not|don’t|don't|leave out|just the|out of scope|not part of|not requesting|not asking|not the (?:documented|default|maximum|minimum)|previous|above|earlier|again)\b",
    re.I,
)


def fingerprint(calls):
    return canonical(sorted(map(canonical, calls)))


def usableAnchor(row):
    if row["history_origin"] not in FRESH or len(row["calls"]) != 1:
        return False
    call = row["calls"][0]
    args, name, question = call["arguments"], call["name"], row["question"]
    subject = args.get("subject")
    if (
        not name.startswith(OPERATIONS)
        or not isinstance(subject, str)
        or any(k.startswith("_") for k in args)
        or EXCLUDED.search(question)
    ):
        return False
    if name in ("explain_change_effect", "change_explain_change_effect") and (
        subject == "default_hrf.t" or re.search("endpoint.*hrf|hrf.*endpoint", question, re.I)
    ):
        return False
    return bool(re.search(r"(?<![a-z0-9_])" + re.escape(subject) + r"(?![a-z0-9_])", question, re.I | re.ASCII))


def inventory():
    candidates, existing, labels = [], set(), defaultdict(set)
    for row in rows(ASSETS / "train-full-v2.jsonl", SOURCE_SHA):
        existing.add(fingerprint(row["calls"]))
        if row["history_origin"] in FRESH:
            labels[row["question"].lower().strip()].add(canonical(row["calls"]))
        if usableAnchor(row):
            call = row["calls"][0]
            candidates.append(
                {
                    "id": row["case_id"],
                    "question": row["question"],
                    "call": call,
                    "subject": call["arguments"]["subject"],
                    "options": {k: v for k, v in call["arguments"].items() if k != "subject"},
                }
            )
    unique = {}
    for anchor in sorted(candidates, key=lambda a: a["id"]):
        if len(labels[anchor["question"].lower().strip()]) == 1:
            unique.setdefault(canonical(anchor["call"]), anchor)
    return list(unique.values()), existing


def contrast(a, b):
    if a["subject"] == b["subject"] and a["call"]["name"] != b["call"]["name"]:
        return "science_same_subject"
    if a["call"]["name"] != b["call"]["name"]:
        return None
    if a["subject"] != b["subject"]:
        return "science_same_operation" if canonical(a["options"]) == canonical(b["options"]) else None
    x, y = a["options"], b["options"]
    if x.keys() == y.keys() and sum(canonical(x[k]) != canonical(y[k]) for k in x) == 1:
        return "science_argument_contrast"
    return None


def buildFamilies():
    anchors, existing = inventory()
    groups = {key: [] for key in ("science_argument_contrast", "science_same_subject", "science_same_operation")}
    for a, b in combinations(anchors, 2):
        group = contrast(a, b)
        if group and fingerprint([a["call"], b["call"]]) not in existing:
            groups[group].append((sha("week-science-v1|" + a["id"] + "|" + b["id"]), a, b))
    families, uses = [], Counter()

    def add(group, members):
        key = fingerprint([a["call"] for a in members])
        if key in existing or any(uses[a["id"]] >= 5 for a in members):
            return False
        existing.add(key)
        families.append({"family_id": "wk-sci-family-" + sha(key)[:20], "group": group, "members": members})
        uses.update(a["id"] for a in members)
        return True

    for group, proposals in groups.items():
        proposals.sort(key=lambda row: row[0])
        count = 0
        for _, a, b in proposals:
            count += add(group, [a, b])
            if count >= 112:
                break
    triples = 0
    for rank, a, b in groups["science_same_subject"]:
        others = [
            c
            for c in anchors
            if c["subject"] != a["subject"]
            and c["call"]["name"] == a["call"]["name"]
            and canonical(a["options"]) == canonical(c["options"])
        ]
        if any(
            add("science_three_obligations", [a, b, c]) for c in sorted(others, key=lambda c: sha(rank + c["id"]))
        ):
            triples += 1
            if triples >= 96:
                break
    return families


def buildCompositions(render=False):
    reg, api, singles, seen = registry(), runtime(), {}, set()
    for group in buildFamilies():
        family = group["family_id"]
        head = TERMS[int(sha(family)[:8], 16) % len(TERMS)]
        for items in (group["members"], list(reversed(group["members"]))):
            question = head + "\n" + "\n".join(f"({i}) " + a["question"] for i, a in enumerate(items, 1))
            calls = [a["call"] for a in items]
            ancestors = [a["id"] for a in items]
            key = "wk-sci-" + sha(canonical(ancestors))[:20]
            check(question not in seen, "Duplicate week composition")
            seen.add(question)
            before = session(reg, key)
            target = api.encodeCalls(calls)
            after = api.applyModelOutput(reg, before, question, target)
            exchange = after["exchanges"][-1]
            check(exchange["outcome"] == "committed" and exchange["model_calls"] == calls, key)
            expected = []
            for a in items:
                if a["id"] not in singles:
                    one = api.applyModelOutput(reg, before, a["question"], api.encodeCalls([a["call"]]))
                    check(one["exchanges"][0]["outcome"] == "committed", a["id"])
                    answers = one["semantic"]["turns"][0]["answers"]
                    check(len(answers) == 1, a["id"])
                    singles[a["id"]] = one["exchanges"][0]["assistant"], answers[0]
                expected.append(singles[a["id"]])
            check(exchange["assistant"] == "\n\n".join(e[0] for e in expected))
            answers = [{**a, "id": f"t1.c{i}"} for i, (_, a) in enumerate(expected, 1)]
            semantic = after["semantic"]
            turn = semantic["turns"][0]
            check(turn["answers"] == answers and turn["effects"] == [])
            check(
                turn["response_parts"]
                == [{"kind": "answer", "id": a["id"], "text": text} for a, (text, _) in zip(answers, expected)]
            )
            check(semantic["focus"] == list(dict.fromkeys(k for a in answers for k in a["focus"])))
            check(semantic["revision"] == 1 and semantic["pending"] == before["semantic"]["pending"])
            row = {
                "case_id": key,
                "family_id": family,
                "group": group["group"],
                "question": question,
                "calls": calls,
                "target": target,
                "history_origin": "fresh",
                "ancestor_case_ids": ancestors,
                "prefix": [],
                "split": "train",
            }
            if render:
                row["model_input"] = render_prompt(reg, before, question)
            yield row


if __name__ == "__main__":
    run_generator(buildCompositions)
