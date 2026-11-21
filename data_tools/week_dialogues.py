"""Construct dialogue supervision from scientific anchors and executed replies."""

import argparse
from collections import defaultdict
from dataclasses import dataclass

from evaluation.history_train import continuation, ordinal
from evaluation.schema import canonical
from .common import (
    ASSETS,
    SOURCE_SHA,
    check,
    compact,
    registry,
    render as render_prompt,
    rows,
    runtime,
    send,
    session,
    sha,
)


def subject(row):
    return row["calls"][0]["arguments"]["subject"]


def name(row):
    return row["calls"][0]["name"]


def gold(row):
    return {
        "kind": "gold",
        "question": row["question"],
        "calls": row["calls"],
        "ancestor_case_ids": [row["case_id"]],
    }


def negative(row):
    return {
        "kind": "provider_error",
        "question": row["question"],
        "error_code": "inference_failed",
        "ancestor_case_ids": [row["case_id"]],
    }


def choose(items, index):
    return items[index % len(items)]


def rotation(items, seed):
    return sorted(items, key=lambda r: sha(seed + r["case_id"]))


def scientificAnchors():
    out = []
    for row in rows(ASSETS / "train-full-v2.jsonl", SOURCE_SHA):
        if (
            not row["case_id"].startswith("train-intent-")
            or len(row["calls"]) != 1
            or not isinstance(row["calls"][0]["arguments"].get("subject"), str)
        ):
            continue
        q = row["question"].lower()
        if name(row) == "explain_change_effect" and (
            subject(row) == "default_hrf.t" or "endpoint" in q and "hrf" in q
        ):
            continue
        out.append({k: row[k] for k in ("case_id", "question", "calls")})
    return out


def scientific_order(anchors, label):
    queues = defaultdict(list)
    for a in rotation(anchors, label):
        queues[name(a)].append(a)
    return [q[i] for i in range(max(map(len, queues.values()))) for q in queues.values() if i < len(q)]


def question_for(kind, position, variant):
    order = ordinal(position)
    questions = {
        "full": [
            f"What exactly did you write back to request number {position}? Please include the entire message.",
            f"I need your complete reply to my {order} question for my notes. Copy it as displayed.",
            f"Show the text that appeared under my {order} request, from beginning to end.",
            f"Could you read the full response to question {position} back to me verbatim?",
        ],
        "fresh": [
            f"Please work out question {position} again with the same details I supplied then.",
            f"Take another run at my {order} request using its original assumptions.",
            f"Can you answer the question in my {order} message afresh?",
            f"I'd like a fresh answer to request {position}. The conditions are as before.",
        ],
        "answer": [
            f"What was the actual scientific answer in your reply to my {order} question? Quote it.",
            f"Retrieve the scientific answer you gave for request number {position}, exactly as recorded.",
            f"Read back the scientific answer from your response to question {position}.",
            f"I need the scientific answer delivered for my {order} request, in its original words.",
        ],
    }
    return choose(questions[kind], variant)


@dataclass
class Scene:
    config: dict
    before: dict
    prefix: tuple = ()

    def append(self, step):
        api, reg = runtime(), self.config["registry"]
        if step["kind"] == "provider_error":
            after = api.recordProviderFailure(reg, self.before, step["question"], step["error_code"])
        else:
            raw = step["raw_output"] if step["kind"] == "malformed" else api.encodeCalls(step["calls"])
            after = api.applyModelOutput(reg, self.before, step["question"], raw)
        expected = {"gold": "committed", "malformed": "error", "provider_error": "provider_error"}[step["kind"]]
        check(after["exchanges"][-1]["outcome"] == expected, step["question"])
        return Scene(self.config, after, (*self.prefix, step))

    def example(self, group, kind, question, *, sources=(), **fields):
        result = continuation(self.before, {"kind": kind, "question": question, **fields})
        question, calls, branch = result["question"], result["calls"], result["branch"]
        step = {"kind": "gold", "question": question, "calls": calls, "ancestor_case_ids": list(sources)}
        after = self.append(step)
        api = runtime()
        target = api.encodeCalls(calls)
        check(api.decodeCalls(target) == calls and after.before["exchanges"][:-1] == self.before["exchanges"])
        if kind == "quote_exchange":
            index = fields["exchange_index"]
            check(calls == [{"name": "repeat_exchange", "arguments": {"target": f"e{index}"}}])
            check(after.before["exchanges"][-1]["assistant"] == self.before["exchanges"][index - 1]["assistant"])
        cfg = self.config
        history = dict(api.sessionHistory(self.before))
        check(history.pop("session_id") == cfg["family"])
        key = "week-case-" + sha(canonical([history, question]))[:20]
        check(key not in cfg["seen"], "Duplicate history and question: " + key)
        cfg["seen"].add(key)
        origins = sorted({k for s in after.prefix for k in s["ancestor_case_ids"]})
        row = {
            "case_id": key,
            "split": "train",
            "family_id": cfg["family"],
            "session_id": cfg["family"],
            "group": group,
            "question": question,
            "calls": calls,
            "target": target,
            "prefix": list(self.prefix),
            "branch": branch,
            "ancestor_case_ids": origins,
            "history_origin": "authored_executed_prefix",
        }
        if cfg["render"]:
            row["model_input"] = render_prompt(cfg["registry"], self.before, question)
        cfg["emit"](row)
        return after


def compound(start, a, b, filler, i):
    intro = choose(
        [
            "Two things for my analysis:",
            "Please cover these questions in order.",
            "I have two separate issues.",
            "Can you cover both parts of this request?",
        ],
        i,
    )
    merged = {
        "kind": "gold",
        "question": " ".join([intro, a["question"], b["question"]]),
        "calls": a["calls"] + b["calls"],
        "ancestor_case_ids": [a["case_id"], b["case_id"]],
    }
    p = len(start.prefix) + 1
    scene = start.append(merged)
    if i % 3:
        scene = scene.append(gold(filler))
    scene.example("whole_message", "quote_exchange", question_for("full", p, i), exchange_index=p)
    for part in (1, 2):
        q = choose(
            [
                f"Pull out just the {ordinal(part)} scientific answer in your reply to request {p}.",
                f"For my {ordinal(p)} request, read back only the {ordinal(part)} of the two answers you gave.",
                f"From the two answers to question {p}, copy only answer number {part} as it appeared.",
            ],
            i,
        )
        scene.example("single_part", "quote_answer", q, exchange_index=p, part=part)
    scene.example(
        "recompute_success", "answer_again", question_for("fresh", p, i), exchange_index=p, calls=merged["calls"]
    )


def retry_twins(start, a, filler, i):
    p = len(start.prefix) + 1
    failure = (
        negative(a)
        if i % 2
        else {
            "kind": "malformed",
            "question": a["question"],
            "raw_output": "Unable to complete this response.",
            "ancestor_case_ids": [a["case_id"]],
        }
    )
    for failed, step in ((True, failure), (False, gold(a))):
        scene = start.append(step)
        if i % 3:
            scene = scene.append(gold(filler))
        scene.example(
            "quote_failure" if failed else "whole_message",
            "quote_exchange",
            question_for("full", p, i),
            exchange_index=p,
        )
        scene.example(
            "missing_answer" if failed else "single_part",
            "quote_answer",
            question_for("answer", p, i),
            exchange_index=p,
            part=1,
        )
        retried = scene.example(
            "retry_failure" if failed else "recompute_success",
            "answer_again",
            question_for("fresh", p, i),
            exchange_index=p,
            calls=a["calls"],
        )
        if failed:
            q = choose(
                [
                    "Quote the scientific answer you just obtained on that second attempt.",
                    "Read back the answer from the successful retry.",
                    "What scientific answer did the new attempt produce? Quote it exactly.",
                ],
                i,
            )
            retried.example("recovered_answer", "quote_answer", q, exchange_index=len(retried.prefix), part=1)


def ambiguous(start, a, b, filler, i):
    topic = subject(a)
    first = len(start.prefix) + 1
    scene = start.append(gold(a))
    if i % 2:
        scene = scene.append(gold(filler))
    scene = scene.append(gold(b))
    q = choose(
        [
            f"I wrote down a reference to one of the two answers on {topic}. I didn't mark which one. Could you pull that answer up?",
            f"I need one of those two scientific answers about {topic}. My note only gives the parameter name. Can you repeat the one I meant?",
            f"Which answer on {topic} did I want to copy? I have two candidates in this chat and haven't specified which.",
        ],
        i,
    )
    clarified = scene.example("ambiguous_reference", "clarify_reference", q, reference_kind="answer")
    for order, pos in (("earlier", first), ("later", len(scene.prefix))):
        q = choose(
            [
                f"I meant the {order} one. Give me that scientific answer in its original words.",
                f"The {order} answer about {topic}, please. Read that one back.",
                f"I can narrow it down: copy just the scientific answer from request number {pos}.",
            ],
            i,
        )
        clarified.example("clarified_reference", "quote_answer", q, exchange_index=pos, part=1)


def repeated_text(start, a, filler, i):
    first = len(start.prefix) + 1
    scene = start.append(gold(a))
    if i % 3:
        scene = scene.append(gold(filler))
    scene = scene.append(gold(a))
    last = len(scene.prefix)
    check(scene.before["exchanges"][first - 1]["assistant"] == scene.before["exchanges"][last - 1]["assistant"])
    for which, pos in (("first", first), ("second", last)):
        q = choose(
            [
                f"Quote the scientific answer from the {which} time I asked that identical question.",
                f"I asked the same question twice. Copy the answer recorded on the {which} occasion.",
                f"Take the {which} occurrence of that repeated question. Read back its scientific answer.",
            ],
            i,
        )
        scene.example("same_text_answer_address", "quote_answer", q, exchange_index=pos, part=1)
    pos = first if i % 2 else last
    scene.example("whole_message", "quote_exchange", question_for("full", pos, i), exchange_index=pos)


def revision(start, a, b, filler, i):
    first = len(start.prefix) + 1
    scene = start.append(gold(a))
    if i % 3:
        scene = scene.append(gold(filler))
    q = choose(
        [
            f"In the scientific answer to question {first}, change the parameter from {subject(a)} to {subject(b)}. Keep every other detail of the request.",
            f"Please correct the answer to my {ordinal(first)} question: the parameter should be {subject(b)} rather than {subject(a)}. Everything else still applies.",
            f"That {ordinal(first)} request used the wrong parameter. Revise its answer for {subject(b)}, keeping the same operation and conditions.",
        ],
        i,
    )
    revised = scene.example(
        "revise_answer",
        "revise_answer",
        q,
        exchange_index=first,
        part=1,
        expected_original=a["calls"][0],
        replacement=b["calls"][0],
        changes={"subject": subject(b)},
        sources=[b["case_id"]],
    )
    check(revised.before["semantic"]["turns"][-1]["answers"][0]["call"]["arguments"]["subject"] == subject(b))
    for text, pos in (
        ("original answer from before my correction", first),
        ("corrected answer from after my change", len(revised.prefix)),
    ):
        q = choose(
            [
                f"Please quote the {text}.",
                f"For the record I need the {text}, in your exact words.",
                f"Which wording did you use in the {text}? Copy the scientific answer.",
            ],
            i,
        )
        revised.example("original_or_corrected", "quote_answer", q, exchange_index=pos, part=1)


def pending(start, anchor, filler, i):
    fn, topic = "explain_boolean_effect", subject(anchor)
    q = choose(
        [
            f"I want the effect of my particular {topic} setting. I haven't given you its value; ask me for it first.",
            f"Can you explain what my setting for {topic} does? First check with me whether it's on or off.",
        ],
        i,
    )
    call = {"name": fn, "arguments": {"subject": topic, "_clarify": True, "_missing": ["value"]}}
    scene = start.append({"kind": "gold", "question": q, "calls": [call], "ancestor_case_ids": [anchor["case_id"]]})
    pos = len(scene.prefix)
    if i % 2:
        scene = scene.append(gold(filler))
    item_id = scene.before["exchanges"][pos - 1]["semantic_turn_id"] + ".p1"
    item = next(p for p in scene.before["semantic"]["pending"] if p["id"] == item_id)
    check(item["function"] == fn and item["known"] == {"subject": topic})
    fields = {"exchange_index": pos, "part": 1, "function": fn, "known": item["known"], "missing": item["missing"]}
    value = anchor["calls"][0]["arguments"]["value"]
    resolved = scene.example(
        "resolve_pending",
        "resolve_pending",
        f"For the open question about {topic}, my setting is {compact(value)}. Please finish that answer.",
        **fields,
        values={"value": value},
    )
    cancelled = scene.example(
        "cancel_pending",
        "cancel_pending",
        f"Drop the open question about the effect of {topic}. I no longer need that answer.",
        **fields,
    )
    for s in (resolved, cancelled):
        check(all(p["id"] != item_id for p in s.before["semantic"]["pending"]))
    scene.example("whole_message", "quote_exchange", question_for("full", pos, i), exchange_index=pos)


def buildWeekDialogues(options):
    anchors = scientificAnchors()
    reg = registry()
    by_topic = defaultdict(list)
    for a in anchors:
        by_topic[subject(a)].append(a)

    def compatible(a, b):
        return (
            name(a) == name(b)
            and subject(a) != subject(b)
            and canonical({**a["calls"][0]["arguments"], "subject": None})
            == canonical({**b["calls"][0]["arguments"], "subject": None})
        )

    pools = {
        "revision": [a for a in anchors if any(compatible(a, b) for b in anchors)],
        "ambiguous": [a for a in anchors if any(name(b) != name(a) for b in by_topic[subject(a)])],
        "pending": [
            a
            for a in anchors
            if name(a) == "explain_boolean_effect" and isinstance(a["calls"][0]["arguments"].get("value"), bool)
        ],
    }
    seen = set()
    for mode in ("compound", "retry", "ambiguous", "same_text", "revision", "pending"):
        ordered = scientific_order(pools.get(mode, anchors), "week-train-v1/" + mode)
        for i in range(64):
            family = "wk-" + sha(f"week-train-v1/{mode}/{i}")[:20]
            a = choose(ordered, i)
            candidates = rotation(anchors, family)
            if mode == "revision":
                b = next(b for b in candidates if compatible(a, b))
            elif mode == "ambiguous":
                b = next(b for b in candidates if subject(b) == subject(a) and name(b) != name(a))
            else:
                b = next(b for b in candidates if subject(b) != subject(a))
            fillers = {}
            for row in candidates:
                if subject(row) not in (subject(a), subject(b)):
                    fillers.setdefault(subject(row), row)
            unique = list(fillers.values())
            check(len(unique) >= 3)
            scene = Scene(
                {
                    "family": family,
                    "render": options.get("render", False),
                    "emit": options["emit"],
                    "registry": reg,
                    "seen": seen,
                },
                session(reg, family),
            )
            for j in range(i % 3):
                scene = scene.append(negative(unique[j]) if i % 4 == 1 and j == 0 else gold(unique[j]))
            if mode == "pending" and not scene.prefix:
                scene = scene.append(gold(unique[0]))
            if mode in ("compound", "ambiguous", "revision"):
                {"compound": compound, "ambiguous": ambiguous, "revision": revision}[mode](
                    scene, a, b, unique[2], i
                )
            else:
                {"retry": retry_twins, "same_text": repeated_text, "pending": pending}[mode](scene, a, unique[2], i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    buildWeekDialogues({"render": parser.parse_args().render, "emit": send})
