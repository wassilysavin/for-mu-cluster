"""Generate executable dialogues that change, retract, or retry a mixed-scope request."""

import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from evaluation.dialogue import declineRequestMessage
from evaluation.schema import canonical
from .common import ASSETS, SOURCE_SHA, check, registry, render as render_prompt, rows, runtime, send, session, sha
from .week_compositions import FRESH, usableAnchor

DECLINE = {"name": "decline_request", "arguments": {}}
INTROS = (
    "Could you deal with both of these requests in order?",
    "I have two separate questions. Please address each in the order listed.",
    "Please go through these two items, keeping their answers separate.",
    "I would like a reply to each of these, starting with the first.",
)


def call(name, **args):
    return {"name": name, "arguments": args}


def part(kind, text):
    return {"kind": kind, "text": text}


def mix(science, outside, outside_first):
    return [outside, science] if outside_first else [science, outside]


def gold(row):
    return {
        "kind": "gold",
        "question": row["question"],
        "calls": row["calls"],
        "ancestor_case_ids": [row["case_id"]],
    }


def name(row):
    return row["calls"][0]["name"]


def subject(row):
    return row["calls"][0]["arguments"]["subject"]


def anchors():
    outside, candidates, labels = [], [], defaultdict(set)
    for row in rows(ASSETS / "train-full-v2.jsonl", SOURCE_SHA):
        if row["history_origin"] not in FRESH:
            continue
        item = {k: row[k] for k in ("case_id", "calls", "question", "history_origin")}
        labels[row["question"].lower().strip()].add(canonical(row["calls"]))
        if re.fullmatch(r"train-outside-\d+", row["case_id"]):
            check(row["calls"] == [DECLINE])
            outside.append(item)
        if usableAnchor(row):
            candidates.append(item)
    unique = {}
    for a in candidates:
        if len(labels[a["question"].lower().strip()]) == 1:
            unique.setdefault(canonical(a["calls"]), a)
    queues = defaultdict(list)
    for a in sorted(unique.values(), key=lambda a: sha(a["case_id"])):
        queues[name(a)].append(a)
    science = [q[i] for i in range(max(map(len, queues.values()))) for q in queues.values() if i < len(q)]
    check(len(outside) == 200)
    return science, outside


@dataclass
class Scene:
    ctx: dict
    state: dict
    prefix: tuple = ()

    def committed(self, step):
        api = runtime()
        after = api.applyModelOutput(
            self.ctx["registry"], self.state, step["question"], api.encodeCalls(step["calls"])
        )
        ex = after["exchanges"][-1]
        check(ex["outcome"] == "committed" and ex["model_calls"] == step["calls"], step["question"])
        check(
            after["exchanges"][:-1] == self.state["exchanges"]
            and after["semantic"]["turns"][:-1] == self.state["semantic"]["turns"]
        )
        return Scene(self.ctx, after, (*self.prefix, step))

    def failure(self, question, ancestors, malformed):
        reg, api = self.ctx["registry"], runtime()
        if malformed:
            trace = {"kind": "malformed", "raw_output": "Unable to return the requested response."}
            after = api.applyModelOutput(reg, self.state, question, trace["raw_output"])
        else:
            trace = {"kind": "provider_error", "error_code": "inference_failed"}
            after = api.recordProviderFailure(reg, self.state, question, trace["error_code"])
        check(after["exchanges"][-1]["outcome"] == ("error" if malformed else "provider_error"))
        check(after["semantic"] == self.state["semantic"])
        return Scene(
            self.ctx, after, (*self.prefix, {**trace, "question": question, "ancestor_case_ids": ancestors})
        )

    def example(
        self,
        group,
        question,
        calls,
        ancestors,
        parts,
        answers=(),
        links=None,
        retry=None,
        opening=None,
        closed=None,
    ):
        after = self.committed(
            {"kind": "gold", "question": question, "calls": calls, "ancestor_case_ids": ancestors}
        )
        semantic = after.state["semantic"]
        turn = semantic["turns"][-1]
        check([part(p["kind"], p["text"]) for p in turn["response_parts"]] == parts)
        check(after.state["exchanges"][-1]["assistant"] == "\n\n".join(p["text"] for p in parts))
        check([a["call"] for a in turn["answers"]] == list(answers))
        old = self.state["semantic"]["pending"]
        if opening:
            fn, known, field = opening
            check(len(semantic["pending"]) == len(old) + 1)
            item = semantic["pending"][-1]
            check(next(p for p in turn["response_parts"] if p["kind"] == "clarification")["id"] == item["id"])
            check(item["function"] == fn and item["known"] == known and item["missing"] == [field])
        elif closed:
            check(any(p["id"] == closed for p in old))
            check(semantic["pending"] == [p for p in old if p["id"] != closed])
        else:
            check(semantic["pending"] == old)
        check(
            [{k: a[k] for k in ("repeats", "revises", "resolves") if a.get(k)} for a in turn["answers"]]
            == (links if links is not None else [{} for _ in answers])
        )
        check(
            [e["target"] for e in turn["effects"] if e["kind"] == "exchange_retried"] == ([retry] if retry else [])
        )
        api = runtime()
        history = {k: v for k, v in api.sessionHistory(self.state).items() if k != "session_id"}
        key = "xs-scope-" + sha(canonical([history, question]))[:20]
        check(key not in self.ctx["seen"], f"Duplicate: {key}")
        self.ctx["seen"].add(key)
        target = api.encodeCalls(calls)
        check(api.decodeCalls(target) == calls)
        origins = sorted({k for s in after.prefix for k in s["ancestor_case_ids"]})
        check(origins)
        row = {
            "case_id": key,
            "family_id": self.ctx["family"],
            "split": "train",
            "group": group,
            "question": question,
            "calls": calls,
            "target": target,
            "prefix": list(self.prefix),
            "ancestor_case_ids": origins,
            "history_origin": "authored_executed_prefix" if self.prefix else "fresh",
        }
        if self.ctx["render"]:
            row["model_input"] = render_prompt(self.ctx["registry"], self.state, question)
        self.ctx["emit"](row)
        self.ctx["counts"][group] += 1
        return after


def different(science, idx, exclude):
    for i in range(len(science)):
        row = science[(idx + i) % len(science)]
        if all(subject(a) != subject(row) and name(a) != name(row) for a in exclude):
            return row
    raise ValueError("Missing independent anchor")


def obligations(start, a, b, outside, filler, i):
    if i % 3 == 2:
        start = start.committed(gold(filler))
    outside_first = i // 2 % 2 == 0
    pos, other = ("second", "first") if outside_first else ("first", "second")
    q = INTROS[i // 4 % len(INTROS)] + "\n" + "\n".join(mix(a["question"], outside["question"], outside_first))
    both = mix(a["calls"][0], DECLINE, outside_first)
    origins = [a["case_id"], outside["case_id"]]
    pa, pb = map(start.ctx["standalone"], (a, b))
    declined = part("declined_request", declineRequestMessage)
    parts = mix(pa, declined, outside_first)
    done = start.example("scope_mixed_parts", q, both, origins, parts, a["calls"])
    exchange = done.state["exchanges"][-1]
    answer = done.state["semantic"]["turns"][-1]["answers"][0]["id"]
    failed = start.failure(q, origins, i // 8 % 2 == 1)
    retry = call("retry_exchange", target=exchange["id"])
    quote = (
        "Copy that entire two-part reply for me, including the part you declined."
        if i % 2
        else "What did your complete reply to those two items say, word for word?"
    )
    done.example(
        "scope_quote_whole",
        quote,
        [call("repeat_exchange", target=exchange["id"])],
        [],
        [part("replayed_exchange", exchange["assistant"])],
    )
    done.example(
        "scope_quote_science",
        f"Quote the scientific answer from that reply, just the {pos} part.",
        [call("repeat_answer", target=answer)],
        [],
        [pa],
        a["calls"],
        [{"repeats": answer}],
    )
    narrow = f"Drop the {other} item. Please try just the {pos} item again, with the same assumptions."
    done.example("scope_retract_outside", narrow, a["calls"], [], [pa], a["calls"])
    failed.example("scope_retry_subset", narrow, [retry, *a["calls"]], [], [pa], a["calls"], retry=exchange["id"])
    again = (
        "Please answer both items afresh in their original order, keeping all the same conditions."
        if i % 2
        else "Can you take another pass at both requests as originally written? Work through them in order."
    )
    done.example("scope_fresh_full", again, both, [], parts, a["calls"])
    failed.example("scope_retry_full", again, [retry, *both], [], parts, a["calls"], retry=exchange["id"])
    if i % 2:
        done.example(
            "scope_keep_outside",
            f"I withdraw the {pos} item. The {other} request still stands: " + outside["question"],
            [DECLINE],
            [],
            [declined],
        )
    else:
        done.example(
            "scope_replace_science",
            "I want to replace the scientific answer in that two-part reply. Use this question in its place: "
            + b["question"],
            [call(name(b), **b["calls"][0]["arguments"], _replace=answer)],
            [b["case_id"]],
            [pb],
            b["calls"],
            [{"revises": answer}],
        )


FIELDS = {
    "explain_change_effect": "direction",
    "explain_boolean_effect": "value",
    "explain_conditional_applicability": "condition_value",
}


def pending_scope(start, a, outside, b, i):
    full = a["calls"][0]
    field = FIELDS[full["name"]]
    known = dict(full["arguments"])
    value = known.pop(field)
    topic = known["subject"]
    if field == "direction":
        setup = f"What would my planned change to {topic} do? First ask which direction I have in mind."
    elif field == "condition_value":
        setup = (
            f"Does {topic} apply in my setup? Ask about my "
            + known["condition_subject"]
            + " setting before deciding."
        )
    else:
        setup = (
            f"What would my setting of {topic} do? Please ask whether I mean enabled or disabled before answering."
        )
    request = call(full["name"], **known, _clarify=True, _missing=[field])
    api = runtime()
    isolated = api.applyModelOutput(start.ctx["registry"], start.state, setup, api.encodeCalls([request]))
    check(isolated["exchanges"][-1]["outcome"] == "committed")
    clar = isolated["semantic"]["turns"][0]["response_parts"][0]
    check(clar["kind"] == "clarification")
    declined = part("declined_request", declineRequestMessage)
    q = INTROS[i // 2 % len(INTROS)] + "\n" + "\n".join(mix(setup, outside["question"], i % 2))
    opened = start.example(
        "scope_mixed_pending",
        q,
        mix(request, DECLINE, i % 2),
        [a["case_id"], outside["case_id"]],
        mix(part("clarification", clar["text"]), declined, i % 2),
        opening=(full["name"], known, field),
    )
    item = opened.state["semantic"]["pending"][0]["id"]
    other = opened.example(
        "scope_keep_pending",
        "I will answer your clarification later. Meanwhile, a separate question: " + b["question"],
        b["calls"],
        [b["case_id"]],
        [start.ctx["standalone"](b)],
        b["calls"],
    )
    scene = other if i % 2 else opened
    detail = (
        "I mean a change in this direction: " + value + "."
        if field == "direction"
        else known.get("condition_subject", topic) + " is " + ("enabled" if value else "disabled") + "."
    )
    scene.example(
        "scope_resolve_only",
        f"Let’s return to the open question about {topic}. {detail} Please finish only that answer.",
        [call(full["name"], _resolve=item, **{field: value})],
        [],
        [start.ctx["standalone"](a)],
        [full],
        [{"resolves": item}],
        closed=item,
    )
    scene.example(
        "scope_cancel_then_decline",
        f"Cancel the open question about {topic}. I still want help with this other request: "
        + outside["question"],
        [call("cancel_pending", target=item), DECLINE],
        [],
        [part("cancellation", "That pending request has been cancelled."), declined],
        closed=item,
    )


def buildExtendedScope(options):
    science, outside = anchors()
    reg, api = registry(), runtime()
    empty = session(reg, "scope-anchor-verification")
    seen, counts, cached = set(), Counter(), {}

    def standalone(row):
        key = row["case_id"]
        if key not in cached:
            after = api.applyModelOutput(reg, empty, row["question"], api.encodeCalls(row["calls"]))
            check(after["exchanges"][0]["outcome"] == "committed", key)
            answer = after["semantic"]["turns"][0]["response_parts"][0]
            check(answer["kind"] == "answer")
            cached[key] = part("answer", answer["text"])
        return cached[key]

    def scene(tag, ids):
        family = "xs-scope-family-" + sha(canonical([tag, ids]))[:16]
        return Scene(
            {
                "registry": reg,
                "seen": seen,
                "counts": counts,
                "family": family,
                "standalone": standalone,
                "emit": options["emit"],
                "render": options.get("render", False),
            },
            session(reg, family),
        )

    for i, out in enumerate(outside):
        a = science[i % len(science)]
        b = different(science, i + 37, [a])
        filler = different(science, i + 17, [a, b])
        obligations(scene("obligations", [a["case_id"], out["case_id"]]), a, b, out, filler, i)
    pending = [a for a in science if name(a) in FIELDS]
    for i, a in enumerate(pending):
        out = outside[(i * 37 + 11) % len(outside)]
        b = different(science, i + 71, [a])
        pending_scope(scene("pending", [a["case_id"], out["case_id"]]), a, out, b, i)
    return {"cases": len(seen), "families": len(outside) + len(pending), "groups": dict(counts)}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    flag = parser.parse_args().render
    summary = buildExtendedScope({"render": flag, "emit": send if flag else lambda row: None})
    if not flag:
        send(summary)
