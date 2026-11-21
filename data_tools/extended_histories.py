"""Ground new continuations in their authenticated TRAIN dialogue and frozen answers."""

import argparse
from collections import defaultdict
from itertools import permutations

from evaluation.history_train import ordinal
from evaluation.registry import createRegistry
from evaluation.schema import canonical
from .common import REGISTRY_SHA, SOURCE_SHA, check, read_json, render as render_prompt, runtime, send, sha
from .transfer_contract import transferRegistry
from .week_replay_check import CONTEXT_SHA, index


def digest(value):
    return sha(canonical(value))


def choose(items, key):
    return items[int(sha(key)[:8], 16) % len(items)]


def call(name, **args):
    return {"name": name, "arguments": args}


def repeat(kind, target):
    return call("repeat_" + kind, target=target)


def suspect(question, calls):
    q = question.lower()
    return any(
        c["name"] in ("explain_change_effect", "change_explain_change_effect")
        and ("default_hrf.t" in canonical(c["arguments"]) or "endpoint" in q and "hrf" in q)
        for c in calls
    )


def ground_sources(rows, contexts, registry):
    standalone, by_subject, by_operation = {}, defaultdict(dict), defaultdict(dict)
    for row in rows.values():
        q, calls = row["question"], row["calls"]
        if row["case_id"] in contexts or suspect(q, calls):
            continue
        if not calls or not all(
            registry.tool(c["name"]) and not any(k.startswith("_") for k in c["arguments"]) for c in calls
        ):
            continue
        if q not in standalone:
            standalone[q] = row
        elif not standalone[q] or canonical(standalone[q]["calls"]) != canonical(calls):
            standalone[q] = None
        if len(calls) != 1:
            continue
        c = calls[0]
        subject = c["arguments"].get("subject")
        if not isinstance(subject, str):
            continue
        name = c["name"]
        by_operation[name][canonical(c["arguments"])] = row
        if name in ("get_units", "get_default_value", "explain_purpose") and len(c["arguments"]) == 1:
            by_subject[subject][name] = row
    return standalone, by_subject, by_operation


def address(exchanges, index, key):
    question = exchanges[index]["question"]
    span = " ".join(question.split()[:7])
    choices = [f"my {ordinal(index + 1)} request", f"question {index + 1}"]
    back = len(exchanges) - index - 1
    if not back:
        choices.append("my last question")
    if back == 1:
        choices.append("the question just before my last one")
    if not index:
        choices.append("my opening message")
    if len(span) >= 12 and span in question and sum(span in ex["question"] for ex in exchanges) == 1:
        choices.extend([f"the request where I wrote “{span}”", f"my question containing “{span}”"])
    return choose(choices, key)


def term(slot):
    return slot["answer"]["call"]["arguments"].get("subject")


def answer_ref(slot, entries, key):
    section = address(entries, slot["index"], key)
    if len(slot["turn"]["answers"]) == 1:
        return f"the scientific answer you gave to {section}"
    return f"the {ordinal(slot['part'] + 1)} scientific answer in your reply to {section}"


def parameter_ref(slot, answer):
    c = slot["answer"]["call"]
    args = c["arguments"]
    if args.get("condition_subject"):
        return f"{args['subject']} from {answer} (the condition there involved {args['condition_subject']})"
    if c["name"] == "explain_parameter_data_matching":
        return f"the configured {args['subject']} discussed in {answer}"
    return f"the main parameter you addressed in {answer}"


def science_prompt(op, r, key):
    choices = {
        "get_units": [
            f"What units does GEM-pRF document for {r}?",
            f"Which units go with {r}?",
            f"I'm checking the units now. What is the unit convention for {r}?",
        ],
        "get_default_value": [
            f"What default is documented for {r}?",
            f"For {r}, can I see the documented default?",
            f"I also need the documented default value for {r}. What is it?",
        ],
        "explain_purpose": [
            f"What is {r} used for in GEM-pRF?",
            f"Can you explain the role of {r} here?",
            f"What is the purpose of {r} in this analysis?",
        ],
    }
    return choose(choices[op], key)


def cases_for(state, source, key):
    standalone, by_subject, by_operation = source
    entries = state["exchanges"]
    turns = {turn["id"]: turn for turn in state["semantic"]["turns"]}
    slots = []
    for index_, e in enumerate(entries):
        turn = turns.get(e.get("semantic_turn_id"))
        if not turn:
            continue
        for part, answer in enumerate(turn["answers"]):
            text = next(
                (r["text"] for r in turn["response_parts"] if r["kind"] == "answer" and r["id"] == answer["id"]),
                None,
            )
            if isinstance(text, str) and text in e["assistant"]:
                slots.append({"index": index_, "part": part, "answer": answer, "text": text, "turn": turn})

    def pick(xs, label):
        return choose(xs, key + "/" + label)

    pos = pick(range(len(entries)), "full")
    e = entries[pos]
    ref = address(entries, pos, key + "whole")
    yield (
        "stored_whole_reply",
        pick(
            [
                f"I need the full wording of your response to {ref}. Copy it for my notes.",
                f"Could you paste everything you sent in reply to {ref}?",
                f"Please read your reply to {ref} back verbatim.",
                f"For the record, what was your complete message after {ref}?",
            ],
            "q-full",
        ),
        [repeat("exchange", e["id"])],
        {"texts": [e["assistant"]]},
        [],
    )
    if slots:
        slot = pick(slots, "single")
        ref = answer_ref(slot, entries, key + "single")
        yield (
            "stored_answer_part",
            pick(
                [
                    f"Pull out {ref}, word for word.",
                    f"For my notes, could you reproduce {ref} exactly?",
                    f"Would you copy just {ref} here?",
                ],
                "q-single",
            ),
            [repeat("answer", slot["answer"]["id"])],
            {"texts": [slot["text"]]},
            [],
        )
    if len(entries) >= 2:
        i, j = pick(list(permutations(range(len(entries)), 2)), "pair")
        a, b = address(entries, i, key + "pair-a"), address(entries, j, key + "pair-b")
        yield (
            "ordered_stored_messages",
            pick(
                [
                    f"I need two complete messages for comparison. Copy your reply to {a}, followed by your reply to {b}.",
                    f"Read back both replies verbatim in this order: the one you sent after {a}, then the one after {b}.",
                    f"For the transcript, put the exact response to {a} first. After it, paste the complete response to {b}.",
                ],
                "q-pair",
            ),
            [repeat("exchange", entries[i]["id"]), repeat("exchange", entries[j]["id"])],
            {"texts": [entries[i]["assistant"], entries[j]["assistant"]]},
            [],
        )

    scientific = [s for s in slots if term(s) in by_subject]
    if scientific:
        slot = pick(scientific, "new-topic")
        r = answer_ref(slot, entries, key + "new")
        targets = [
            t for t in by_subject[term(slot)].values() if t["calls"][0]["name"] != slot["answer"]["call"]["name"]
        ]
        if targets:
            target = pick(targets, "new-op")
            subject_ref = parameter_ref(slot, r)
            op = target["calls"][0]["name"]
            yield (
                "new_operation_on_referenced_subject",
                science_prompt(op, subject_ref, key),
                target["calls"],
                {"scientific": target["calls"]},
                [target["case_id"]],
            )
            if len(entries) >= 2:
                idx = pick([i for i in range(len(entries)) if i != slot["index"]], "other")
                other = entries[idx]
                ref = address(entries, idx, key + "other")
                question = science_prompt(op, subject_ref, key + "two")
                first = pick([True, False], "order")
                quoted = repeat("exchange", other["id"])
                if first:
                    question += f" I also need your complete reply to {ref} copied exactly."
                else:
                    question = f"Start by quoting your entire reply to {ref}. Then: " + question
                yield (
                    "mixed_lookup_and_quote",
                    question,
                    [*target["calls"], quoted] if first else [quoted, *target["calls"]],
                    {"scientific": target["calls"], "quotePart": [1 if first else 0, other["assistant"]]},
                    [target["case_id"]],
                )

        revision = []
        for slot in scientific:
            prior = slot["answer"]["call"]
            choices = []
            for row in by_operation.get(prior["name"], {}).values():
                alt = row["calls"][0]["arguments"]
                if isinstance(alt.get("subject"), str) and alt["subject"] != prior["arguments"]["subject"]:
                    if canonical({**alt, "subject": prior["arguments"]["subject"]}) == canonical(
                        prior["arguments"]
                    ):
                        choices.append(row)
            if choices:
                revision.append((slot, choices))
        if revision:
            slot, choices = pick(revision, "rev-slot")
            exemplar = pick(choices, "rev-alt")
            target = exemplar["calls"][0]
            r, frm, to = answer_ref(slot, entries, key + "rev"), term(slot), target["arguments"]["subject"]
            yield (
                "change_referenced_subject",
                pick(
                    [
                        f"In {r}, the parameter was {frm}. I meant {to}. Revise that answer and keep the rest of the request the same.",
                        f"Please update {r} for {to} in place of {frm}. Keep the operation and all other arguments as they were.",
                        f"One correction to {r}: use the parameter {to} instead of {frm}. Everything else in that answer still applies.",
                    ],
                    "q-rev",
                ),
                [call("change_" + target["name"], target=slot["answer"]["id"], changes={"subject": to})],
                {"scientific": [target], "revision": slot["answer"]["id"]},
                [exemplar["case_id"]],
            )
    fresh = [(ex, i, standalone[ex["question"]]) for i, ex in enumerate(entries) if standalone.get(ex["question"])]
    if fresh:
        failures = [f for f in fresh if f[0]["outcome"] != "committed"]
        ex, i, row = pick(failures or fresh, "rerun")
        ref = address(entries, i, key + "rerun")
        yield (
            "retry_stored_failure" if failures else "recompute_stored_request",
            pick(
                [
                    f"Could you work through {ref} again from scratch using the details I originally supplied?",
                    f"Please answer {ref} anew. Use the same assumptions I gave you then.",
                    f"Take another run at {ref}; I want a newly worked answer under the original conditions.",
                ],
                "q-rerun",
            ),
            ([call("retry_exchange", target=ex["id"])] if failures else []) + row["calls"],
            {"scientific": row["calls"], **({"retry": ex["id"]} if failures else {})},
            [row["case_id"]],
        )
    failed = [(ex, i) for i, ex in enumerate(entries) if ex["outcome"] != "committed"]
    clarification = [call("clarify_reference", kind="answer")]
    if failed:
        ex, i = pick(failed, "absent")
        ref = address(entries, i, key + "absent")
        check("semantic_turn_id" not in ex)
        yield (
            "requested_absent_scientific_answer",
            f"I want the scientific answer you actually gave to {ref}. Can you quote that specific answer?",
            clarification,
            {"clarification": True},
            [],
        )
        if ex["id"] != e["id"]:
            yield (
                "copy_displayed_failure",
                f"What was the complete message shown when {ref} failed? Please paste its text exactly.",
                [repeat("exchange", ex["id"])],
                {"texts": [ex["assistant"]]},
                [],
            )
    topics = defaultdict(set)
    for s in scientific:
        topics[term(s)].add(s["text"])
    ambiguous = [topic for topic, texts in topics.items() if len(texts) >= 2]
    if ambiguous:
        topic = pick(ambiguous, "ambiguous")
        yield (
            "clarify_multiple_real_answers",
            pick(
                [
                    f"I need one particular earlier scientific answer about {topic}. I have several in these notes and haven't specified which one to quote.",
                    f"Could you retrieve the particular {topic} answer I have in mind? I haven't said which of the different earlier answers I'm referring to.",
                    f"One of the answers about {topic} is the one I want verbatim. Do you have enough information to tell which one I mean?",
                ],
                "q-ambig",
            ),
            clarification,
            {"clarification": True},
            [],
        )


def check_continuation(registry, state, question, calls, expected):
    api = runtime()
    target = api.encodeCalls(calls)
    check(api.decodeCalls(target) == calls)
    after = api.applyModelOutput(registry, state, question, target)
    check(after["exchanges"][-1]["outcome"] == "committed", question)
    check(after["exchanges"][:-1] == state["exchanges"])
    check(after["semantic"]["turns"][:-1] == state["semantic"]["turns"])
    check(after["semantic"]["pending"] == state["semantic"]["pending"])
    turn = after["semantic"]["turns"][-1]
    parts = turn["response_parts"]
    if "texts" in expected:
        check([p["text"] for p in parts] == expected["texts"])
    if "scientific" in expected:
        check([a["call"] for a in turn["answers"]] == expected["scientific"])
    if "quotePart" in expected:
        idx, text = expected["quotePart"]
        check(parts[idx]["text"] == text)
    if expected.get("clarification"):
        check(parts[0]["kind"] == "reference_clarification")
    if expected.get("revision"):
        check(turn["answers"][0]["revises"] == expected["revision"])
    if expected.get("retry"):
        check(any(e["kind"] == "exchange_retried" and e["target"] == expected["retry"] for e in turn["effects"]))
    return target


def buildExtendedHistories(options):
    spec = read_json("diagnostic-train-contexts.json", CONTEXT_SHA)
    check(spec["split"] == "train" and spec["source_dataset"]["sha256"] == SOURCE_SHA)
    bundle = read_json("registry.json", REGISTRY_SHA)
    source_rows = index(
        "train-full-v2.jsonl",
        SOURCE_SHA,
        lambda r: {k: r[k] for k in ("case_id", "question", "calls", "history_origin")},
    )
    contexts = index("diagnostic-train-contexts.jsonl", spec["contexts"]["sha256"], lambda r: r)
    original = createRegistry(bundle["knowledge"], bundle["contracts"])
    revised = transferRegistry(bundle)
    check(original.fingerprint() == spec["registry_sha256"])
    revised_id = revised.fingerprint()
    source = ground_sources(source_rows, contexts, original)
    states, seen = {}, set()
    api = runtime()
    for cid, record in contexts.items():
        row, before = source_rows[cid], record["before"]
        check(
            digest(before) == record["before_sha256"]
            and row["history_origin"] == record["provenance"]["history_origin"]
        )
        if suspect(row["question"], row["calls"]) or any(
            suspect(e["question"], e.get("model_calls") or []) for e in before["exchanges"]
        ):
            continue
        check(sha(render_prompt(original, before, row["question"])) == record["prompt_sha256"])
        official = api.applyModelOutput(original, before, row["question"], api.encodeCalls(row["calls"]))
        for field in ("exchanges", "semantic"):
            check(digest(official[field]) == record[f"expected_after_{field}_sha256"])
        for state, terminal in ((before, False), (official, True)):
            history = {k: v for k, v in api.sessionHistory(state).items() if k != "session_id"}
            key = digest(history)
            if key in states and (terminal or not states[key][2]):
                continue
            states[key] = {**state, "registry_id": revised_id}, row, terminal
    for key, (state, row, terminal) in states.items():
        for group, question, calls, expected, ancestors in cases_for(state, source, key):
            cid = "extra-history-" + sha(key + "\n" + question)[:24]
            check(cid not in seen, cid)
            seen.add(cid)
            target = check_continuation(revised, state, question, calls, expected)
            out = {
                "case_id": cid,
                "split": "train",
                "family_id": "replay-session:" + state["session_id"],
                "group": group,
                "question": question,
                "calls": calls,
                "target": target,
                "history_origin": "authenticated_"
                + row["history_origin"]
                + ("_plus_frozen_gold" if terminal else ""),
                "ancestor_case_ids": list(dict.fromkeys([row["case_id"], *ancestors])),
                "history_anchor": row["case_id"],
                "history_fingerprint": key,
                "frozen_terminal": terminal,
            }
            if options.get("render"):
                out["model_input"] = render_prompt(revised, state, question)
            options["emit"](out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    buildExtendedHistories({"render": parser.parse_args().render, "emit": send})
