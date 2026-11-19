"""Summarize the published TRAIN without loading a model or reading held-out examples."""
from collections import Counter, defaultdict
import json
from math import ceil

import extended_train as ext


def wording_probe(records):
    """Predict from exact question wording seen in other TRAIN families."""
    votes = defaultdict(Counter)
    local = defaultdict(Counter)
    for q, family, label in records:
        votes[q][label] += 1
        local[q, family][label] += 1
    result = Counter(cases=len(records))
    for q, family, label in records:
        other = votes[q] - local[q, family]
        if other:
            result["seen_wording_elsewhere"] += 1
            # A deterministic tie break uses no information from the held-out family.
            prediction = min(other, key=lambda x: (-other[x], x))
            result["correct_among_seen"] += int(label == prediction)
    return dict(result)


def audit(path=ext.DATA):
    _, offsets, strata, receipt = ext.inspect(path)
    labels, programs, lengths, by_id = [], [], {}, {}
    ancestry, family_by_id, targets = {}, {}, {}
    origins = defaultdict(Counter)
    with path.open("rb") as source:
        header = json.loads(source.readline())
        ext.base.ensure(header["author_files"] == ext.file_bindings(), "Dataset and author revision differ")
        ext.base.ensure(header["train_source_sha256"] == ext.base.FROZEN["train-full-v2.jsonl"]
                        and header["public_dev_sha256"] == {n: ext.base.digest(ext.base.ASSETS / n)
                                                           for n in ext.base.PUBLIC_DEV},
                        "Dataset and source manifest differ")
        for line in source:
            row = json.loads(line)
            if row["type"] != "example":
                continue
            cid, stream = row["case_id"], row["stream"]
            question = ext.base.normalize(row["question"])
            family = row["family_id"]
            calls = row["calls"]
            labels.append((question, family, tuple(call["name"] for call in calls)))
            targets[cid] = ext.base.compact(calls)
            programs.append((question, family, targets[cid]))
            lengths[cid] = row["prompt_tokens"], row["target_tokens"]
            ancestry[cid] = row["ancestor_case_ids"]
            family_by_id[cid] = family
            origins[stream][row["history_origin"]] += 1
            by_id[cid] = ext.stratum(row)
    updates = ceil(len(offsets) / 4)
    schedules = {arm: list(ext.draws(strata, arm=arm, updates=updates)) for arm in ("full", "matched")}
    ext.base.ensure(all(by_id[a] == by_id[b] for a, b in zip(schedules["full"], schedules["matched"], strict=True)),
                    "Task or token-length stratum differs across matched schedules")
    totals = {arm: {"unique_examples": len(set(ids)), "presentations": len(ids),
                    "families": len({family_by_id[c] for c in ids}),
                    "source_train_ids": len({ancestor for c in ids for ancestor in ancestry[c]}),
                    "target_programs": len({targets[c] for c in ids}),
                    "prompt_tokens": sum(lengths[c][0] for c in ids), "target_tokens": sum(lengths[c][1] for c in ids)}
              for arm, ids in schedules.items()}
    return {"file_sha256": ext.base.digest(path), "size_bytes": path.stat().st_size, "updates_per_arm": updates,
            "schedule_totals": totals, "wording_only_function_names": wording_probe(labels),
            "wording_only_full_program": wording_probe(programs), "origins_by_stream": origins,
            "corpus": receipt}


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2, ensure_ascii=False))
