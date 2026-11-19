"""A larger weekly TRAIN and a compute-matched 5,169-case control."""
import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import week_train as base

DATA = base.ASSETS / "week-train-v2.jsonl"
PARENT_SHA = "5187779fc940c9ff19adaa90fee2233b9ea5815a5acb74420b0f012af12ae50e"
CONTROL_SIZE = 5169
FORMAT = "gemprf-week-train-expanded-v2"
AUTHORS = {"history_followup": "extended_histories.mjs", "scientific_conditions": "extended_conditions.mjs",
           "scope_change": "extended_scope.mjs"}
SOURCE_FILES = (*base.SOURCES, "extended_train.py", *AUTHORS.values())


def file_bindings():
    return {name: base.digest(base.ROOT / name) for name in SOURCE_FILES}


def visible_key(data):
    history = {k: v for k, v in data["history"].items() if k != "session_id"}
    return base.text_hash(base.compact({**data, "history": history}))


def stratum(row):
    return (row["stream"], row["group"], row["prompt_tokens"] // 2048, row["target_tokens"] // 32)


def order(seed, label, item):
    return base.text_hash(base.compact([seed, label, item]))


def choose_control(groups, size=CONTROL_SIZE, seed=1729):
    """Allocate proportional quotas, retaining at least one representative of rare strata."""
    total = sum(map(len, groups.values()))
    base.ensure(len(groups) <= size <= total and all(groups.values()), "Control cannot cover all TRAIN strata")
    quotas = {g: max(1, size * len(ids) // total) for g, ids in groups.items()}
    gap = size - sum(quotas.values())
    while gap:
        if gap > 0:
            g = min((g for g in groups if quotas[g] < len(groups[g])),
                    key=lambda g: (quotas[g] * total - size * len(groups[g]), g))
            quotas[g] += 1
            gap -= 1
        else:
            g = min((g for g in groups if quotas[g] > 1),
                    key=lambda g: (size * len(groups[g]) - quotas[g] * total, g))
            quotas[g] -= 1
            gap += 1
    return {g: sorted(ids, key=lambda cid: order(seed, "control membership", cid))[:quotas[g]]
            for g, ids in groups.items()}


def draws(groups, *, arm, updates, seed=1729):
    """Both sizes see the same sequence of strata at every optimizer update."""
    base.ensure(arm in {"full", "matched"} and type(updates) is int and updates >= 0, "Invalid TRAIN schedule")
    membership = choose_control(groups, seed=seed) if arm == "matched" else groups
    by_id = {cid: g for g, ids in groups.items() for cid in ids}
    base.ensure(len(by_id) == sum(map(len, groups.values())), "Repeated TRAIN case ID")
    cycles, queues = Counter(), {}
    remaining = 4 * updates
    epoch = 0
    while remaining:
        for cid in sorted(by_id, key=lambda i: order(seed, epoch, i)):
            g = by_id[cid]
            if arm == "matched":
                if not queues.get(g):
                    queues[g] = sorted(membership[g], key=lambda i: order(seed, ["cycle", cycles[g]], i))
                    cycles[g] += 1
                cid = queues[g].pop()
            yield cid
            remaining -= 1
            if not remaining:
                return
        epoch += 1


class Inventory:
    def __init__(self):
        self.ids = {}
        self.targets = {}
        self.families = set()
        self.streams = Counter()
        self.groups = Counter()
        self.lengths = []
        self.tokens = Counter()
        self.calls = set()
        self.strata = defaultdict(list)
        self.questions = defaultdict(set)
        self.origins = Counter()
        self.programs = set()

    def add(self, r, key):
        cid = r["case_id"]
        base.ensure(cid not in self.ids and key not in self.targets, f"Duplicate TRAIN example: {cid}")
        self.ids[cid] = r["tokens_sha256"]
        self.targets[key] = base.text_hash(r["target"])
        self.families.add(r["family_id"])
        self.streams[r["stream"]] += 1
        self.groups[r["group"]] += 1
        self.lengths.append(r["prompt_tokens"])
        self.tokens.update(prompt=r["prompt_tokens"], target=r["target_tokens"])
        self.calls.update(c["name"] for c in r["calls"])
        self.strata[stratum(r)].append(cid)
        self.questions[r["stream"]].add(base.normalize(r["question"]))
        self.origins[r["history_origin"]] += 1
        self.programs.add(base.text_hash(base.compact(r["calls"])))

    def summary(self):
        lengths = sorted(self.lengths)
        control = choose_control(self.strata)
        members = sorted(cid for ids in control.values() for cid in ids)
        parent_words = set().union(*(self.questions[k] for k in base.STREAMS))
        expanded_words = set().union(*self.questions.values())
        return {"cases": len(self.ids), "distinct_final_questions": len(expanded_words),
                "added_distinct_final_questions": len(expanded_words - parent_words),
                "history_origins": dict(self.origins), "distinct_target_programs": len(self.programs), "families": len(self.families), "streams": dict(self.streams),
                "groups": dict(self.groups), "token_counts": dict(self.tokens), "distinct_call_names": len(self.calls),
                "distinct_final_questions_by_stream": {k: len(v) for k, v in self.questions.items()},
                "p90_prompt_tokens": lengths[(9 * len(lengths) - 1) // 10], "max_prompt_tokens": lengths[-1],
                "matched_control": {"cases": len(members), "seed": 1729, "strata": len(control),
                                    "membership_sha256": base.text_hash(base.compact(members))}}


def parent_records():
    base.ensure(base.digest(base.DATA) == PARENT_SHA, "Frozen weekly control file changed")
    with base.DATA.open("rb") as f:
        header = json.loads(next(f))
        base.ensure(header["format"] == base.FORMAT and header["split"] == "train", "Unexpected parent format")
        for line in f:
            r = json.loads(line)
            if r["type"] == "summary":
                base.ensure(r["cases"] == CONTROL_SIZE and not f.read(1), "Parent dataset incomplete")
                return
            yield line, r
    raise ValueError("Missing parent TRAIN footer")


def build(path=DATA):
    tok = base.load_tokenizer()
    words, dev_ids = base.public_development()
    ancestors = {r["case_id"] for r in base.items(base.ASSETS / "train-full-v2.jsonl")}
    inputs = file_bindings()
    header = {"type": "manifest", "format": FORMAT, "split": "train", "parent_sha256": PARENT_SHA,
              "tokenizer_sha256": base.FROZEN["tokenizer.json"], "registry_source_sha256": base.FROZEN["registry.json"],
              "train_source_sha256": base.FROZEN["train-full-v2.jsonl"], "author_files": inputs,
              "public_dev_sha256": {n: base.digest(base.ASSETS / n) for n in base.PUBLIC_DEV}}
    h, parent_body = sha256(), sha256()
    census, prefixes, blocks, filtered = Inventory(), set(), {}, Counter()
    temp = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix=".week-expanded-", delete=False) as out:
            temp = Path(out.name)
            base.emit(out, h, header)
            for line, row in parent_records():
                out.write(line)
                h.update(line)
                parent_body.update(line)
                if row["type"] == "example":
                    census.add(row, visible_key(base.visible(row["input_suffix"])))
                elif row["type"] == "developer_blocks":
                    base.ensure(row["start"] == len(blocks), "Invalid parent dictionary")
                    for block in row["fragments"]:
                        blocks[block] = len(blocks)
                elif row["type"] == "developer_prefix":
                    prefixes.add(row["sha256"])
                else:
                    raise ValueError("Unknown parent row")
            base.ensure(len(census.ids) == CONTROL_SIZE, "Incomplete frozen control")
            for stream, script in AUTHORS.items():
                for raw in base.rendered(script):
                    cid = raw["case_id"]
                    base.ensure(raw.get("split", "train") == "train" and cid not in census.ids and cid not in dev_ids
                                and raw["family_id"] not in dev_ids, f"Case or family identity clash: {cid}")
                    base.ensure(raw["ancestor_case_ids"] and set(raw["ancestor_case_ids"]) <= ancestors,
                                f"Undeclared TRAIN source: {cid}")
                    seq = base.tokenized(raw, tok)
                    row, developer, key, data = base.pack(stream, raw, seq)
                    if set(base.questions(data)) & words:
                        filtered[stream + "/public_development_wording"] += 1
                        continue
                    if key in census.targets:
                        base.ensure(census.targets[key] == base.text_hash(row["target"]),
                                    f"Conflicting targets for identical visible histories: {cid}")
                        filtered[stream + "/identical_input_and_target"] += 1
                        continue
                    if row["prefix"] not in prefixes:
                        prefixes.add(row["prefix"])
                        base.write_prefix(out, h, row["prefix"], developer, blocks)
                    base.emit(out, h, row)
                    census.add(row, key)
                base.ensure(census.streams[stream] > 0, f"Empty extension: {stream}")
                print(f"{stream}: {census.streams[stream]} new cases", flush=True)
            base.ensure(file_bindings() == inputs, "Authoring code changed during build")
            footer = {"type": "summary", **census.summary(), "prefixes": len(prefixes), "filtered": dict(filtered),
                      "parent_records_sha256": parent_body.hexdigest(), "body_sha256": h.hexdigest()}
            out.write((base.compact(footer) + "\n").encode())
            out.flush()
            os.fsync(out.fileno())
        try:
            os.link(temp, path)
        except FileExistsError:
            base.ensure(not path.is_symlink() and base.digest(temp) == base.digest(path),
                        "Existing corpus differs; publish a separate revision")
        return footer
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def inspect(path=DATA, tok=None):
    """Check the complete artifact, exact baseline inclusion and independently recounted metrics."""
    blocks, prefixes, offsets = [], {}, {}
    inventory, h = Inventory(), sha256()
    parent = iter(parent_records())
    parent_line = next(parent, None)
    parent_hash = sha256()
    with path.open("rb") as f:
        line = f.readline()
        header = json.loads(line)
        base.ensure(header["format"] == FORMAT and header["type"] == "manifest" and header["split"] == "train"
                    and header["parent_sha256"] == PARENT_SHA and header["tokenizer_sha256"] == base.FROZEN["tokenizer.json"]
                    and header["registry_source_sha256"] == base.FROZEN["registry.json"], "Unexpected corpus binding")
        h.update(line)
        while True:
            pos = f.tell()
            line = f.readline()
            base.ensure(line, "Missing expanded TRAIN receipt")
            row = json.loads(line)
            if parent_line is not None:
                base.ensure(line == parent_line[0], "Frozen parent records changed or were omitted")
                parent_hash.update(line)
                parent_line = next(parent, None)
            kind = row["type"]
            if kind == "summary":
                expected = inventory.summary()
                base.ensure(not f.read(1) and parent_line is None and row["parent_records_sha256"] == parent_hash.hexdigest()
                            and row["body_sha256"] == h.hexdigest() and row["prefixes"] == len(prefixes)
                            and all(row[k] == v for k, v in expected.items())
                            and set(inventory.streams) == set(base.STREAMS) | AUTHORS.keys(), "Invalid final TRAIN receipt")
                return prefixes, offsets, inventory.strata, row
            h.update(line)
            if kind == "developer_blocks":
                base.ensure(row["start"] == len(blocks) and all(isinstance(x, str) for x in row["fragments"]),
                            "Changed developer block sequence")
                blocks.extend(row["fragments"])
            elif kind == "developer_prefix":
                ids = row["blocks"]
                base.ensure(ids and all(type(i) is int and 0 <= i < len(blocks) for i in ids), "Invalid prefix reference")
                text = "},{".join(blocks[i] for i in ids)
                key = base.text_hash(text)
                base.ensure(key == row["sha256"] and key not in prefixes, "Invalid developer prefix")
                prefixes[key] = text
            else:
                base.ensure(kind == "example" and row["split"] == "train" and row["prefix"] in prefixes, "Invalid case")
                data = base.visible(row["input_suffix"])
                base.ensure(data["question"] == row["question"], "Visible question differs")
                inventory.add(row, visible_key(data))
                offsets[row["case_id"]] = pos
                if tok is not None:
                    base.unpack(row, prefixes, tok)


def iter_batches(*, arm, updates, seed=1729, path=DATA):
    tok = base.load_tokenizer()
    prefixes, offsets, strata, _ = inspect(path)
    batch = []
    with path.open("rb") as f:
        for cid in draws(strata, arm=arm, updates=updates, seed=seed):
            f.seek(offsets[cid])
            row = json.loads(f.readline())
            base.ensure(row["case_id"] == cid, "TRAIN changed after indexing")
            batch.append({"case_id": cid, **base.unpack(row, prefixes, tok)})
            if len(batch) == 4:
                yield batch
                batch = []


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "check"))
    args = parser.parse_args()
    if args.action == "build":
        build()
    _, _, _, result = inspect(DATA, base.load_tokenizer())
    print(f"{result['cases']} TRAIN cases, including {CONTROL_SIZE} preserved cases; {result['streams']}")
    print(f"Matched control: {result['matched_control']['cases']}, {result['matched_control']['strata']} strata")
    print(f"{DATA.name}: {DATA.stat().st_size:,} bytes, SHA-256 {base.digest(DATA)}")


if __name__ == "__main__":
    main()
