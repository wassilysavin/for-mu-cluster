"""Build or inspect the standalone, prefix-shared GEM-pRF weekly TRAIN corpus."""

from private_runtime import python_tool
import argparse
from collections import Counter
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile

from private_runtime import python_tool
from source_provenance import runtime_sources
from transfer_contract import MODEL_TURN, USER_TURN, upgrade_replay
from data_tools.transfer_split import items, normalize, questions
from transfer_train import FROZEN, ROOT, compact, digest, tokenized
from data_tools.week_replay import iter_replay, select_replay

ASSETS = ROOT / "assets"
DATA = ASSETS / "week-train-v1.jsonl"
FORMAT = "gemprf-week-train-v1"
PUBLIC_DEV = ("dev700.jsonl", "dev-volume-v1.jsonl", "actual-history-v3-dev.json", "transfer-dev-v1.json")
SOURCES = ("week_train.py", "transfer_train.py", "transfer_contract.py", *runtime_sources(ROOT))
STREAMS = ("replay", "dialogue", "composition")


def ensure(ok, message):
    if not ok:
        raise ValueError(message)


def text_hash(text):
    return sha256(text.encode()).hexdigest()


def load_tokenizer():
    from tokenizers import Tokenizer

    path = ASSETS / "tokenizer.json"
    ensure(digest(path) == FROZEN[path.name], "Tokenizer differs from the FunctionGemma training tokenizer")
    tok = Tokenizer.from_file(str(path))
    tok.no_truncation()
    tok.no_padding()
    return tok


def visible(suffix):
    ensure(suffix.startswith(USER_TURN) and suffix.endswith(MODEL_TURN), "Malformed user/model turn")
    return json.loads(suffix[len(USER_TURN):-len(MODEL_TURN)])


def public_development():
    words, identifiers = set(), set()
    for name in PUBLIC_DEV:
        for entry in items(ASSETS / name):
            words.update(questions(entry))
            rows = [r for panel in entry["panels"].values() for r in panel] if "panels" in entry else [entry]
            for r in rows:
                identifiers.update(r[k] for k in ("case_id", "family_id", "bootstrap_family_id", "root_id")
                                   if r.get(k))
    return words, identifiers


def rendered(script):
    with subprocess.Popen([*python_tool(script), "--render"], cwd=ROOT, stdout=subprocess.PIPE) as proc:
        try:
            for line in proc.stdout:
                yield json.loads(line)
            ensure(proc.wait() == 0, f"Native TRAIN authoring failed in {script}")
        finally:
            if proc.poll() is None:
                proc.terminate()
            proc.stdout.close()


def replay_rows(refs, tok):
    with subprocess.Popen([*python_tool("data_tools/week_replay_check.py")], cwd=ROOT, stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, text=True, encoding="utf-8") as proc:
        try:
            for ref, original in iter_replay(refs):
                seq = upgrade_replay(original, tok)
                ids, cut = seq["input_ids"], seq["prompt_tokens"]
                prompt = tok.decode(ids[:cut], skip_special_tokens=False)
                target = tok.decode(ids[cut:], skip_special_tokens=False)
                request = {"case_id": ref["case_id"], "source_case_id": ref["source_case_id"], "model_input": prompt,
                           "target": target, "calls": original["calls"], "question": original["question"]}
                proc.stdin.write(compact(request) + "\n")
                proc.stdin.flush()
                reply = json.loads(proc.stdout.readline())
                ensure(reply["kind"] == "verified" and reply["case_id"] == request["case_id"]
                       and reply["target_sha256"] == ref["target_sha256"] and reply["prompt_sha256"] == text_hash(prompt),
                       f"Native replay verification failed: {request['case_id']}")
                yield {**request, "family_id": ref["family_id"], "group": ref["group"],
                       "history_origin": ref["origin"], "ancestor_case_ids": [ref["source_case_id"]]}, seq
            proc.stdin.close()
            end = json.loads(proc.stdout.readline())
            ensure(end["kind"] == "complete" and end["cases"] == len(refs) and proc.wait() == 0,
                   "Replay verifier did not finish")
        finally:
            if proc.poll() is None:
                proc.terminate()
            proc.stdout.close()


def source_rows(refs, tok):
    for row, tokens in replay_rows(refs, tok):
        yield "replay", row, tokens
    for stream, script in [("dialogue", "data_tools/week_dialogues.py"), ("composition", "data_tools/week_compositions.py")]:
        for row in rendered(script):
            yield stream, row, tokenized(row, tok)


def pack(stream, row, tokens):
    prompt = row["model_input"]
    ensure(prompt.count(USER_TURN) == 1, "Multiple user boundaries in the native prompt")
    cut = prompt.index(USER_TURN)
    pre, suffix = prompt[:cut], prompt[cut:]
    data = visible(suffix)
    ensure(data["question"] == row["question"], f"Question differs from native prompt: {row['case_id']}")
    history = {k: v for k, v in data["history"].items() if k != "session_id"}
    semantic_key = text_hash(compact({**data, "history": history}))
    case = {k: row[k] for k in ("case_id", "family_id", "group", "history_origin", "ancestor_case_ids",
                                 "question", "calls", "target")}
    case.update(type="example", split="train", stream=stream, prefix=text_hash(pre), input_suffix=suffix,
                prompt_tokens=tokens["prompt_tokens"], target_tokens=len(tokens["input_ids"]) - tokens["prompt_tokens"],
                tokens_sha256=text_hash(compact(tokens["input_ids"])))
    return case, pre, semantic_key, data


def same_corpus(published, rebuilt, original_sha):
    """Compare every regenerated row while allowing new author source receipts."""
    if published.is_symlink() or not published.is_file():
        return False
    checksum = digest(published)
    if checksum == digest(rebuilt):
        return True
    if checksum != original_sha:
        return False
    with published.open("rb") as old, rebuilt.open("rb") as new:
        new_header = new.readline()
        header, candidate = json.loads(old.readline()), json.loads(new_header)
        for obj in (header, candidate):
            if not obj.pop("author_files", None):
                return False
        if header != candidate:
            return False
        candidate_hash = sha256(new_header)
        previous = old.readline()
        for following in old:
            row = new.readline()
            if row != previous:
                return False
            candidate_hash.update(row)
            previous = following
        last = new.readline()
        if not last or not previous or new.read(1):
            return False
        footer, regenerated = json.loads(previous), json.loads(last)
        if footer.get("type") != "summary" or regenerated.pop("body_sha256", None) != candidate_hash.hexdigest():
            return False
        footer.pop("body_sha256")
        return footer == regenerated


def emit(out, h, record):
    line = (compact(record) + "\n").encode("utf-8")
    h.update(line)
    out.write(line)


def write_prefix(out, h, key, text, block_ids):
    # Tool schemas repeat across histories; splitting on a literal delimiter preserves every byte.
    ids, fresh = [], []
    start = len(block_ids)
    for fragment in text.split("},{"):
        if fragment not in block_ids:
            block_ids[fragment] = len(block_ids)
            fresh.append(fragment)
        ids.append(block_ids[fragment])
    if fresh:
        emit(out, h, {"type": "developer_blocks", "start": start, "fragments": fresh})
    emit(out, h, {"type": "developer_prefix", "sha256": key, "blocks": ids})


def build(path):
    tok = load_tokenizer()
    refs, replay_receipt = select_replay()
    public_words, public_ids = public_development()
    source_ids = {r["case_id"] for r in items(ASSETS / "train-full-v2.jsonl")}
    seen, seen_ids, families, prefixes = {}, set(), set(), set()
    block_ids = {}
    counts, groups, ops, lengths = Counter(), Counter(), Counter(), []
    tally = {"prompt": 0, "target": 0}
    declarations = {"type": "manifest", "format": FORMAT, "split": "train",
                    "tokenizer_sha256": FROZEN["tokenizer.json"], "registry_source_sha256": FROZEN["registry.json"],
                    "author_files": {n: digest(ROOT / n) for n in SOURCES}, "replay": replay_receipt,
                    "public_dev_sha256": {n: digest(ASSETS / n) for n in PUBLIC_DEV}}
    h = sha256()
    temp = None
    try:
        with NamedTemporaryFile(dir=path.parent, prefix=".week-train-", delete=False) as out:
            temp = Path(out.name)
            emit(out, h, declarations)
            for stream, row, seq in source_rows(refs, tok):
                case, pre, key, data = pack(stream, row, seq)
                cid = case["case_id"]
                ensure(cid not in seen_ids and cid not in public_ids and case["family_id"] not in public_ids,
                       f"Repeated TRAIN identity or public DEV identity: {cid}")
                ensure(set(case["ancestor_case_ids"]) <= source_ids and case["ancestor_case_ids"],
                       f"Unknown TRAIN ancestor: {cid}")
                ensure(not (set(questions(data)) & public_words), f"Public DEV wording in TRAIN: {cid}")
                ensure(key not in seen, f"Repeated or conflicting visible TRAIN input: {seen.get(key)} / {cid}")
                seen[key] = cid
                seen_ids.add(cid)
                if case["prefix"] not in prefixes:
                    prefixes.add(case["prefix"])
                    write_prefix(out, h, case["prefix"], pre, block_ids)
                emit(out, h, case)
                counts[stream] += 1
                groups[case["group"]] += 1
                ops.update({c["name"] for c in case["calls"]})
                families.add(case["family_id"])
                lengths.append(case["prompt_tokens"])
                for k in tally:
                    tally[k] += case[k + "_tokens"]
                if len(lengths) % 500 == 0:
                    print(f"Verified {len(lengths)} TRAIN cases", flush=True)
            ensure(set(counts) == set(STREAMS) and counts["replay"] == len(refs), "Incomplete TRAIN sources")
            ensure(declarations["author_files"] == {n: digest(ROOT / n) for n in SOURCES},
                   "Authoring code changed during dataset generation")
            ordered = sorted(lengths)
            summary = {"type": "summary", "cases": len(seen_ids), "families": len(families), "streams": dict(counts),
                       "groups": dict(groups), "distinct_call_names": len(ops), "prefixes": len(prefixes),
                       "token_counts": tally, "p90_prompt_tokens": ordered[(len(ordered)*9-1)//10],
                       "max_prompt_tokens": ordered[-1], "body_sha256": h.hexdigest()}
            out.write((compact(summary) + "\n").encode())
            out.flush()
            os.fsync(out.fileno())
        try:
            os.link(temp, path)
        except FileExistsError:
            ensure(same_corpus(path, temp, "5187779fc940c9ff19adaa90fee2233b9ea5815a5acb74420b0f012af12ae50e"),
                   "Existing weekly TRAIN differs; publish revisions separately")
        return summary
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def scan(path, tok=None):
    """Verify the receipt and index examples without keeping expanded prompts in memory."""
    prefixes, index, counts, h = {}, {}, Counter(), sha256()
    blocks = []
    with path.open("rb") as f:
        header_line = f.readline()
        header = json.loads(header_line)
        ensure(header["type"] == "manifest" and header["format"] == FORMAT and header["split"] == "train"
               and header["tokenizer_sha256"] == FROZEN["tokenizer.json"], "Unexpected corpus identity")
        h.update(header_line)
        while True:
            offset = f.tell()
            line = f.readline()
            ensure(line, "Missing final corpus receipt")
            row = json.loads(line)
            if row["type"] == "summary":
                ensure(not f.read(1) and row["body_sha256"] == h.hexdigest() and row["cases"] == len(index)
                       and row["prefixes"] == len(prefixes) and row["streams"] == counts and set(counts) == set(STREAMS),
                       "Dataset checksum or coverage differs")
                return prefixes, index, row
            h.update(line)
            if row["type"] == "developer_blocks":
                ensure(row["start"] == len(blocks) and isinstance(row["fragments"], list)
                       and all(isinstance(x, str) for x in row["fragments"]), "Invalid developer text dictionary")
                blocks.extend(row["fragments"])
                continue
            if row["type"] == "developer_prefix":
                ids = row["blocks"]
                ensure(isinstance(ids, list) and ids and all(type(i) is int and 0 <= i < len(blocks) for i in ids),
                       "Invalid developer text reference")
                text = "},{".join(blocks[i] for i in ids)
                k = text_hash(text)
                ensure(row["sha256"] == k and k not in prefixes, "Invalid shared prefix")
                prefixes[k] = text
                continue
            ensure(row["type"] == "example" and row["split"] == "train" and row["stream"] in STREAMS
                   and row["case_id"] not in index and row["prefix"] in prefixes, "Invalid TRAIN identity")
            index[row["case_id"]] = offset
            counts[row["stream"]] += 1
            if tok is not None:
                unpack(row, prefixes, tok)


def unpack(row, prefixes, tok):
    prompt = prefixes[row["prefix"]] + row["input_suffix"]
    ensure(visible(row["input_suffix"])["question"] == row["question"], "Incorrect visible question")
    seq = tokenized({"model_input": prompt, "case_id": row["case_id"], "target": row["target"]}, tok)
    ensure(seq["prompt_tokens"] == row["prompt_tokens"] and len(seq["input_ids"]) - seq["prompt_tokens"] == row["target_tokens"]
           and text_hash(compact(seq["input_ids"])) == row["tokens_sha256"], f"Native tokens differ: {row['case_id']}")
    return seq


def iter_epoch(path=DATA, *, epoch=0, seed=1729):
    """Yield exact input IDs and the loss-mask boundary in a reproducible, mixed order."""
    ensure(type(epoch) is int and epoch >= 0, "Expected a nonnegative epoch")
    tok = load_tokenizer()
    prefixes, index, _ = scan(path)
    with path.open("rb") as f:
        for cid in sorted(index, key=lambda c: text_hash(compact([seed, epoch, c]))):
            f.seek(index[cid])
            row = json.loads(f.readline())
            ensure(row["case_id"] == cid, "Corpus changed after indexing")
            yield {"case_id": cid, **unpack(row, prefixes, tok)}


def iter_batches(path=DATA, *, updates, seed=1729):
    """Read exactly four examples per optimizer update; carry the final partial epoch batch forward."""
    ensure(type(updates) is int and updates >= 0, "Expected a nonnegative number of optimizer updates")
    epoch, batch = 0, []
    while updates:
        for case in iter_epoch(path, epoch=epoch, seed=seed):
            batch.append(case)
            if len(batch) == 4:
                yield batch
                batch = []
                updates -= 1
                if not updates:
                    return
        epoch += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["build", "check"])
    args = parser.parse_args()
    if args.action == "build":
        build(DATA)
    _, _, summary = scan(DATA, load_tokenizer())
    print(f"{summary['cases']} examples in {summary['families']} families: {summary['streams']}")
    print(f"{DATA.name}: {DATA.stat().st_size:,} bytes, SHA-256 {digest(DATA)}")


if __name__ == "__main__":
    main()
