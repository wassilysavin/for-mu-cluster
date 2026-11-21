"""Index the broad, frozen TRAIN replay and quarantine ambiguous supervision."""
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

from transfer_contract import MODEL_TURN, USER_TURN
from transfer_train import FROZEN, ROOT, compact, digest, generic_interval_conflict

SOURCE = "train-full-v2.jsonl"
CONTEXT_MANIFEST = "diagnostic-train-contexts.json"
CONTEXT_SHA = "b181085faba8cdb73bcf602e83c0eb0842f52496b4d85b29f9e584804add8aa3"


def text_hash(text):
    return sha256(text.encode("utf-8")).hexdigest()


def asset(root, name, expected):
    path = root / "assets" / name
    if path.resolve(strict=True) != path or digest(path) != expected:
        raise ValueError(f"Frozen TRAIN asset changed: {name}")
    return path


def read_contexts(assets, receipt):
    declared = receipt["contexts"]
    path = asset(assets.parent, "diagnostic-train-contexts.jsonl", declared["sha256"])
    contexts = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            cid = row["case_id"]
            if cid in contexts:
                raise ValueError(f"Repeated TRAIN context: {cid}")
            contexts[cid] = (row["prompt_sha256"], row["before"]["session_id"])
    if len(contexts) != receipt["restored_history_rows"]:
        raise ValueError("Incomplete original TRAIN histories")
    return contexts


def group_for(calls, history):
    names = {c["name"] for c in calls}
    if names == {"decline_request"}:
        return "outside_scope"
    if "decline_request" in names:
        return "mixed_scope"
    for name, group in (("repeat_exchange", "quote_whole"), ("retry_exchange", "retry"),
                        ("clarify_reference", "clarify"), ("repeat_answer", "repeat_answer")):
        if name in names:
            return group
    if any(c["arguments"].get("_clarify") for c in calls):
        return "clarify_argument"
    if any(name.startswith("change_") for name in names) or any("_replace" in c["arguments"] for c in calls):
        return "revision"
    if any(e["outcome"] == "error" for e in history["exchanges"]):
        return "after_error"
    if history["exchanges"]:
        return "history_other"
    return "composition" if len(calls) > 1 else "standalone"


def entry(row, offset, line, tokenizer, contexts):
    cid = row["case_id"]
    ids, cut = row["input_ids"], row["prompt_tokens"]
    if not 0 < cut <= 32768 - 512 or not 0 < len(ids) - cut <= 512 or ids[-1] != 50:
        raise ValueError(f"Incomplete TRAIN input or target: {cid}")
    prompt = tokenizer.decode(ids[:cut], skip_special_tokens=False)
    target = tokenizer.decode(ids[cut:], skip_special_tokens=False)
    if prompt.count(USER_TURN) != 1 or not prompt.endswith(MODEL_TURN):
        raise ValueError(f"Invalid native TRAIN prompt: {cid}")
    data = json.loads(prompt.split(USER_TURN)[1].removesuffix(MODEL_TURN))
    history = data["history"]
    session_id = history["session_id"]
    prompt_sha = text_hash(prompt)
    if data["question"] != row["question"] or not session_id:
        raise ValueError(f"Question or session mismatch in {cid}")
    restored = contexts.get(cid)
    if bool(history["exchanges"]) != (restored is not None) or restored and restored != (prompt_sha, session_id):
        raise ValueError(f"Historical TRAIN state differs: {cid}")
    # Remove only the opaque session UUID. Preserve exchanges, local addresses, focus and pending decisions.
    visible = {**data, "history": {k: v for k, v in history.items() if k != "session_id"}}
    return {"case_id": "week-replay:" + cid, "source_case_id": cid, "source": SOURCE,
            "family_id": "replay-session:" + session_id, "group": group_for(row["calls"], history),
            "origin": row["history_origin"], "offset": offset, "bytes": len(line),
            "row_sha256": sha256(line).hexdigest(), "source_prompt_sha256": prompt_sha,
            "visible_sha256": text_hash(compact(visible)), "target_sha256": text_hash(target),
            "source_prompt_tokens": cut, "target_tokens": len(ids) - cut}


def curate(candidates):
    retained, exclusions = [], {}
    by_visible = defaultdict(list)
    for record, reason in candidates:
        by_visible[record["visible_sha256"]].append((record, reason))
    for equivalents in by_visible.values():
        targets = {r["target_sha256"] for r, _ in equivalents}
        ambiguous = len(targets) > 1
        chosen = None
        for r, reason in sorted(equivalents, key=lambda item: item[0]["source_case_id"]):
            cid = r["source_case_id"]
            if ambiguous:
                exclusions[cid] = "conflicting_targets"
            elif reason:
                exclusions[cid] = reason
            elif chosen:
                exclusions[cid] = f"duplicate_of:{chosen}"
            else:
                retained.append(r)
                chosen = cid
    return sorted(retained, key=lambda r: r["offset"]), dict(sorted(exclusions.items()))


def select_replay(root=ROOT):
    """Return unique eligible row references and a reproducible source/coverage receipt."""
    from tokenizers import Tokenizer

    root = Path(root).resolve(strict=True)
    path = root / "assets" / SOURCE
    if path.resolve(strict=True) != path:
        raise ValueError("Expected original in-checkout TRAIN file")
    context_manifest = json.loads(asset(root, CONTEXT_MANIFEST, CONTEXT_SHA).read_bytes())
    if context_manifest["split"] != "train" or context_manifest["source_dataset"]["sha256"] != FROZEN[SOURCE]:
        raise ValueError("Unexpected TRAIN context source")
    contexts = read_contexts(path.parent, context_manifest)
    coverage_name = "history-common-coverage.json"
    coverage = json.loads(asset(root, coverage_name, FROZEN[coverage_name]).read_bytes())
    if coverage["source_sha256"] != FROZEN[SOURCE]:
        raise ValueError("Wrong source for original function inventory")
    tokenizer = Tokenizer.from_file(str(asset(root, "tokenizer.json", FROZEN["tokenizer.json"])))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    candidates, calls = [], {}
    source_hash = sha256()
    offset = 0
    with path.open("rb") as stream:
        for line in stream:
            source_hash.update(line)
            row = json.loads(line)
            cid = row["case_id"]
            if cid in calls:
                raise ValueError(f"Repeated original TRAIN case ID: {cid}")
            calls[cid] = {c["name"] for c in row["calls"]}
            r = entry(row, offset, line, tokenizer, contexts)
            reason = "legacy_hrf_interval_target" if generic_interval_conflict(row) else None
            candidates.append((r, reason))
            offset += len(line)
    if source_hash.hexdigest() != FROZEN[SOURCE] or not contexts.keys() <= calls.keys():
        raise ValueError("Wrong TRAIN bytes or invalid context IDs")
    selected, excluded = curate(candidates)
    ids = [r["source_case_id"] for r in selected]
    covered = Counter(name for cid in ids for name in calls[cid])
    required = {r["function"] for r in coverage["functions"]}
    if not required <= covered.keys():
        raise ValueError(f"Lost required scientific functions: {sorted(required - covered.keys())}")
    pairs = [(r["source_prompt_sha256"], r["target_sha256"]) for r, _ in candidates]
    receipt = {"source": {"path": f"assets/{SOURCE}", "sha256": FROZEN[SOURCE], "bytes": offset},
               "context_manifest_sha256": CONTEXT_SHA, "contexts_sha256": context_manifest["contexts"]["sha256"],
               "tokenizer_sha256": FROZEN["tokenizer.json"], "function_inventory_sha256": FROZEN[coverage_name],
               "source_rows": len(candidates), "selected_rows": len(selected), "excluded": excluded,
               "exact_source_input_target_duplicate_rows": len(pairs) - len(set(pairs)),
               "unique_selected_programs_native_text": len({r["target_sha256"] for r in selected}),
               "selected_origin_counts": dict(sorted(Counter(r["origin"] for r in selected).items())),
               "selected_group_counts": dict(sorted(Counter(r["group"] for r in selected).items())),
               "covered_scientific_functions": len(required), "selected_call_case_counts": dict(sorted(covered.items())),
               "selected_ref_sha256": text_hash(compact(selected))}
    return selected, receipt


def iter_replay(records, root=ROOT):
    """Read original rows in requested order. Verify their binding before exposing tokens."""
    root = Path(root).resolve(strict=True)
    path = root / "assets" / SOURCE
    if path.resolve(strict=True) != path:
        raise ValueError("Unexpected replay asset path")
    with path.open("rb") as stream:
        for ref in records:
            if ref["source"] != SOURCE or type(ref["offset"]) is not int or ref["offset"] < 0:
                raise ValueError("Invalid frozen replay reference")
            stream.seek(ref["offset"])
            line = stream.readline()
            if len(line) != ref["bytes"] or sha256(line).hexdigest() != ref["row_sha256"]:
                raise ValueError(f"Replay row changed: {ref['source_case_id']}")
            row = json.loads(line)
            if row["case_id"] != ref["source_case_id"] or ref["case_id"] != "week-replay:" + row["case_id"]:
                raise ValueError("Replay case identifier differs")
            yield ref, row
