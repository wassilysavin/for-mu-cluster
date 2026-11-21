"""Audit fresh DEV against the declared TRAIN lineages, historical prompts and known DEV."""
from pathlib import Path
import json
import re
import unicodedata

ASSETS = Path(__file__).resolve().parents[1] / "assets"
HISTORICAL = ("train-full-v2.jsonl", "history-train-specs.jsonl", "diagnostic-train-contexts.jsonl",
              "dev700.jsonl", "dev-volume-v1.jsonl", "actual-history-v3-dev.json")


def normalize(text):
    return " ".join(re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", text).casefold()))


def questions(obj):
    if isinstance(obj, list):
        for child in obj:
            if isinstance(child, (dict, list)):
                yield from questions(child)
    elif isinstance(obj, dict):
        for key, value in obj.items():
            if (key == "question" or key.endswith("_question")) and isinstance(value, str):
                yield normalize(value)
            elif key == "questions":
                variants = value.values() if isinstance(value, dict) else value
                for variant in variants:
                    if not isinstance(variant, str):
                        raise ValueError("Nontext question variant")
                    yield normalize(variant)
            elif key not in {"input_ids", "positions", "raw_ids"} and isinstance(value, (dict, list)):
                yield from questions(value)


def items(path):
    with path.open(encoding="utf-8") as stream:
        if path.suffix == ".json":
            yield json.load(stream)
        else:
            for line in stream:
                yield json.loads(line)


def check_separation():
    data, = items(ASSETS / "transfer-dev-v1.json")
    dev_rows = [row for panel in data["panels"].values() for row in panel]
    fresh = set(questions(dev_rows))
    dev_ids = {r[key] for r in dev_rows for key in ("case_id", "family_id", "bootstrap_family_id")}
    train_ids, counts = set(), {}
    new = tuple(f"transfer-train-{part}.jsonl" for part in ("roots", "ordinary", "contrast"))
    for name in new + HISTORICAL:
        seen = set()
        for row in items(ASSETS / name):
            seen.update(questions(row))
            if name in new:
                train_ids.update(row[k] for k in ("lineage", "root_id", "case_id") if k in row)
        overlap = seen & fresh
        if overlap:
            raise ValueError(f"Fresh DEV repeats {name} wording: {sorted(overlap)[:3]}")
        if not seen:
            raise ValueError(f"No question text audited in {name}")
        counts[name] = len(seen)
    if train_ids & dev_ids:
        raise ValueError("TRAIN and fresh DEV share declared case or family identifiers")
    return {"fresh_distinct_questions": len(fresh), "source_distinct_questions": counts,
            "normalized_question_overlaps": 0, "declared_case_or_family_id_overlaps": 0}
