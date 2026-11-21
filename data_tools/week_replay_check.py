"""Verify each migrated replay prompt and complete gold transition."""

import json
import sys

from .common import ASSETS, REGISTRY_SHA, SOURCE_SHA, check, read_json, rows, send, sha

CONTEXT_SHA = "b181085faba8cdb73bcf602e83c0eb0842f52496b4d85b29f9e584804add8aa3"
USER_TURN = "<end_of_turn>\n<start_of_turn>user\n"
MODEL_TURN = "<end_of_turn>\n<start_of_turn>model\n"


def index(name, digest, project):
    data = {}
    for row in rows(ASSETS / name, digest):
        key = row["case_id"]
        check(key not in data, f"Repeated frozen TRAIN ID: {key}")
        data[key] = project(row)
    return data


def verifier():
    from eval_tools.new_train_fit import createScorer
    from evaluation.schema import canonical
    from evaluation.session import applyModelOutput, createSession

    manifest = read_json("diagnostic-train-contexts.json", CONTEXT_SHA)
    check(manifest["split"] == "train" and manifest["source_dataset"]["sha256"] == SOURCE_SHA)
    contexts = index("diagnostic-train-contexts.jsonl", manifest["contexts"]["sha256"], lambda r: r)
    source = index("train-full-v2.jsonl", SOURCE_SHA, lambda r: (r["question"], r["calls"]))
    scorer = createScorer(read_json("registry.json", REGISTRY_SHA))
    registry_id = scorer.registry.fingerprint()

    def verify(row):
        case, source_id = row["case_id"], row["source_case_id"]
        question, calls, target = row["question"], row["calls"], row["target"]
        check(case == "week-replay:" + source_id and source[source_id] == (question, calls), case)
        parts = row["model_input"].split(USER_TURN)
        check(len(parts) == 2 and parts[1].endswith(MODEL_TURN), case)
        data = json.loads(parts[1][: -len(MODEL_TURN)])
        check(data["question"] == question)
        saved = contexts.get(source_id)
        history = data["history"]
        check(bool(saved) == bool(history["exchanges"]))
        before = (
            saved["before"]
            if saved
            else createSession(
                scorer.original,
                {"sessionId": history["session_id"], "dialogueVersion": 2 if data["format"].endswith("v2") else 1},
            )
        )
        proof = scorer.prepare(
            {"case_id": case, "prompt": row["model_input"], "target": target},
            {"case_id": case, "question": question, "calls": calls, "group": "shared_replay"},
            before,
            {"section": "shared", "migrated": True, "branch": "frozen_train"},
        )
        old = applyModelOutput(scorer.original, before, question, target)
        now = applyModelOutput(scorer.registry, {**before, "registry_id": registry_id}, question, target)
        check(now["exchanges"][-1]["outcome"] == "committed", case)
        check(now["exchanges"] == old["exchanges"] and now["semantic"] == old["semantic"], case)
        if saved:
            check(sha(canonical(before)) == saved["before_sha256"])
            for key in ("semantic", "exchanges"):
                check(sha(canonical(now[key])) == saved[f"expected_after_{key}_sha256"])
        visible = {k: v for k, v in history.items() if k != "session_id"}
        return {
            "kind": "verified",
            "case_id": case,
            "source_case_id": source_id,
            "prompt_sha256": proof["prompt_sha256"],
            "target_sha256": proof["target_sha256"],
            "visible_sha256": sha(canonical({**data, "history": visible})),
            "group": proof["group"],
        }

    return verify


if __name__ == "__main__":
    verify = verifier()
    seen = set()
    for raw in sys.stdin:
        request = json.loads(raw)
        key = request["case_id"]
        check(key not in seen, f"Duplicate TRAIN replay request: {key}")
        send(verify(request))
        seen.add(key)
    send({"kind": "complete", "cases": len(seen)})
