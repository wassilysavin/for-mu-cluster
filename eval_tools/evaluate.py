"""Evaluate a frozen DEV plan using a newline-delimited model-provider channel."""

import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace

from evaluation import api
from evaluation.development_evaluator import evaluateDevelopment, validateDevelopmentPlan
from evaluation.evaluation import summarizeEvaluation
from evaluation.history_train import renderPrompt
from evaluation.known_review import applyKnownReview
from evaluation.model_result import IncompleteModelOutput
from evaluation.schema import ContractError, canonical
from .io import ROOT, digest, emit, read_saved, receive, records, registry_at, require, save


LEGACY = {"cases": 700, "supported": 600, "outside_scope": 100, "dialogues": 60, "dialogue_version": 1}


def _session_id(identifier, spec):
    suite = "dev700/session-v1" if spec["dialogue_version"] == 1 else spec["id"] + "/session-v2"
    code = digest(f"gemprf/cluster/{suite}/{identifier}")
    return f"{code[:8]}-{code[8:12]}-4{code[13:16]}-8{code[17:20]}-{code[20:32]}"


def _suite(registry_path, dev_path, spec, known):
    version = spec["dialogue_version"]
    require(version in (1, 2) and spec["supported"] + spec["outside_scope"] == spec["cases"], "Invalid DEV specification")
    if version == 2:
        require(re.fullmatch(r"dev-volume-[a-z0-9-]+", spec["id"]), "Unexpected dialogue-v2 suite")
    registry, source = registry_at(registry_path)
    rows = list(records(dev_path))
    require(len(rows) == spec["cases"] == len({r["case_id"] for r in rows}), "DEV case count or uniqueness differs")
    for cohort in ("supported", "outside_scope"):
        require(sum(r["cohort"] == cohort for r in rows) == spec[cohort], f"Wrong {cohort} cohort size")
    require(len({r["dialogue_id"] for r in rows if r.get("dialogue_id")}) == spec["dialogues"], "Wrong DEV dialogue count")
    validateDevelopmentPlan(registry, rows, {"split": "dev", "dialogueVersion": version})
    reviews = {}
    for verdict in source["actual_history_reviews"] if version == 1 else ():
        key = (verdict["case_id"], verdict["before_visible_history_sha256"])
        require(verdict["registry_sha256"] == source["fingerprint"]
                and verdict["independence"]["current_question_candidate_output_read"] is False and key not in reviews,
                "Invalid or duplicate actual-history review")
        reviews[key] = verdict
    policy = None
    if known:
        policy = json.loads((ROOT / "assets/history-known-review.json").read_bytes())
        require(policy["registry_sha256"] == source["fingerprint"], "Known review registry differs")
    return SimpleNamespace(registry=registry, rows=rows, reviews=reviews, spec=spec, policy=policy)


def native_output(generation):
    raw, ids = generation["raw_text"], generation["raw_ids"]
    require(isinstance(raw, str) and isinstance(ids, list) and 0 < len(ids) <= 512
            and all(type(i) is int and 0 <= i < 262144 for i in ids), "Invalid native generation")
    if ids[-1] != 50:
        require(len(ids) == 512 and 50 not in ids, "Invalid incomplete native token sequence")
        raise IncompleteModelOutput(raw, ids, "token_limit")
    require(50 not in ids[:-1], "Extra native batch stop")
    return raw


def _evaluate(suite, output, reserve):
    known = suite.policy is not None
    prior = read_saved(output, retain_valid_tail=False) if known else []
    require(len(prior) <= len(suite.rows) and all(r["case_id"] == s["case_id"] for r, s in zip(prior, suite.rows)),
            "Saved known DEV prefix differs")
    reviewed, by_case = [], {r["case_id"]: r for r in suite.rows}
    replay_index = 0
    core = api()

    def adjudicate(request):
        verdict = suite.reviews.get((request["case_id"], request["before_visible_history_sha256"]))
        if verdict is None:
            return {"decision": "review"}
        require(verdict["question_sha256"] == digest(request["question"]), "Question fingerprint differs")
        return verdict

    def provider(request, metadata):
        if replay_index < len(prior):
            previous = prior[replay_index]
            require(previous["case_id"] == metadata["case_id"] and canonical(previous["before"]) == canonical(request["session"]),
                    "Saved provider observation differs")
            ex = previous["after"]["exchanges"][-1]
            if previous.get("provider_failure"):
                if ex.get("error_code") == "context_overflow":
                    raise ContractError("context_overflow", "Retained context overflow")
                failure = RuntimeError("Retained infrastructure failure")
                failure.code = previous["provider_failure"]
                raise failure
            if (ex.get("generation") or {}).get("complete_native_batch") is False:
                raise IncompleteModelOutput(ex["raw_output"], ex["generation"]["token_ids"], "token_limit")
            return ex["raw_output"]
        prompt = renderPrompt(suite.registry, core, request["session"], request["question"])
        emit({"kind": "generate", "case_id": metadata["case_id"], "prompt": prompt})
        return native_output(receive())

    with Path(output).open("ab" if known else "xb") as handle:
        def on_record(original):
            nonlocal replay_index
            record = original
            if known:
                record = applyKnownReview(suite.registry, by_case[original["case_id"]], original, suite.policy)
                reviewed.append(record["result"])
            index = replay_index
            replay_index += 1
            if index < len(prior):
                require(canonical(record) == canonical(prior[index]), "Resumed known DEV prefix differs")
                return
            save(handle, record, reserve)
            if known and record.get("provider_failure"):
                raise RuntimeError("Provider failed; its failed attempt was saved. Resume continues from that actual state.")
        score = evaluateDevelopment(suite.registry, suite.rows, {"split": "dev", "dialogueVersion": suite.spec["dialogue_version"],
                                    "sessionIdFor": lambda key: _session_id(key, suite.spec), "adjudicate": adjudicate,
                                    "provider": provider, "onRecord": on_record, "stopOnProviderFailure": not known})
    if known:
        failures = sum(bool(r.get("provider_failure")) and r["provider_failure"] != "context_overflow" for r in prior)
        score = {**score, "results": reviewed, "summary": summarizeEvaluation(suite.rows, reviewed),
                 "provider_failures": failures, "quality_valid": failures == 0}
    emit({"kind": "result", "score": score})
    return int(bool(score.get("aborted")))


def main(registry_path, dev_path, output, spec=None, reserve="0", *, known=False):
    suite = _suite(registry_path, dev_path, json.loads(spec) if spec else LEGACY, known)
    if output == "--check":
        emit({"kind": "check", "cases": len(suite.rows), "history_reviews": len(suite.reviews)})
        return 0
    return _evaluate(suite, output, int(reserve))


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
