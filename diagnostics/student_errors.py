"""Replay saved FunctionGemma dialogue evidence and classify supported-call discrepancies."""

from collections import Counter
import itertools
import re
import sys

from evaluation import api
from evaluation.actual_history import createActualHistoryHarness
from evaluation.functiongemma import decodeCalls
from evaluation.history_train import renderPrompt
from evaluation.schema import canonical, ContractError
from eval_tools.diagnose_dev import full_observation, loadPanel, summarize
from eval_tools.evaluate_actual import _materialize as replay_generation
from eval_tools.io import emit, records, require


REFERENCE_KEYS = frozenset(("target", "_replace", "_resolve", "_supersede"))
REFERENCE_ID = re.compile(r"(?:e[0-9]+|t[0-9]+\.[cp][0-9]+)")


def _differences(actual, expected):
    if sorted(map(canonical, actual)) == sorted(map(canonical, expected)):
        return {"signals": ["call_order"], "argument_fields": []}
    signals, fields = set(), set()
    if len(actual) < len(expected):
        signals.add("fewer_calls")
    if len(actual) > len(expected):
        signals.add("extra_calls")
    a_names, e_names = ([c["name"] for c in batch] for batch in (actual, expected))
    a, b = Counter(a_names), Counter(e_names)
    if a - b and b - a:
        signals.add("operation_substitution")
    if a_names != e_names and a == b:
        signals.add("call_order")
    if a_names == e_names:
        for left, right in zip(actual, expected):
            left, right = left["arguments"], right["arguments"]
            for key in left.keys() | right.keys():
                if key in left and key in right and canonical(left[key]) == canonical(right[key]):
                    continue
                fields.add(key)
                signals.add("missing_argument" if key not in left else "extra_argument" if key not in right else "argument_value")
                if key in REFERENCE_KEYS and any(isinstance(v, str) and REFERENCE_ID.fullmatch(v)
                                                 for v in (left.get(key), right.get(key))):
                    signals.add("reference_argument")
    return {"signals": sorted(signals), "argument_fields": sorted(fields)}


def comparePrograms(actual, allowedBatches):
    require(isinstance(allowedBatches, list) and allowedBatches, "Missing frozen allowed programs")
    original = canonical(actual)
    if any(canonical(expected) == original for expected in allowedBatches):
        return {"signals": [], "argument_fields": [], "comparison": "exact_allowed"}
    differences = [_differences(actual, batch) for batch in allowedBatches]
    common = {key: sorted(set.intersection(*(set(r[key]) for r in differences))) for key in ("signals", "argument_fields")}
    return {**common, "comparison": "different" if all(r == differences[0] for r in differences) else "alternative_dependent"}


def _decode(exchange):
    if exchange["outcome"] == "committed":
        return exchange["model_calls"]
    if exchange["outcome"] == "provider_error" or (exchange.get("generation") or {}).get("complete_native_batch") is False:
        return None
    try:
        return decodeCalls(exchange["raw_output"])
    except ContractError:
        return None


def _names(calls):
    return [c["name"][:80] for c in calls[:5]] + (["..."] if len(calls) > 5 else [])


def _targets(ids):
    return [v[:32] if isinstance(v, str) and REFERENCE_ID.fullmatch(v) else "(invalid ID)" for v in ids[:5]]


def caseRecord(row, observed, actual):
    exchange = observed["after"]["exchanges"][-1]
    result = {**{k: row[k] for k in ("case_id", "family_id", "panel", "cohort", "turn_index", "group") if k in row},
              "dialogue_id": row.get("dialogue_id"),
              **{k: observed[k] for k in ("success", "critical_status", "critical_category", "failure_category", "reference_success")
                 if k in observed},
              "outcome": exchange["outcome"], "error_code": exchange.get("error_code"),
              "comparison": "passed" if observed["success"] else "not_decodable", "signals": [], "argument_fields": []}
    if not observed["success"]:
        batches = observed["expected_model_call_batches"]
        options = observed.get("expected_reference_alternatives") or [[]]
        examples = list(dict.fromkeys(tuple(_names(batch)) for batch in batches))
        result.update(selected_targets=_targets(observed.get("selected_reference_targets") or []),
                      first_allowed_targets=_targets(options[0]), expected_names=[list(names) for names in examples[:3]],
                      actual_names=_names(actual) if actual is not None else None)
        if actual is not None:
            result.update(comparePrograms(actual, batches))
    return result


def inspectArm(context, observationsPath):
    core = api()
    harness = createActualHistoryHarness(core)
    last = {(r["panel"], r["dialogue_id"]): i for i, r in enumerate(context.plan) if r.get("dialogue_id")}
    sessions, cases, failures = {}, [], []
    stream = records(observationsPath)
    try:
        for index, (row, observed) in enumerate(itertools.zip_longest(context.plan, stream)):
            require(row is not None, "Extra saved DEV observation")
            require(observed is not None, "Incomplete saved DEV observations")
            require(observed.get("case_id") == row["case_id"] and observed.get("bundle_sha256") == context.bundleHash,
                    f"Different DEV case, order or bundle at row {index + 1}")
            key = (row["panel"], row.get("dialogue_id"))
            before = sessions.get(key) or harness.initial(context.registry, row)
            resolved = harness.resolve(row, before)
            prompt = renderPrompt(context.registry, core, before, resolved["question"])
            replay = full_observation(context, harness, row, before, resolved, prompt, observed["generation"], replay_generation)
            require(canonical(observed) == canonical(replay), "Saved state, prompt or grade differs from replay for " + row["case_id"])
            after = replay["after"]
            exchange = after["exchanges"][-1]
            if exchange["outcome"] == "provider_error" and exchange.get("error_code") != "context_overflow":
                failures.append(row["case_id"])
            cases.append(caseRecord(row, replay, None if replay["success"] else _decode(exchange)))
            if row.get("dialogue_id") and last[key] != index:
                sessions[key] = after
            else:
                sessions.pop(key, None)
    finally:
        stream.close()
    summary = summarize(context.data, cases)
    score = {**summary["summary"], **summary, "bundle_sha256": context.bundleHash, "diagnostic_reference_panel": "fixed_prefix",
             "infrastructure_failures": len(failures), "infrastructure_failure_case_ids": failures, "quality_valid": not failures}
    return {"rows": cases, "score": score, "bundle_sha256": context.bundleHash}


if __name__ == "__main__":
    try:
        require(len(sys.argv) == 4, "Expected registry, approved DEV bundle and observations path")
        emit(inspectArm(loadPanel(*sys.argv[1:3]), sys.argv[3]))
    except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
        sys.exit(f"Saved student DEV inspection failed: {error}")
