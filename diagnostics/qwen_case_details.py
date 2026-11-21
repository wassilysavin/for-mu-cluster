"""Trace saved qualification calls and compare disputed answers against each allowed program."""

import json
import re
import sys

from eval_tools.diagnose_data import createTrainDiagnostics
from eval_tools.io import digest, registry_at, require
from evaluation.actual_history import refs
from evaluation.dialogue import executeTurn
from evaluation.functiongemma import decodeCalls, encodeCalls
from evaluation.model_tools import compileModelCalls, modelTools
from evaluation.schema import ContractError, canonical, clone, js_json, validate
from evaluation.session import applyModelOutput, recoveryExchanges


def _attempt(operation, *args):
    try:
        return {"value": operation(*args)}
    except ContractError as e:
        return {"error": {"code": e.code, "message": e.message, "details": clone(e.details)}}


def _first_failure(registry, before, question, calls, internal, stage, error):
    exchanges = recoveryExchanges(before)
    state = before["semantic"]
    if stage == "execute" and any(
        c["name"] == "complete_call"
        and any(p["id"] == c["arguments"]["target"] and p["mode"] == "each" for p in state["pending"])
        for c in internal
    ):
        return dict(call_index=None, attribution="unassigned_group_finalization_depends_on_full_batch")
    for count in range(1, len(calls) + 1):
        found = (
            _attempt(compileModelCalls, registry, state, calls[:count], exchanges)
            if stage == "compile"
            else executeTurn(registry, state, question, internal[:count], exchanges)
        ).get("error")
        if found is None:
            continue
        if canonical(found) == canonical(error):
            reason = (
                "first_failing_compile_prefix_same_frozen_state"
                if stage == "compile"
                else "first_failing_execution_prefix_no_group_finalization"
            )
            return dict(call_index=count - 1, attribution=reason)
        if stage == "execute" and found["code"] == "invalid_retry" and internal[0]["name"] == "retry_exchange":
            continue
        return dict(call_index=None, attribution="unassigned_prefix_differs_from_full_batch")
    return dict(call_index=None, attribution="unassigned")


def _trace(registry, item, full):
    before = item["before"]
    state, exchanges = before["semantic"], recoveryExchanges(before)
    decoded = _attempt(decodeCalls, full["generation"]["raw_text"])
    if "error" in decoded:
        return dict(model_calls=None, trace=dict(stage="decode", error=decoded["error"], call_index=None))
    calls = decoded["value"]
    compiled = _attempt(compileModelCalls, registry, state, calls, exchanges)
    internal = compiled.get("value")
    stage = "compile"
    failure = compiled.get("error")
    if failure is None:
        stage = "execute"
        executed = executeTurn(registry, state, item["question"], internal, exchanges)
        failure = executed.get("error")
    saved = full["after"]["exchanges"][-1]
    if failure is not None:
        require(saved["outcome"] == "error", "Full-batch replay differs from saved outcome")
        require(saved["error_code"] == failure["code"], "Full-batch replay differs from saved error")
        trace = dict(
            stage=stage,
            error=failure,
            **_first_failure(registry, before, item["question"], calls, internal, stage, failure),
        )
    else:
        require(saved["outcome"] == "committed", "Full-batch replay differs from saved outcome")
        require(saved["assistant"] == executed["turn"]["text"], "Full-batch delivered text differs")
        require(
            canonical(full["after"]["semantic"]) == canonical(executed["state"]),
            "Full-batch semantic state differs",
        )
        trace = dict(stage="committed", error=None, call_index=None, attribution="no_execution_failure")
    return dict(model_calls=calls, internal_calls=internal, trace=trace)


def _schema_fragment(schema, message):
    match = re.match(r"^(\$(?:\.[^.:\[\]\s]+|\[\d+\])*):", message or "")
    path = match[1] if match else "$"
    selected = schema
    for key in re.findall(r"[^.\[\]$]+|\[\d+\]", path):
        selected = (
            (selected or {}).get("items")
            if key.startswith("[")
            else (selected or {}).get("properties", {}).get(key)
        )
    rule = re.search(r": (?:missing|unexpected) (\w+)$", message or "", flags=re.ASCII)
    if rule and selected and selected.get("type") == "object":
        field = rule[1]
        props = selected.get("properties", {})
        return dict(
            path=path,
            fragment=dict(
                type="object",
                required=selected.get("required", []),
                additionalProperties=selected.get("additionalProperties"),
                declared_properties=list(props),
                field=field,
                field_schema=props.get(field),
            ),
        )
    return dict(
        path=path, path_resolved=selected is not None, fragment=clone(schema if selected is None else selected)
    )


def _schema_evidence(name, source, schema, args, message):
    error = _attempt(validate, schema, args).get("error")
    return dict(
        name=name,
        source=source,
        valid=error is None,
        error=error,
        parameters_sha256=digest(schema),
        **_schema_fragment(schema, error["message"] if error else message),
    )


def _schemas(registry, item, traced):
    before = item["before"]
    tools = {v["name"]: v for v in modelTools(registry, before["semantic"], recoveryExchanges(before))}
    trace = traced["trace"]
    index = trace["call_index"]
    model_calls = traced["model_calls"] or []
    chosen = enumerate(model_calls) if index is None else [(index, model_calls[index])]
    err = trace.get("error") or {}
    message = err.get("message")
    result = []
    for i, call in chosen:
        tool = tools.get(call["name"])
        if tool:
            result.append(
                dict(
                    call_index=i,
                    attributed=index is not None,
                    **_schema_evidence(
                        call["name"], "dynamic_model_tool", tool["parameters"], call["arguments"], message
                    ),
                )
            )
    domain = err.get("details", {}).get("call")
    if domain is None and index is not None:
        call = model_calls[index]
        args = call["arguments"]
        if registry.tool(call["name"]) and not args.get("_clarify") and not args.get("_resolve"):
            metadata = {"_clarify", "_missing", "_each", "_replace", "_resolve", "_supersede"}
            domain = dict(name=call["name"], arguments={k: v for k, v in args.items() if k not in metadata})
    if domain and registry.tool(domain["name"]):
        result.append(
            dict(
                call_index=index,
                **_schema_evidence(
                    domain["name"],
                    "domain_contract",
                    registry.tool(domain["name"])["parameters"],
                    domain["arguments"],
                    message,
                ),
            )
        )
    return result


def _binding(registry, contracts, domain):
    require(domain and registry.tool(domain["name"]), "Missing exact domain call for contract failure")
    require(
        "error" not in _attempt(registry.validateCall, domain),
        "Unbound contract call unexpectedly fails schema validation",
    )
    same = [b for b in contracts["bindings"] if b["call"]["name"] == domain["name"]]
    args = domain["arguments"]
    subjects = [
        b
        for b in same
        if "subject" in args
        and "subject" in b["call"]["arguments"]
        and canonical(b["call"]["arguments"]["subject"]) == canonical(args["subject"])
    ]
    require(
        all(canonical(b["call"]) != canonical(domain) for b in same), "Exact missing binding unexpectedly exists"
    )

    def sample(items):
        return [
            dict(call=clone(b["call"]), knowledge_ids=clone(b.get("knowledge_ids", [b["knowledge_id"]])))
            for b in items[:3]
        ]

    return dict(
        domain_call=clone(domain),
        schema_valid=True,
        exact_binding_present=False,
        same_function_count=len(same),
        same_function_subject_count=len(subjects),
        same_function_examples=sample(same),
        same_function_subject_examples=sample(subjects),
        interpretation="Schema-valid call has no exact frozen binding; model error versus missing implementation is not adjudicated.",
    )


def _units(s):
    return s.encode("utf-16-le", "surrogatepass")


def _summary(value):
    if isinstance(value, str):
        units = _units(value)
        return value if len(units) <= 480 else units[:480].decode("utf-16-le", "surrogatepass") + "…"
    if isinstance(value, list):
        return dict(kind="array", length=len(value), sha256=digest(value))
    if isinstance(value, dict):
        return dict(kind="object", keys=list(value), sha256=digest(value))
    return value


def _differences(actual, expected, limit=8):
    entries = []
    truncated = False

    def walk(a, b, path):
        nonlocal truncated
        if canonical(a) == canonical(b):
            return
        if len(entries) >= limit:
            truncated = True
            return
        if (isinstance(a, list) and isinstance(b, list)) or (isinstance(a, dict) and isinstance(b, dict)):
            if isinstance(a, list):
                a, b = dict(enumerate(a)), dict(enumerate(b))
            for key in dict.fromkeys([*a, *b]):
                inner = f"{path}.{key}"
                if key not in a or key not in b:
                    if len(entries) >= limit:
                        truncated = True
                        break
                    entries.append(
                        dict(
                            path=inner,
                            actual_present=key in a,
                            expected_present=key in b,
                            actual=_summary(a.get(key)),
                            expected=_summary(b.get(key)),
                        )
                    )
                else:
                    walk(a[key], b[key], inner)
                if truncated:
                    break
        else:
            entries.append(dict(path=path, actual=_summary(a), expected=_summary(b)))

    walk(actual, expected, "$")
    return dict(entries=entries, truncated=truncated, limit=limit)


def _text_comparison(actual, expected):
    a, b = _units(actual), _units(expected)
    n, m = len(a) // 2, len(b) // 2
    first = next((i for i in range(min(n, m)) if a[2 * i : 2 * i + 2] != b[2 * i : 2 * i + 2]), min(n, m))
    start, end = max(0, first - 80), first + 320
    return dict(
        equal=a == b,
        actual_sha256=digest(actual),
        expected_sha256=digest(expected),
        actual_chars=n,
        expected_chars=m,
        first_difference=None if a == b else first,
        excerpt_start=start,
        actual_excerpt_end=min(n, end),
        expected_excerpt_end=min(m, end),
        actual_excerpt=a[2 * start : 2 * end].decode("utf-16-le", "surrogatepass"),
        expected_excerpt=b[2 * start : 2 * end].decode("utf-16-le", "surrogatepass"),
    )


def _review(registry, item, full):
    actual = full["after"]
    text, state = actual["exchanges"][-1]["assistant"], actual["semantic"]
    alternatives = []
    for index, batch in enumerate(item["allowed_batches"]):
        expected = applyModelOutput(registry, item["before"], item["question"], encodeCalls(batch))
        exchange = expected["exchanges"][-1]
        require(exchange["outcome"] == "committed", "Frozen allowed program fails execution")
        reply, sem = exchange["assistant"], expected["semantic"]
        alternatives.append(
            dict(
                index=index,
                calls=clone(batch),
                reference_targets=refs(batch),
                expected=dict(delivered=reply, semantic=clone(sem)),
                delivered=_text_comparison(text, reply),
                semantic=dict(
                    equal=canonical(state) == canonical(sem),
                    actual_sha256=digest(state),
                    expected_sha256=digest(sem),
                    **_differences(state, sem),
                ),
            )
        )
    result = full.get("semantic_result") or {}
    return dict(
        adjudicated=False,
        status=full["critical_status"],
        actual=dict(delivered=text, semantic=clone(state)),
        failure_category=result.get("failure_category"),
        selected_reference_targets=result.get("selected_reference_targets", []),
        alternatives=alternatives,
    )


def createCaseInspector(registry, contracts):
    diagnostic = createTrainDiagnostics(registry)

    def inspect(item, observation):
        full = diagnostic.expand(item, observation)
        require(full["semantic_eligible"] and full["after"], "Inspection needs a reproducible before/after state")
        require(
            full["error_kind"] in ("contract_not_implemented", "invalid_arguments")
            or full["critical_status"] == "review_required",
            "Case is outside the requested inspection groups",
        )
        traced = _trace(registry, item, full)
        output = dict(
            case_id=item["case_id"],
            case_sha256=item["case_sha256"],
            error_kind=full["error_kind"],
            critical_status=full["critical_status"],
            receipt_verified=True,
            scores_changed=False,
            **traced,
            schemas=_schemas(registry, item, traced),
        )
        if full["error_kind"] == "contract_not_implemented":
            output["contract"] = _binding(registry, contracts, traced["trace"]["error"]["details"].get("call"))
        if full["critical_status"] == "review_required":
            output["review"] = _review(registry, item, full)
        return output

    return inspect


def inspectCase(registry, contracts, item, observation):
    return createCaseInspector(registry, contracts)(item, observation)


def main(path):
    reg, bundle = registry_at(path)
    inspect = createCaseInspector(reg, bundle["contracts"])
    for line in sys.stdin:
        payload = json.loads(line)
        require(payload["action"] == "inspect", "Unknown inspection action")
        print(
            js_json(dict(kind="inspection", inspection=inspect(payload["case"], payload["observation"]))),
            flush=True,
        )


if __name__ == "__main__":
    main(sys.argv[1])
