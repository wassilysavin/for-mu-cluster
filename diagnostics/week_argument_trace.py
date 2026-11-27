"""Check stored DEV155 argument failures against the original call contracts."""

import json
import re

from eval_tools.io import records
from evaluation.model_tools import _META, modelTools
from evaluation.schema import ContractError, canonical, validate
from evaluation.session import applyModelOutput, recoveryExchanges

from .inspect_week_failures import passed
from .qwen_case_details import _schema_fragment, _trace


_PATH_PART = re.compile(r"\.([A-Za-z_]\w*)|\[(\d+)\]")
_MISSING = object()


def _brief(value):
    if value is _MISSING:
        return "<missing>"
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return rendered if len(rendered) <= 120 else rendered[:119] + "…"


def _domain_call(registry, state, call, internal):
    if internal is None:
        if registry.tool(call["name"]) and not any(call["arguments"].get(key) for key in ("_clarify", "_resolve")):
            return dict(name=call["name"], arguments={k: v for k, v in call["arguments"].items() if k not in _META})
        return None
    name, args = internal["name"], internal["arguments"]
    if name in ("replace_call", "replace_pending"):
        return args["replacement"]
    if name == "revise_call":
        old = next(a["call"] for turn in state["turns"] for a in turn["answers"] if a["id"] == args["target"])
        updated = {**old["arguments"], **args["changes"]}
        for field in args.get("remove", []):
            updated.pop(field, None)
        return dict(name=old["name"], arguments=updated)
    if name == "complete_call":
        pending = next(item for item in state["pending"] if item["id"] == args["target"])
        values = args["values"]
        if pending["mode"] == "one" and set(values) == set(pending.get("missing", [pending.get("slot")])):
            return dict(name=pending["function"], arguments={**pending.get("known", {}), **values})
    return internal if registry.tool(name) else None


def _schema_failure(schema, arguments, error):
    try:
        validate(schema, arguments)
    except ContractError as raised:
        if (raised.code, raised.message) == (error["code"], error["message"]):
            return _schema_fragment(schema, raised.message)
    return None


def _schema_evidence(registry, before, traced):
    trace = traced["trace"]
    index = trace["call_index"]
    if index is None:
        return None
    call = traced["model_calls"][index]
    internal = traced["internal_calls"]
    state = before["semantic"]
    candidates = []
    if trace["stage"] == "compile":
        dynamic = next(tool for tool in modelTools(registry, state, recoveryExchanges(before))
                       if tool["name"] == call["name"])
        candidates.append(("dialogue tool schema", dynamic["parameters"], call["arguments"]))
    domain = _domain_call(registry, state, call, internal[index] if internal else None)
    if domain:
        candidates.append((domain["name"] + " function contract", registry.tool(domain["name"])["parameters"],
                           domain["arguments"]))
    for source, schema, arguments in candidates:
        if failure := _schema_failure(schema, arguments, trace["error"]):
            return source, arguments, failure
    return None


def _format_schema(source, arguments, evidence):
    path, fragment = evidence["path"], evidence["fragment"]
    if field := fragment.get("field"):
        path += "." + field
    value = arguments
    for field, index in _PATH_PART.findall(path):
        value = value.get(field, _MISSING) if field else value[int(index)]
    rule = fragment
    if fragment.get("field"):
        rule = ({"required": True} if fragment["field"] in fragment["required"]
                else {"declared": fragment["field"] in fragment["declared_properties"],
                      "additionalProperties": fragment["additionalProperties"]})
    else:
        rule = {key: rule[key] for key in (
            "type", "enum", "const", "required", "additionalProperties", "minItems",
            "maxItems", "uniqueItems", "minLength", "maxLength", "oneOf", "anyOf"
        ) if key in rule}
        if "enum" in rule and len(rule["enum"]) > 8:
            rule["enum"] = f"{len(rule['enum'])} values in the frozen registry"
    return f"{source} | {path}={_brief(value)} | rule {_brief(rule)}"


def _saved_failure(record):
    if record.get("status") != "observed":
        return None
    exchange = record["after"]["exchanges"][-1]
    return exchange if (exchange.get("outcome"), exchange.get("error_code")) == ("error", "invalid_arguments") else None


def trace_argument(registry, record, exchange):
    before = record["before"]
    question, raw = exchange["question"], exchange["raw_output"]
    after = record["after"]
    if canonical(applyModelOutput(registry, before, question, raw)) != canonical(after):
        raise ValueError(f"Saved exchange does not reproduce for {record['case_id']}")
    traced = _trace(registry, {"before": before, "question": question}, {
        "generation": {"raw_text": raw}, "after": after,
    })
    trace = traced["trace"]
    error = trace.get("error") or {}
    if trace["stage"] not in {"compile", "execute"} or error.get("code") != "invalid_arguments":
        raise ValueError(f"Invalid-arguments verdict does not reproduce for {record['case_id']}")
    index, calls = trace["call_index"], traced["model_calls"]
    if index is None:
        names = ", ".join(call["name"] for call in calls)
        return f"{trace['stage']} | batch ({names}) | {error['message']} | {trace['attribution']}"
    call = calls[index]
    evidence = _schema_evidence(registry, before, traced)
    detail = _format_schema(*evidence) if evidence else f"call arguments {_brief(call['arguments'])}"
    return f"{trace['stage']} | call {index + 1} {call['name']} | {error['message']} | {detail}"


def render_arguments(registry, plan, saved, paths):
    lines = ["DEV155 argument failures on previously passing supported cases"]
    baseline = saved["baseline"]
    for name in ("matched", "full"):
        verdicts = saved[name]
        losses = {cid for cid, row in plan.items()
                  if row["cohort"] == "supported" and passed(baseline[cid]) and not passed(verdicts[cid])}
        details = []
        for path in paths[name]:
            for record in records(path):
                cid = record["case_id"]
                if cid in losses and (exchange := _saved_failure(record)):
                    verdict = verdicts[cid]["critical_status"]
                    details.append(f"  {cid} | grade={verdict} | {trace_argument(registry, record, exchange)}")
        lines.append(f"{name}: {len(details)} of {len(losses)} losses recorded invalid_arguments")
        lines.extend(details)
    return lines
