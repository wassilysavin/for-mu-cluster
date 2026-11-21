"""Compact and expand shared tool contracts losslessly."""

import re
from collections import Counter
from .schema import assertSchema, canonical, ContractError, freeze, js_json

PREFIX = " Complete answers require: "

SUFFIX = ". Omission is permitted only for explicit clarification or pending completion."

FORMAT = "gemprf-shared-tool-contract"

VERSION = 1

COMMON = {
    "parameter_object": {"type": "object", "required": [], "additionalProperties": False},
    "description_suffix": {
        "prefix": " Complete answers require: ",
        "join_complete_arguments_with": ", ",
        "if_none": "no mandatory arguments",
        "suffix": ". Omission is permitted only for explicit clarification or pending completion.",
    },
    "dialogue_metadata": {},
    "inheritance_rules": [
        "Only entries without Tool or Raw inherit parameter_object and dialogue_metadata.",
        "ArgumentNames is the listed arguments in their listed order. "
        "GroupableArgumentNames excludes object and array arguments.",
        "Use ArgumentNames for _missing.items.enum; use GroupableArgumentNames for "
        "_each.enum, omitting _each if empty.",
        "An Addressing entry includes that metadata argument with its listed Targets; all "
        "other addressing metadata arguments are absent.",
        "Append description_suffix using Complete; Complete lists the arguments required for a complete answer.",
        "An enum domain reference expands to its entire literal array, preserving order; "
        "reference names are never emitted as argument values.",
    ],
}

READER = (
    "Read these shared JSON function contracts by expanding Common and Domains. Tool and Raw entries are "
    "complete tools without inheritance. Emit ordinary native function calls, never presentation keys or "
    "domain-reference names."
)


ADDRESS = ("_replace", "_resolve", "_supersede")
META = ("_clarify", "_missing", "_each", *ADDRESS)


def _fail(message):
    raise ContractError("invalid_tool_contract", message)


def _copy(value, *_):
    if isinstance(value, dict):
        return {key: _copy(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_copy(val) for val in value]
    return value


def _keys(value, names):
    return isinstance(value, dict) and value.keys() == set(names)


def _walk(schema, mapper, path=()):
    result = _copy(schema)
    if "enum" in schema:
        result["enum"] = mapper(schema["enum"], path)
    if "properties" in schema:
        result["properties"] = {k: _walk(v, mapper, (*path, k)) for k, v in schema["properties"].items()}
    if "items" in schema:
        result["items"] = _walk(schema["items"], mapper, (*path, "items"))
    for key in ("oneOf", "anyOf"):
        if key in schema:
            result[key] = [_walk(v, mapper, path) for v in schema[key]]
    return result


def _supported(tool):
    if not _keys(tool, ("name", "description", "parameters")) or not all(
        isinstance(tool[k], str) for k in ("name", "description")
    ):
        return False

    def enum(vals, _):
        if not isinstance(vals, list):
            raise ValueError("An enum must be an array.")
        return vals

    try:
        assertSchema(tool["parameters"])
        _walk(tool["parameters"], enum)
        return True
    except (TypeError, ValueError, AttributeError):
        return False


def _candidate(tool):
    p = tool["parameters"]
    if (
        not _keys(p, ("type", "properties", "required", "additionalProperties"))
        or p["type"] != "object"
        or p["additionalProperties"] is not False
        or p["required"] != []
        or not isinstance(p["properties"], dict)
    ):
        return None
    props = p["properties"]
    desc = tool["description"]
    if (
        "_clarify" not in props
        or "items" not in props.get("_missing", {})
        or not desc.endswith(SUFFIX)
        or PREFIX not in desc
    ):
        return None
    split = desc.rindex(PREFIX)
    complete = desc[split + len(PREFIX) : -len(SUFFIX)]
    required = [] if complete == "no mandatory arguments" else complete.split(", ")
    args = {k: v for k, v in props.items() if k not in META}
    names = list(args)
    groupable = [k for k in names if args[k].get("type") not in ("object", "array")]
    if any(name not in args for name in required) or props["_missing"]["items"].get("enum") != names:
        return None
    if (groupable and groupable != props.get("_each", {}).get("enum")) or (not groupable and "_each" in props):
        return None
    if any(k in props and not isinstance(props[k].get("enum"), list) for k in ADDRESS):
        return None
    return dict(name=tool["name"], description=desc[:split], Complete=required, arguments=args)


def _restore(row, common, expand):
    args = {k: _walk(schema, expand) for k, schema in row["arguments"].items()}
    names = list(args)
    groupable = [k for k in names if args[k].get("type") not in ("object", "array")]
    meta = common["dialogue_metadata"]
    for key in ("_clarify", "_missing"):
        if key not in meta:
            _fail(f"Missing common {key} definition.")
        args[key] = _copy(meta[key])
    args["_missing"]["items"]["enum"] = names
    if groupable:
        if "_each" not in meta:
            _fail("Missing common _each definition.")
        args["_each"] = dict(_copy(meta["_each"]), enum=groupable)
    for key, targets in row.get("Addressing", {}).items():
        if key not in ADDRESS or key not in meta:
            _fail(f"Unknown addressing metadata {key}.")
        args[key] = dict(_copy(meta[key]), enum=expand(targets))
    suffix = common["description_suffix"]
    complete = suffix["join_complete_arguments_with"].join(row["Complete"]) or suffix["if_none"]
    return dict(
        name=row["name"],
        description=row["description"] + suffix["prefix"] + complete + suffix["suffix"],
        parameters=dict(_copy(common["parameter_object"]), properties=args),
    )


def _map_row(row, mapper):
    if "Raw" in row:
        return _copy(row)
    if "Tool" in row:
        tool = row["Tool"]
        return {"Tool": dict(_copy(tool), parameters=_walk(tool["parameters"], mapper))}
    result = _copy(row)
    result["arguments"] = {k: _walk(v, mapper, (k,)) for k, v in row["arguments"].items()}
    for k, v in row.get("Addressing", {}).items():
        result["Addressing"][k] = mapper(v, (k,))
    return result


def _domain_name(name, path, values):
    arg = path[0] if path else "argument"
    if len(values) == 60 and arg == "subject":
        return "Parameters"
    if arg in ADDRESS[:1]:
        return "AnswerTargets"
    if arg == "_supersede":
        return "PendingTargets"
    words = (
        name + "_pending_targets"
        if arg == "_resolve"
        else re.sub(r"^(explain_|get_|describe_|interpret_|compare_)", "", name) + "_" + arg + "_values"
    )
    return "".join(w.capitalize() for w in re.split("[^a-zA-Z0-9]+", words)) or "Values"


def packTools(input):
    if not isinstance(input, list):
        _fail("Tools must be an array.")
    canonical(input)
    tools = _copy(input)
    supported = [_supported(t) for t in tools]
    candidates = [_candidate(t) if ok else None for t, ok in zip(tools, supported)]
    common = _copy(COMMON)
    meta = common["dialogue_metadata"]
    for key in META:
        prop = next(
            (
                t["parameters"]["properties"][key]
                for c, t in zip(candidates, tools)
                if c and key in t["parameters"]["properties"]
            ),
            None,
        )
        if prop is not None:
            meta[key] = _copy(prop)
            if key == "_missing":
                meta[key]["items"]["enum"] = "ArgumentNames"
            if key == "_each":
                meta[key]["enum"] = "GroupableArgumentNames"
            if key in ADDRESS:
                meta[key]["enum"] = "Targets"
    rows = []
    for tool, ok, row in zip(tools, supported, candidates):
        if not ok:
            rows.append({"Raw": tool})
            continue
        if row:
            props = tool["parameters"]["properties"]
            addressing = {k: props[k]["enum"] for k in ADDRESS if k in props}
            if addressing:
                row["Addressing"] = addressing
            if canonical(_restore(row, common, _copy)) == canonical(tool):
                rows.append(row)
                continue
        rows.append({"Tool": tool})
    proposals = []
    counts = Counter()
    for row in rows:

        def propose(values, path):
            key = canonical(values)
            counts[key] += 1
            proposals.append(
                (key, values, _domain_name(row.get("name", row.get("Tool", {}).get("name", "")), path, values))
            )
            return values

        _map_row(row, propose)
    domains, references = {}, {}
    for key, values, proposal in proposals:
        if key in references or counts[key] < 2:
            continue
        name = proposal
        n = 2
        while name in domains:
            name = proposal + str(n)
            n += 1
        if len(js_json(values).encode("utf-16-le", "surrogatepass")) // 2 <= len(name) + 12:
            continue
        domains[name] = values
        references[key] = name

    def refer(values, path):
        key = canonical(values)
        return {"domain": references[key]} if key in references else values

    packed = dict(
        format=FORMAT,
        version=VERSION,
        Reader=READER,
        Common=common,
        Domains=domains,
        Functions=[_map_row(row, refer) for row in rows],
    )
    if canonical(unpackTools(packed)) != canonical(tools):
        _fail("Tool packing did not round-trip exactly.")
    return freeze(packed)


def unpackTools(input):
    canonical(input)
    packed = _copy(input)
    if (
        not isinstance(packed, dict)
        or packed.get("format") != FORMAT
        or type(packed.get("version")) is not int
        or packed["version"] != VERSION
        or not isinstance(packed.get("Functions"), list)
        or not isinstance(packed.get("Domains"), dict)
        or not isinstance(packed.get("Common"), dict)
    ):
        _fail("Unknown or malformed tool contract format.")

    def expand(values, path=None):
        if isinstance(values, list):
            return _copy(values)
        if (
            not _keys(values, ["domain"])
            or not isinstance(values["domain"], str)
            or not isinstance(packed["Domains"].get(values["domain"]), list)
        ):
            _fail("Invalid enum domain reference.")
        return _copy(packed["Domains"][values["domain"]])

    result = []
    for row in packed["Functions"]:
        if _keys(row, ["Raw"]):
            result.append(_copy(row["Raw"]))
        elif _keys(row, ["Tool"]):
            tool = row["Tool"]
            result.append(dict(_copy(tool), parameters=_walk(tool["parameters"], expand)))
        else:
            keys = ["name", "description", "Complete", "arguments"]
            if isinstance(row, dict) and "Addressing" in row:
                keys.append("Addressing")
            if (
                not _keys(row, keys)
                or not isinstance(row["arguments"], dict)
                or not isinstance(row["Complete"], list)
                or not all(isinstance(row[k], str) for k in ("name", "description"))
            ):
                _fail("Unknown or malformed function entry.")
            result.append(_restore(row, packed["Common"], expand))
    return result


def renderToolContract(tools):
    return js_json(packTools(tools))
