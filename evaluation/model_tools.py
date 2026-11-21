"""Concrete typed model tools available in the current dialogue."""

from .dialogue import declineRequestTool, recoveryTools
from .schema import assertSchema, clone, ContractError, freeze, validate

_META = ("_clarify", "_missing", "_each", "_replace", "_resolve", "_supersede")


def _obj(properties, required=()):
    return dict(type="object", properties=properties, required=list(required), additionalProperties=False)


def _targets(ids, desc):
    return dict(type="string", enum=ids, description=desc)


def modelTools(registry, state, exchanges=None):
    answers = [a for t in state["turns"] for a in t["answers"]]
    cache = getattr(registry, "_model_tool_cache", None)
    if cache is not None:
        key = (
            tuple((a["id"], a["call"]["name"]) for a in answers),
            tuple((p["id"], p["function"]) for p in state["pending"]),
            None if exchanges is None else tuple((e["id"], e["outcome"]) for e in exchanges),
        )
        with registry._cache_lock:
            if key in cache:
                cache.move_to_end(key)
                return cache[key]
    ids = [a["id"] for a in answers]
    pending = [a["id"] for a in state["pending"]]
    result = [clone(declineRequestTool)]
    for domain in registry.tools:
        name = domain["name"]
        p = domain["parameters"]
        props = clone(p["properties"])
        if set(props).intersection(_META):
            raise ValueError("Domain argument collides with dialogue metadata.")
        keys = list(props)
        props["_clarify"] = dict(
            type="boolean", const=True, description="Ask for missing arguments instead of answering."
        )
        props["_missing"] = dict(
            type="array",
            items=dict(type="string", enum=keys),
            minItems=1,
            uniqueItems=True,
            description="Missing optional arguments, if needed; required missing arguments are retained automatically.",
        )
        scalars = [k for k in keys if props[k].get("type") not in ("object", "array")]
        if scalars:
            props["_each"] = _targets(
                scalars, "Clarify an unknown group, then perform one call per member for this argument."
            )
        if ids:
            props["_replace"] = _targets(ids, "Address an actual answer to replace with this operation.")
        matching = [v["id"] for v in state["pending"] if v["function"] == name]
        if matching:
            props["_resolve"] = _targets(matching, "Supply missing arguments of this actual pending request.")
        if pending:
            props["_supersede"] = _targets(pending, "Explicitly change this pending request to this operation.")
        required = ", ".join(p.get("required", [])) or "no mandatory arguments"
        result.append(
            dict(
                name=name,
                description=f"{domain['description']} Complete answers require: {required}. Omission is permitted only for explicit clarification or pending completion.",
                parameters=dict(clone(p), properties=props, required=[]),
            )
        )
        targets = [a["id"] for a in answers if a["call"]["name"] == name]
        if targets:
            result.append(
                dict(
                    name="change_" + name,
                    description=f"Change only named arguments of an actual {name} answer. Retain all other arguments and conditions.",
                    parameters=_obj(
                        dict(
                            target=dict(type="string", enum=targets),
                            changes=dict(clone(p), required=[]),
                            remove=dict(type="array", items=dict(type="string", enum=keys), uniqueItems=True),
                        ),
                        ["target", "changes"],
                    ),
                )
            )
    for name, targets, desc in [
        ("repeat_answer", ids, "Repeat exactly one actual answer."),
        ("cancel_pending", pending, "Cancel only this actual pending request."),
    ]:
        if targets:
            result.append(
                dict(
                    name=name,
                    description=desc,
                    parameters=_obj(dict(target=dict(type="string", enum=targets)), ["target"]),
                )
            )
    result.extend(recoveryTools(exchanges))
    if len({t["name"] for t in result}) != len(result):
        raise ValueError("Domain and dialogue function names collide.")
    for t in result:
        assertSchema(t["parameters"])
    result = freeze(result)
    if cache is not None:
        with registry._cache_lock:
            cache[key] = result
            if len(cache) > 32:
                cache.popitem(last=False)
    return result


def compileModelCalls(registry, state, input, exchanges=None):
    if not isinstance(input, list) or not input:
        raise ContractError("invalid_calls", "At least one model call is required.")
    exposed = {t["name"]: t for t in modelTools(registry, state, exchanges)}
    controls = {
        "decline_request",
        "repeat_answer",
        "cancel_pending",
        *(t["name"] for t in recoveryTools(exchanges)),
    }

    def compile_call(call):
        validate(
            _obj(dict(name=dict(type="string", minLength=1), arguments=dict(type="object")), ["name", "arguments"]),
            call,
        )
        name, args = call["name"], call["arguments"]
        if name not in exposed:
            raise ContractError(
                "unknown_function", "The model selected a function not declared for this conversation."
            )
        validate(exposed[name]["parameters"], args)
        if name in controls:
            return call
        if registry.tool(name) is None:
            if not name.startswith("change_"):
                raise ContractError("unknown_function", "Unknown dialogue function.")
            return dict(name="revise_call", arguments=args)
        clarify, missing, each, replace, resolve, supersede = [args.get(k) for k in _META]
        args = {k: v for k, v in args.items() if k not in _META}
        if sum(bool(v) for v in (replace, resolve, supersede)) > 1:
            raise ContractError("conflicting_actions", "A call cannot address multiple different actions.")
        if (missing is not None or each) and not clarify:
            raise ContractError("invalid_arguments", "Missing-argument metadata requires explicit clarification.")
        if clarify and resolve:
            raise ContractError(
                "invalid_arguments",
                "Pending completion already retains missing arguments; it cannot also start a separate clarification.",
            )
        if clarify:
            keys = list(
                dict.fromkeys(
                    [k for k in registry.tool(name)["parameters"].get("required", []) if k not in args]
                    + (missing or [])
                )
            )
            if not keys:
                raise ContractError("invalid_arguments", "There is no missing argument to clarify.")
            result = dict(function=name, known=args, missing=keys)
            for k, v in [("each", each), ("revision_target", replace), ("pending_target", supersede)]:
                if v:
                    result[k] = v
            return dict(name="clarify_call", arguments=result)
        if resolve:
            return dict(name="complete_call", arguments=dict(target=resolve, values=args))
        replacement = dict(name=name, arguments=args)
        registry.validateCall(replacement)
        if replace or supersede:
            return dict(
                name="replace_call" if replace else "replace_pending",
                arguments=dict(target=replace or supersede, replacement=replacement),
            )
        return replacement

    return [compile_call(c) for c in clone(input)]
