"""Apply a complete model-selected batch as a single dialogue transaction."""

from .schema import JS_TRIM, canonical, clone, ContractError, freeze, validate

declineRequestMessage = (
    "I can help with GEM-pRF parameter questions, but that part of the request is outside this assistant's scope."
)

dialogueTools = [
    {
        "name": "decline_request",
        "description": "Decline a request outside GEM-pRF parameter assistance. For a mixed request, also issue "
        "the supported calls in their requested order. Does not cancel pending requests. Do not "
        "use for missing source evidence, incomplete arguments or implementation errors.",
        "parameters": {"type": "object", "properties": {}, "required": [], "additionalProperties": False},
    },
    {
        "name": "repeat_answer",
        "description": "Repeat the full answer for an actual completed call.",
        "parameters": {
            "type": "object",
            "properties": {"target": {"type": "string", "minLength": 1}},
            "required": ["target"],
            "additionalProperties": False,
        },
    },
    {
        "name": "revise_call",
        "description": "Change named arguments of an actual completed call, retaining all other arguments. "
        "Re-execute atomically.",
        "parameters": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "minLength": 1},
                "changes": {"type": "object"},
                "remove": {"type": "array", "items": {"type": "string", "minLength": 1}, "uniqueItems": True},
            },
            "required": ["target", "changes"],
            "additionalProperties": False,
        },
    },
    {
        "name": "replace_call",
        "description": "Replace the selected completed operation with a fully specified new call.",
        "parameters": {
            "type": "object",
            "properties": {"target": {"type": "string", "minLength": 1}, "replacement": {"type": "object"}},
            "required": ["target", "replacement"],
            "additionalProperties": False,
        },
    },
    {
        "name": "clarify_call",
        "description": "Retain a partially specified domain call and ask for missing arguments.",
        "parameters": {
            "type": "object",
            "properties": {
                "function": {"type": "string", "minLength": 1},
                "known": {"type": "object"},
                "missing": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "minItems": 1,
                    "uniqueItems": True,
                },
                "each": {"type": "string", "minLength": 1},
                "revision_target": {"type": "string", "minLength": 1},
                "pending_target": {"type": "string", "minLength": 1},
            },
            "required": ["function", "known", "missing"],
            "additionalProperties": False,
        },
    },
    {
        "name": "complete_call",
        "description": "Fill missing arguments of an actual pending call, retaining already known arguments.",
        "parameters": {
            "type": "object",
            "properties": {"target": {"type": "string", "minLength": 1}, "values": {"type": "object"}},
            "required": ["target", "values"],
            "additionalProperties": False,
        },
    },
    {
        "name": "replace_pending",
        "description": "Resolve an actual pending request with an explicitly changed complete domain call.",
        "parameters": {
            "type": "object",
            "properties": {"target": {"type": "string", "minLength": 1}, "replacement": {"type": "object"}},
            "required": ["target", "replacement"],
            "additionalProperties": False,
        },
    },
    {
        "name": "cancel_pending",
        "description": "Cancel a specifically addressed pending request.",
        "parameters": {
            "type": "object",
            "properties": {"target": {"type": "string", "minLength": 1}},
            "required": ["target"],
            "additionalProperties": False,
        },
    },
    {
        "name": "clarify_subjects",
        "description": "Ask which parameter(s) an identified function applies to. Mode each preserves an "
        "unresolved group and explicitly requests one call per member.",
        "parameters": {
            "type": "object",
            "properties": {
                "function": {"type": "string", "minLength": 1},
                "mode": {"type": "string", "enum": ["one", "each"]},
            },
            "required": ["function", "mode"],
            "additionalProperties": False,
        },
    },
    {
        "name": "resolve_subjects",
        "description": "Supply the exact members for one actual pending subject question. Retains other pending "
        "questions.",
        "parameters": {
            "type": "object",
            "properties": {
                "target": {"type": "string", "minLength": 1},
                "subjects": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "minItems": 1,
                    "uniqueItems": True,
                },
            },
            "required": ["target", "subjects"],
            "additionalProperties": False,
        },
    },
]

callShape = {
    "type": "object",
    "properties": {"name": {"type": "string", "minLength": 1}, "arguments": {"type": "object"}},
    "required": ["name", "arguments"],
    "additionalProperties": False,
}

referenceQuestions = {
    "answer": "Which earlier answer do you mean? If none matches, please restate your question.",
    "exchange": "Which earlier request do you mean? If none matches, please restate it.",
    "pending": "Which pending request do you mean? If none matches, please restate what you want to resolve.",
}


dialogueTools = freeze(dialogueTools)
declineRequestTool = dialogueTools[0]
_DIALOGUE = {t["name"]: t for t in dialogueTools}
_RECOVERY_DESCS = {
    "repeat_exchange": "Repeat the entire displayed reply to this actual exchange verbatim, including an error or clarification. Do not try to answer the old request.",
    "retry_exchange": "Start a batch that answers this actual failed exchange again. Follow with the fresh typed domain calls for its original request and the current corrections. Do not modify previous answers or pending items.",
    "clarify_reference": "Ask the user to identify or restate the intended prior answer, exchange or pending request when it cannot be safely addressed.",
}
_LABELS = dict(
    subject="parameter",
    left="first item",
    right="second item",
    condition_subject="condition parameter",
    condition_value="whether the condition is enabled or disabled",
    direction="increase or decrease",
    value="value",
    component="component",
)


def recoveryTools(exchanges=None):
    if exchanges is None:
        return []
    result = []
    for name, key, ids in [
        ("repeat_exchange", "target", [e["id"] for e in exchanges]),
        ("retry_exchange", "target", [e["id"] for e in exchanges if e["outcome"] in ("error", "provider_error")]),
        ("clarify_reference", "kind", list(referenceQuestions)),
    ]:
        if ids:
            result.append(
                dict(
                    name=name,
                    description=_RECOVERY_DESCS[name],
                    parameters=dict(
                        type="object",
                        properties={key: dict(type="string", enum=ids)},
                        required=[key],
                        additionalProperties=False,
                    ),
                )
            )
    return result


def createState():
    return freeze(dict(version=1, revision=0, turns=[], pending=[], focus=[]))


def _answer(state, id):
    for turn in state["turns"]:
        for answer in turn["answers"]:
            if answer["id"] == id:
                return answer
    raise ContractError("unknown_reference", f"No actual completed call has ID {id}.")


def _make_answer(registry, call, id, extra):
    entry = registry.resolve(call)
    record = entry["record"]
    a = record["answer"]
    delivery = entry.get("delivery", {})
    qual = delivery.get("evidence_qualification")
    sources = list({canonical(s): s for s in [*a["sources"], *delivery.get("additional_sources", [])]}.values())
    result = dict(
        id=id,
        call=clone(call),
        knowledge_id=record["id"],
        meaning_id=record["meaning_id"],
        status=qual["status"] if qual else a["status"],
    )
    if qual:
        result.update(source_status=a["status"], evidence_qualification=clone(qual))
    result.update(
        knowledge_ids=clone(record["source_record_ids"]), meaning_ids=clone(record["meaning_ids"]), text=a["text"]
    )
    if delivery:
        result["display_text"] = delivery["text"]
    result.update(
        sources=clone(sources),
        limitations=[*a["limitations"], *([qual["reason"]] if qual else [])],
        focus=clone(entry["focus"]),
    )
    result.update(extra)
    return result


def renderAnswer(answer):
    text = answer.get("display_text", answer["text"])
    extra = [item for item in answer["limitations"] if item not in text]
    return text + ("\n\nLimitations: " + " ".join(extra) if extra else "")


def _revision(item):
    return dict(revises=item["revision_target"]) if item.get("revision_target") else {}


class _Turn:
    def __init__(self, registry, state, calls, exchanges):
        self.registry, self.state, self.calls, self.exchanges = registry, state, calls, exchanges
        self.id = f"t{state['revision'] + 1}"
        self.answers, self.parts, self.effects = [], [], []
        self.pending = clone(state["pending"])
        self.touched = set()
        self.groups, self.owners = {}, {}
        self.recovery = {t["name"]: t for t in recoveryTools(exchanges)}

    def append(self, call, **extra):
        a = _make_answer(self.registry, call, f"{self.id}.c{len(self.answers) + 1}", extra)
        self.answers.append(a)
        self.parts.append(dict(kind="answer", id=a["id"], text=renderAnswer(a)))
        return a["id"]

    def effect(self, kind, **fields):
        self.effects.append(dict(kind=kind, **fields))

    def touch(self, target):
        if target in self.touched:
            raise ContractError("conflicting_actions", f"Target {target} is changed more than once in this turn.")
        self.touched.add(target)

    def schema(self, name):
        tool = self.registry.tool(name)
        if tool is None:
            raise ContractError("unknown_function", f"Function is not implemented: {name}")
        return dict(tool["parameters"], required=[])

    def get_pending(self, id):
        for item in self.state["pending"]:
            if item["id"] == id:
                return item
        raise ContractError("unknown_reference", f"No actual pending question has ID {id}.")

    def claim(self, item):
        target = item.get("revision_target")
        if not target or (item["mode"] == "each" and self.owners.get(target) == item["id"]):
            return
        self.touch(target)
        self.owners[target] = item["id"]

    def remove(self, id):
        for index, item in enumerate(self.pending):
            if item["id"] == id:
                self.pending.pop(index)
                return
        raise ContractError("conflicting_actions", f"Pending request {id} was already changed.")

    def pending_id(self):
        return f"{self.id}.p{1 + sum(p['id'].startswith(self.id + '.') for p in self.pending)}"

    def add_pending(self, args, marker=None):
        name, known, missing = (args[k] for k in ("function", "known", "missing"))
        schema = self.schema(name)
        validate(schema, known)
        members = args.get("members", [])
        for m in members:
            validate(callShape, m)
            if m["name"] != name:
                raise ContractError("invalid_arguments", "A pending group member has a different operation.")
            validate(schema, m["arguments"])
        if any(k not in schema["properties"] or k in known for k in missing):
            raise ContractError("invalid_arguments", "Missing arguments must be declared and genuinely absent.")
        if any(
            k not in known and k not in missing and (not members or any(k not in m["arguments"] for m in members))
            for k in self.registry.tool(name)["parameters"].get("required", [])
        ):
            raise ContractError("invalid_arguments", "An omitted required argument must remain pending.")
        each = args.get("each")
        if each and (
            each not in schema["properties"]
            or schema["properties"][each].get("type") in ("object", "array")
            or each in known
            or (any(each not in m["arguments"] for m in members) if members else each not in missing)
        ):
            raise ContractError(
                "invalid_arguments",
                "Distribution requires one explicitly selected scalar slot, supplied separately for each member.",
            )
        revision = args.get("revision_target")
        if revision:
            _answer(self.state, revision)
        id = self.pending_id()
        item = dict(
            id=id, function=name, known=clone(known), missing=clone(missing), mode="each" if each else "one"
        )
        if each:
            item["slot"] = each
        if members:
            item["members"] = clone(members)
        if revision:
            item["revision_target"] = revision
        self.pending.append(item)

        def label(k):
            return _LABELS.get(k, k.replace("_", " "))

        if each and each in missing:
            text = (
                "Which parameters do you mean?"
                if each == "subject"
                else f"Which values do you mean for {label(each)}?"
            )
            other = " and ".join(label(k) for k in missing if k != each)
            if other:
                text += f" Please also specify {other}."
        else:
            text = "Please specify " + " and ".join(map(label, missing)) + "."
        response = dict(kind="clarification", id=id, text=text)
        if marker is None:
            self.parts.append(response)
        else:
            self.parts[next(i for i, v in enumerate(self.parts) if v is marker)] = response
        self.effect("pending_created", target=id, **_revision(item))
        return item

    def complete_group(self, item, values):
        id, slot = item["id"], item["slot"]
        progress = self.groups.get(id)
        if progress is None:
            self.touch(id)
            self.claim(item)
            self.remove(id)
            members = clone(item.get("members", []))
            known = clone(item.get("known", {}))
            required = list(
                dict.fromkeys(
                    [
                        *self.registry.tool(item["function"])["parameters"].get("required", []),
                        *item.get("missing", [slot]),
                        *known,
                        *(k for m in members for k in m["arguments"]),
                        slot,
                    ]
                )
            )
            progress = dict(item=item, known=known, members=members, answered=[], done=set(), required=required)
            self.groups[id] = progress
        members, known, required = progress["members"], progress["known"], progress["required"]
        if not values or any(k not in required or k in known for k in values):
            raise ContractError(
                "invalid_arguments",
                "Group completion must fill missing arguments without changing known common values.",
            )
        validate(self.schema(item["function"]), values)
        if slot in values:
            signature = canonical(values[slot])
            if signature in progress["done"]:
                raise ContractError("invalid_arguments", "A completed group member is repeated.")
            member = next((m for m in members if canonical(m["arguments"][slot]) == signature), None)
            if member is not None:
                changed = [k for k in values if k != slot]
                if not changed or any(k in member["arguments"] for k in changed):
                    raise ContractError(
                        "invalid_arguments", "An existing group member may receive only its missing arguments."
                    )
                member["arguments"].update(values)
            else:
                members.append(dict(name=item["function"], arguments={**known, **values}))
        else:
            if members:
                if any(all(k in m["arguments"] for m in members) for k in values):
                    raise ContractError(
                        "invalid_arguments",
                        "A common completion must fill a missing argument of at least one member.",
                    )
                for member in members:
                    for k, v in values.items():
                        member["arguments"].setdefault(k, v)
            elif progress["answered"]:
                raise ContractError(
                    "invalid_arguments",
                    "All selected members are already complete; there is no common missing argument.",
                )
            known.update(values)
        unfinished = []
        for member in members:
            if any(k not in member["arguments"] for k in required):
                unfinished.append(member)
                continue
            answer_id = self.append(member, resolves=id, **_revision(item))
            progress["answered"].append(answer_id)
            progress["done"].add(canonical(member["arguments"][slot]))
            self.record_revision(item, answer_id)
        progress["members"] = unfinished
        if (unfinished or not progress["answered"]) and "marker" not in progress:
            progress["marker"] = {}
            self.parts.append(progress["marker"])

    def record_revision(self, item, id):
        if item.get("revision_target"):
            self.effect("revision", target=item["revision_target"], replacement=id)

    def finish_groups(self):
        for g in self.groups.values():
            item, members = g["item"], g["members"]
            marker = g.get("marker")
            if members or not g["answered"]:
                missing = [
                    k
                    for k in g["required"]
                    if (any(k not in m["arguments"] for m in members) if members else k not in g["known"])
                ]
                updated = self.add_pending(
                    dict(
                        function=item["function"],
                        known=g["known"],
                        missing=missing,
                        each=item["slot"],
                        members=members,
                        revision_target=item.get("revision_target"),
                    ),
                    marker,
                )
                self.effect("pending_progress", target=item["id"], replacement=updated["id"])
            else:
                if marker is not None:
                    self.parts.pop(next(i for i, v in enumerate(self.parts) if v is marker))
                self.effect("pending_resolved", target=item["id"])

    def apply(self, index, call):
        validate(callShape, call)
        args, name = call["arguments"], call["name"]
        target = args.get("target")
        if name in self.recovery:
            validate(self.recovery[name]["parameters"], args)
            if name == "retry_exchange":
                followers = self.calls[1:]
                for follower in followers:
                    validate(callShape, follower)

                def fresh(v):
                    return self.registry.tool(v["name"]) is not None or (
                        v["name"] == "clarify_call"
                        and not v["arguments"].get("revision_target")
                        and not v["arguments"].get("pending_target")
                    )

                if (
                    index != 0
                    or not any(map(fresh, followers))
                    or any(v["name"] != "decline_request" and not fresh(v) for v in followers)
                ):
                    raise ContractError(
                        "invalid_retry",
                        "A retry must start a batch of fresh domain calls, with at least one supported operation.",
                    )
                self.effect("exchange_retried", target=target)
            elif name == "repeat_exchange":
                previous = next(v for v in self.exchanges if v["id"] == target)
                self.parts.append(dict(kind="replayed_exchange", id=target, text=previous["assistant"]))
                self.effect("exchange_repeated", target=target)
            else:
                self.parts.append(dict(kind="reference_clarification", text=referenceQuestions[args["kind"]]))
                self.effect("reference_clarification", reference=args["kind"])
            return
        if name not in _DIALOGUE:
            self.append(call)
            return
        validate(_DIALOGUE[name]["parameters"], args)
        if name == "decline_request":
            id = f"{self.id}.d{1 + sum(e['kind'] == 'declined_request' for e in self.effects)}"
            self.parts.append(dict(kind="declined_request", id=id, text=declineRequestMessage))
            self.effect("declined_request", id=id, call_index=index, text=declineRequestMessage)
        elif name == "repeat_answer":
            previous = _answer(self.state, target)
            id = f"{self.id}.c{len(self.answers) + 1}"
            fields = (
                "call",
                "knowledge_id",
                "knowledge_ids",
                "meaning_id",
                "meaning_ids",
                "status",
                "text",
                "display_text",
                "source_status",
                "evidence_qualification",
                "sources",
                "limitations",
                "focus",
            )
            repeated = dict(id=id, **{k: clone(previous[k]) for k in fields if k in previous}, repeats=target)
            self.answers.append(repeated)
            self.parts.append(dict(kind="answer", id=id, text=renderAnswer(repeated)))
            self.effect("repeat", target=target, answer=id)
        elif name in ("revise_call", "replace_call"):
            self.touch(target)
            prior = _answer(self.state, target)["call"]
            if name == "revise_call":
                changes, remove = args["changes"], args.get("remove", [])
                if not changes and not remove:
                    raise ContractError("invalid_arguments", "Correction has no changes.")
                validate(self.schema(prior["name"]), changes)
                updated = dict(name=prior["name"], arguments={**prior["arguments"], **changes})
                for k in remove:
                    if k not in prior["arguments"] or k in changes:
                        raise ContractError(
                            "invalid_arguments", "Removed arguments must exist and cannot also be changed."
                        )
                    del updated["arguments"][k]
            else:
                updated = args["replacement"]
            if canonical(updated) == canonical(prior):
                prefix = "Correction" if name == "revise_call" else "Replacement"
                raise ContractError("invalid_arguments", prefix + " does not change the call.")
            self.effect("revision", target=target, replacement=self.append(updated, revises=target))
        elif name == "clarify_call":
            updated = clone(args)
            pending = args.get("pending_target")
            if pending:
                self.touch(pending)
                previous = self.get_pending(pending)
                if previous.get("revision_target"):
                    if updated.get("revision_target") and updated["revision_target"] != previous["revision_target"]:
                        raise ContractError(
                            "conflicting_actions",
                            "A pending correction cannot silently change its original target.",
                        )
                    updated["revision_target"] = previous["revision_target"]
                self.remove(pending)
            if updated.get("revision_target"):
                self.touch(updated["revision_target"])
            added = self.add_pending(updated)
            if pending:
                self.effect("pending_revision", target=pending, replacement=added["id"])
        elif name == "complete_call":
            item = self.get_pending(target)
            values = args["values"]
            if item["mode"] == "each":
                self.complete_group(item, values)
                return
            missing = item.get("missing", [item.get("slot")])
            if not values or any(k not in missing for k in values):
                raise ContractError(
                    "invalid_arguments",
                    "Completion may supply only missing arguments; changing known arguments requires an explicit replacement.",
                )
            validate(self.schema(item["function"]), values)
            combined = {**item.get("known", {}), **values}
            remaining = [k for k in missing if k not in combined]
            self.touch(target)
            self.remove(target)
            self.claim(item)
            if remaining:
                updated = self.add_pending(
                    dict(
                        function=item["function"],
                        known=combined,
                        missing=remaining,
                        revision_target=item.get("revision_target"),
                    )
                )
                self.effect("pending_progress", target=target, replacement=updated["id"])
            else:
                id = self.append(
                    dict(name=item["function"], arguments=combined), resolves=target, **_revision(item)
                )
                self.effect("pending_resolved", target=target)
                self.record_revision(item, id)
        elif name == "replace_pending":
            self.touch(target)
            item = self.get_pending(target)
            self.remove(target)
            self.claim(item)
            id = self.append(args["replacement"], resolves=target, **_revision(item))
            self.effect("pending_replaced", target=target, answer=id)
            self.record_revision(item, id)
        elif name == "cancel_pending":
            self.touch(target)
            self.get_pending(target)
            self.remove(target)
            self.parts.append(dict(kind="cancellation", id=target, text="That pending request has been cancelled."))
            self.effect("pending_cancelled", target=target)
        elif name == "clarify_subjects":
            domain = self.registry.tool(args["function"])
            if (
                domain is None
                or domain["parameters"].get("required") != ["subject"]
                or len(domain["parameters"]["properties"]) != 1
            ):
                raise ContractError(
                    "clarification_not_implemented",
                    "This function needs a different declared clarification contract.",
                )
            id = self.pending_id()
            self.pending.append(dict(id=id, function=args["function"], mode=args["mode"], slot="subject"))
            self.parts.append(
                dict(
                    kind="clarification",
                    id=id,
                    text="Which parameters do you mean?"
                    if args["mode"] == "each"
                    else "Which parameter do you mean?",
                )
            )
            self.effect("pending_created", target=id)
        elif name == "resolve_subjects":
            self.touch(target)
            item = self.get_pending(target)
            if ("missing" in item and item["missing"] != ["subject"]) or (
                item.get("slot") and item["slot"] != "subject"
            ):
                raise ContractError(
                    "invalid_arguments", "This pending request is not a subject-only clarification."
                )
            if item["mode"] == "one" and len(args["subjects"]) != 1:
                raise ContractError("invalid_arguments", "This pending question requires one parameter.")
            self.claim(item)
            for subject in args["subjects"]:
                id = self.append(
                    dict(name=item["function"], arguments=dict(item.get("known", {}), subject=subject)),
                    resolves=target,
                    **_revision(item),
                )
                self.record_revision(item, id)
            self.remove(target)
            self.effect("pending_resolved", target=target)


def executeTurn(registry, state, question, inputCalls, exchanges=None):
    try:
        if (
            isinstance(state.get("version"), bool)
            or state.get("version") != 1
            or isinstance(state.get("revision"), bool)
            or state.get("revision") != len(state["turns"])
        ):
            raise ContractError("invalid_state", "Invalid state revision.")
        if not isinstance(question, str) or not question.strip(JS_TRIM):
            raise ContractError("invalid_question", "A nonempty question is required.")
        canonical(inputCalls)
        if not isinstance(inputCalls, list) or not inputCalls:
            raise ContractError("invalid_calls", "At least one call is required.")
        calls = clone(inputCalls)
        tx = _Turn(registry, state, calls, exchanges)
        for i, call in enumerate(calls):
            tx.apply(i, call)
        tx.finish_groups()
        turn = dict(
            id=tx.id,
            question=question,
            calls=calls,
            answers=tx.answers,
            response_parts=tx.parts,
            text="\n\n".join(p["text"] for p in tx.parts),
            effects=tx.effects,
        )
        focus = (
            list(dict.fromkeys(k for a in tx.answers for k in a["focus"])) if tx.answers else clone(state["focus"])
        )
        new = freeze(
            dict(
                version=1,
                revision=state["revision"] + 1,
                turns=[*state["turns"], turn],
                pending=tx.pending,
                focus=focus,
            )
        )
        return freeze(dict(ok=True, state=new, turn=new["turns"][-1]))
    except ContractError as e:
        return freeze(dict(ok=False, state=state, error=dict(code=e.code, message=e.message, details=e.details)))


def modelHistory(state):
    turns = []
    for turn in state["turns"]:
        calls = []
        for answer in turn["answers"]:
            call = dict(id=answer["id"], **answer["call"])
            for k in ("revises", "resolves", "repeats"):
                if answer.get(k):
                    call[k] = answer[k]
            if answer.get("evidence_qualification"):
                call.update(evidence_status=answer["status"], source_evidence_status=answer["source_status"])
            calls.append(call)
        turns.append(
            dict(
                id=turn["id"],
                user=turn["question"],
                assistant=turn["text"],
                calls=calls,
                actions=turn["effects"],
                clarifications=[p for p in turn["response_parts"] if p["kind"] == "clarification"],
            )
        )
    return clone(dict(turns=turns, pending=state["pending"], focus=state["focus"]))
