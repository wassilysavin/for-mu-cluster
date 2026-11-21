"""Validate and resolve the frozen scientific function contracts."""

from collections import OrderedDict
from threading import RLock
from hashlib import sha256
from .schema import assertSchema, canonical, clone, ContractError, freeze, validate

callShape = {
    "type": "object",
    "additionalProperties": False,
    "required": ["name", "arguments"],
    "properties": {"name": {"type": "string", "minLength": 1}, "arguments": {"type": "object"}},
}

deliveryShape = {
    "type": "object",
    "additionalProperties": False,
    "required": ["source_segments", "display_order", "evidence"],
    "properties": {
        "source_segments": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        "display_order": {"type": "array", "minItems": 1, "uniqueItems": True, "items": {"type": "integer"}},
        "before_text": {"type": "string"},
        "after_text": {"type": "string"},
        "additional_sources": {
            "type": "array",
            "items": {
                "anyOf": [
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["commit", "path", "line_start", "line_end", "url"],
                        "properties": {
                            "commit": {"type": "string", "minLength": 1},
                            "path": {"type": "string", "minLength": 1},
                            "line_start": {"type": "integer"},
                            "line_end": {"type": "integer"},
                            "url": {"type": "string", "minLength": 1},
                        },
                    },
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["kind", "title", "version", "url"],
                        "properties": {
                            "kind": {"type": "string", "const": "documentation"},
                            "title": {"type": "string", "minLength": 1},
                            "version": {"type": "string", "minLength": 1},
                            "url": {"type": "string", "minLength": 1},
                        },
                    },
                ]
            },
        },
        "evidence": {"type": "object"},
        "evidence_qualification": {
            "type": "object",
            "additionalProperties": False,
            "required": ["status", "reason"],
            "properties": {
                "status": {"type": "string", "const": "insufficient_evidence"},
                "reason": {"type": "string", "minLength": 1},
            },
        },
    },
}


def _unique(values):
    return list({canonical(v): v for v in values}.values())


class Registry:
    def __init__(self, knowledge, contracts):
        self.knowledge = freeze(clone(knowledge))
        self.contracts = freeze(clone(contracts))
        knowledge, contracts = self.knowledge, self.contracts
        if knowledge.get("schema_version") != 1 or contracts.get("schema_version") != 1:
            raise ValueError("Unsupported registry version.")
        records = {}
        for record in knowledge["records"]:
            if record["id"] in records:
                raise ValueError(f"Duplicate knowledge record {record['id']}")
            if not isinstance(record.get("answer", {}).get("text"), str):
                raise ValueError(f"Missing source answer {record['id']}")
            records[record["id"]] = record
        self._tools = {}
        for tool in contracts["tools"]:
            if tool["name"] in self._tools:
                raise ValueError(f"Duplicate function {tool['name']}")
            assertSchema(tool["parameters"])
            self._tools[tool["name"]] = tool
        self.tools = freeze(list(self._tools.values()))
        self.coverage = contracts.get("coverage")
        self.parameterIds = freeze([p["id"] for p in knowledge["parameters"]])
        parameters = set(self.parameterIds)
        self._resolved = {}
        self._fingerprint = None
        self._model_tool_cache = OrderedDict()
        self._prompt_cache = OrderedDict()
        self._cache_lock = RLock()
        statuses = ["insufficient_evidence", "supported_with_limitations", "supported"]
        for binding in contracts["bindings"]:
            self.validateCall(binding["call"])
            first = binding["knowledge_id"]
            if first not in records:
                raise ValueError(f"Missing source {first}")
            ids = binding.get("knowledge_ids", [first])
            if (
                not isinstance(ids, list)
                or not ids
                or ids[0] != first
                or len(set(ids)) != len(ids)
                or set(ids) - records.keys()
            ):
                raise ValueError(f"Invalid source composition for {first}")
            if len(ids) > 1 and not binding.get("review", {}).get("equivalence_basis"):
                raise ValueError(f"Source composition needs an explicit equivalence justification: {first}")
            focus = binding.get("focus_parameters")
            if not isinstance(focus, list) or len(focus) != len(set(focus)) or set(focus) - parameters:
                raise ValueError(f"Missing or invalid explicit focus contract for {first}")
            key = canonical(binding["call"])
            if key in self._resolved:
                raise ValueError(f"Ambiguous exact call {key}")
            chosen = [records[i] for i in ids]
            answers = [r["answer"] for r in chosen]
            if any(a["status"] not in statuses for a in answers):
                raise ValueError("Unknown source evidence status.")
            answer = dict(
                text="\n\n".join(dict.fromkeys(a["text"] for a in answers)),
                status=min((a["status"] for a in answers), key=statuses.index),
                sources=_unique(s for a in answers for s in a["sources"]),
                limitations=_unique(s for a in answers for s in a["limitations"]),
            )
            record = dict(
                chosen[0],
                source_record_ids=ids,
                meaning_ids=list(dict.fromkeys(r["meaning_id"] for r in chosen)),
                answer=answer,
            )
            entry = dict(record=record, focus=focus)
            if "delivery" in binding:
                delivery = binding["delivery"]
                validate(deliveryShape, delivery)
                order = delivery["display_order"]
                segments = delivery["source_segments"]
                if (
                    "".join(segments) != answer["text"]
                    or len(order) != len(segments)
                    or any(not 0 <= i < len(segments) for i in order)
                ):
                    raise ValueError(f"Delivery must retain every exact source segment once: {first}")
                sources = delivery.get("additional_sources", [])
                if (delivery.get("before_text") or delivery.get("after_text")) and not sources:
                    raise ValueError("Added scientific explanation requires explicit supporting sources.")
                if any(
                    not s["url"].startswith("https://")
                    or (s.get("kind") != "documentation" and not 1 <= s["line_start"] <= s["line_end"])
                    for s in sources
                ):
                    raise ValueError("Invalid additional scientific source.")
                body = "".join(segments[i] for i in order)
                entry["delivery"] = dict(
                    text="\n\n".join(filter(None, [delivery.get("before_text"), body, delivery.get("after_text")])),
                    additional_sources=clone(sources),
                )
                if "evidence_qualification" in delivery:
                    entry["delivery"]["evidence_qualification"] = clone(delivery["evidence_qualification"])
            self._resolved[key] = freeze(entry)

    def fingerprint(self):
        if self._fingerprint is None:
            material = dict(
                version=1,
                contracts=self.contracts,
                parameters=self.knowledge["parameters"],
                records=[{k: r[k] for k in ("id", "meaning_id", "answer")} for r in self.knowledge["records"]],
            )
            self._fingerprint = sha256(canonical(material).encode()).hexdigest()
        return self._fingerprint

    def tool(self, name):
        return self._tools.get(name)

    def validateCall(self, value):
        canonical(value)
        validate(callShape, value)
        tool = self.tool(value["name"])
        if tool is None:
            raise ContractError("unknown_function", f"Function is not implemented: {value['name']}")
        validate(tool["parameters"], value["arguments"])
        return tool

    def resolve(self, value):
        self.validateCall(value)
        entry = self._resolved.get(canonical(value))
        if entry is None:
            raise ContractError(
                "contract_not_implemented",
                "This exact call has no implemented source contract.",
                {"call": clone(value)},
            )
        return freeze(dict(call=clone(value), **entry))


def createRegistry(knowledgeInput, contractInput):
    return Registry(knowledgeInput, contractInput)
