"""Pair the five admissible data-matching parameters and grade wrong quotations."""

import argparse
import json
import sys
from copy import deepcopy

from .common import check, read_json

FINGERPRINT = "14f523a431292bcae2a50bbc692974a7d7dfec0251bdb305e0ae8ac17b9046a0"
NAME = "explain_parameter_data_matching"


def paired_schema(source):
    check(source["fingerprint"] == FINGERPRINT)
    pairs = [row["call"]["arguments"] for row in source["contracts"]["bindings"] if row["call"]["name"] == NAME]
    check(len(pairs) == 5)
    return [{"properties": {k: {"const": pair[k]} for k in ("subject", "reference")}} for pair in pairs]


def transferRegistry(source):
    from evaluation.registry import createRegistry

    original = createRegistry(source["knowledge"], source["contracts"])
    check(original.fingerprint() == FINGERPRINT, "Unexpected original registry")
    contracts = deepcopy(source["contracts"])
    tool = next(tool for tool in contracts["tools"] if tool["name"] == NAME)
    tool["parameters"]["anyOf"] = paired_schema(source)
    return createRegistry(source["knowledge"], contracts)


def upgradePacked(packed, source):
    from evaluation.prompt import renderToolContract, unpackTools
    from evaluation.schema import clone

    tools = unpackTools(packed)
    domain = next(tool for tool in tools if tool["name"] == NAME)
    check("anyOf" not in domain["parameters"], "Expected original data-matching tool")
    original = next(tool for tool in source["contracts"]["tools"] if tool["name"] == NAME)
    revised = clone({**original["parameters"], "anyOf": paired_schema(source)})
    domain["parameters"] = {**revised, "properties": domain["parameters"]["properties"], "required": []}
    for tool in tools:
        if tool["name"] == "change_" + NAME:
            tool["parameters"]["properties"]["changes"] = {**revised, "required": []}
    return renderToolContract(tools)


def gradeTransfer(frozenGrade, resolved, before, after):
    if (
        frozenGrade["success"]
        or frozenGrade.get("failure_category") != "semantic_unresolved"
        or resolved["branch"] != "absent"
        or resolved["reference_targets"]
    ):
        return frozenGrade
    batches = resolved["allowed_batches"]
    clarification = batches and all(
        len(batch) == 1
        and batch[0]["name"] == "clarify_reference"
        and batch[0]["arguments"].get("kind") == "answer"
        for batch in batches
    )
    exchange = after["exchanges"][-1]
    displayed = {answer["id"] for turn in before["semantic"]["turns"] for answer in turn["answers"]}
    wrong = exchange["outcome"] == "committed" and any(
        call["name"] == "repeat_answer" and call["arguments"].get("target") in displayed
        for call in exchange.get("model_calls") or []
    )
    if clarification and wrong:
        return {
            **frozenGrade,
            "critical_status": "detected",
            "critical_category": "wrong_reference",
            "failure_category": "wrong_reference",
        }
    return frozenGrade


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--upgrade-packed", action="store_true", required=True)
    parser.parse_args()
    sys.stdout.write(upgradePacked(json.load(sys.stdin), read_json("registry.json")))
