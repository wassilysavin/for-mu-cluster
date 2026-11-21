"""FunctionGemma's typed function batch transport."""

import math
import re

from .schema import canonical, ContractError, DANGEROUS

START = "<start_function_call>call:"
END = "<end_function_call>"
functionBatchStopToken = "<start_function_response>"
ESC = "<escape>"
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
NUMBER = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?")
SPACE = re.compile(r"[\t-\r \xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000\ufeff]*")


def decodeCalls(output):
    if not isinstance(output, str) or len(output.encode("utf-16-le", "surrogatepass")) > 262144:
        raise ContractError("invalid_model_output", "Invalid function output length.")
    at = 0

    def fail(message, code="invalid_model_output"):
        raise ContractError(code, message, {"offset": len(output[:at].encode("utf-16-le", "surrogatepass")) // 2})

    def whitespace():
        nonlocal at
        at = SPACE.match(output, at).end()

    def consume(token):
        nonlocal at
        if not output.startswith(token, at):
            fail(f"Expected {token}.")
        at += len(token)

    def name():
        nonlocal at
        match = IDENT.match(output, at)
        if not match:
            fail("Expected an argument or function name.")
        at = match.end()
        if match[0] in DANGEROUS:
            fail("Unsafe argument name.")
        return match[0]

    def value(depth=0):
        nonlocal at
        if depth > 24:
            fail("Function arguments are nested too deeply.")
        whitespace()
        if output.startswith(ESC, at):
            at += len(ESC)
            end = output.find(ESC, at)
            if end < 0:
                fail("Unterminated string.", "incomplete_model_output")
            text = output[at:end]
            at = end + len(ESC)
            return text
        if output[at : at + 1] in ("{", "["):
            obj = output[at] == "{"
            close = "}" if obj else "]"
            result = {} if obj else []
            at += 1
            whitespace()
            if output.startswith(close, at):
                at += 1
                return result
            while True:
                if obj:
                    whitespace()
                    key = name()
                    if key in result:
                        fail("Duplicate argument key.")
                    whitespace()
                    consume(":")
                    result[key] = value(depth + 1)
                else:
                    result.append(value(depth + 1))
                whitespace()
                if output.startswith(close, at):
                    at += 1
                    return result
                consume(",")
        for token, val in [("true", True), ("false", False), ("null", None)]:
            if output.startswith(token, at) and not re.match(
                "[A-Za-z0-9_]", output[at + len(token) : at + len(token) + 1]
            ):
                at += len(token)
                return val
        match = NUMBER.match(output, at)
        if match:
            at = match.end()
            num = float(match[0])
            if not math.isfinite(num):
                fail("Non-finite number.")
            return int(num) if num.is_integer() else num
        fail("Expected a typed function argument.")

    calls = []
    whitespace()
    while output.startswith(START, at):
        consume(START)
        function = name()
        args = value()
        if not isinstance(args, dict):
            fail("Arguments must be an object.")
        consume(END)
        calls.append({"name": function, "arguments": args})
        whitespace()
    if not calls:
        fail("The model did not return a function call.")
    if not output.startswith(functionBatchStopToken, at):
        fail("Missing complete-batch marker.", "incomplete_model_output")
    consume(functionBatchStopToken)
    whitespace()
    if at != len(output):
        fail("Unexpected content after the function batch.")
    canonical(calls)
    return calls


def encodeCalls(calls):
    canonical(calls)
    if not isinstance(calls, list) or not calls:
        raise ContractError("invalid_calls", "At least one call is required.")

    def safe(key):
        if not isinstance(key, str) or not IDENT.fullmatch(key) or key in DANGEROUS:
            raise ContractError("invalid_transport_key", "A native function argument name is invalid.")
        return key

    def render(item):
        if isinstance(item, str):
            if ESC in item:
                raise ContractError(
                    "invalid_transport_literal", "The vendor string delimiter cannot occur inside a literal."
                )
            return ESC + item + ESC
        if isinstance(item, list):
            return "[" + ",".join(map(render, item)) + "]"
        if isinstance(item, dict):
            return "{" + ",".join(safe(k) + ":" + render(item[k]) for k in sorted(item)) + "}"
        return canonical(item)

    output = []
    for call in calls:
        if (
            not isinstance(call, dict)
            or set(call) != {"name", "arguments"}
            or not isinstance(call["arguments"], dict)
        ):
            raise ContractError("invalid_calls", "Expected a named call with an argument object.")
        output.append(START + safe(call["name"]) + render(call["arguments"]) + END)
    return "".join(output) + functionBatchStopToken
