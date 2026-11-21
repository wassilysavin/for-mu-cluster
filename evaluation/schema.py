"""The finite JSON and schema subset used by the model contracts."""

import json
import math
import re
from copy import deepcopy
from functools import lru_cache
from decimal import Decimal

DANGEROUS = frozenset(("__proto__", "prototype", "constructor"))
KEYWORDS = frozenset(
    (
        "type",
        "properties",
        "required",
        "additionalProperties",
        "items",
        "minItems",
        "maxItems",
        "uniqueItems",
        "minLength",
        "maxLength",
        "enum",
        "const",
        "oneOf",
        "anyOf",
        "description",
        "title",
    )
)
TYPES = frozenset(("object", "array", "string", "number", "integer", "boolean", "null"))


class ContractError(ValueError):
    def __init__(self, code, message, details=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = {} if details is None else details


JS_TRIM = "\t\n\v\f\r \xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff"


def is_safe_integer(value):
    return type(value) in (int, float) and abs(value) < 2**53 and (type(value) is int or value.is_integer())


def _number(value, strict):
    if isinstance(value, int) and abs(value) <= 2**53:
        return str(value)
    try:
        value = float(value)
    except OverflowError:
        value = math.inf
    if not math.isfinite(value):
        if not strict:
            return "null"
        raise ContractError("invalid_json", "Only finite JSON values are allowed.")
    if value == 0:
        return "0"
    digits = str(value).lower()
    if 1e-6 <= abs(value) < 1e21:
        digits = format(Decimal(digits), "f")
        return digits.rstrip("0").rstrip(".") if "." in digits else digits
    if "e" in digits:
        mantissa, exponent = digits.split("e")
        return mantissa.removesuffix(".0") + "e" + f"{int(exponent):+d}"
    return digits


_SURROGATE = re.compile("[\ud800-\udfff]")


@lru_cache(maxsize=2048)
def _short_quote(value):
    return _safe_quote(value)


def _safe_quote(value):
    text = json.encoder.encode_basestring(value)
    if text.isascii() or not _SURROGATE.search(text):
        return text
    text = text.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "surrogatepass")
    return _SURROGATE.sub(lambda m: f"\\u{ord(m[0]):04x}", text)


def _quote(value):
    return _short_quote(value) if len(value) < 128 else _safe_quote(value)


def _object_keys(value, ordered):
    if any(not isinstance(key, str) for key in value):
        raise ContractError("invalid_json", "Only finite JSON values are allowed.")
    if ordered:
        if DANGEROUS.intersection(value):
            raise ContractError("invalid_json", "Unsafe object property.")
        if all(key.isascii() for key in value):
            return sorted(value)
        return sorted(value, key=lambda key: key.encode("utf-16-be", "surrogatepass"))
    indices = [
        key
        for key in value
        if key.isascii()
        and key.isdecimal()
        and len(key) <= 10
        and (len(key) == 1 or key[0] != "0")
        and int(key) < 2**32 - 1
    ]
    return sorted(indices, key=int) + [key for key in value if key not in indices]


def _serialize(value, ordered):
    if value is None:
        return "null"
    if isinstance(value, str):
        return _quote(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (float, int)):
        return _number(value, ordered)
    if isinstance(value, list):
        return "[" + ",".join(_serialize(item, ordered) for item in value) + "]"
    if isinstance(value, dict):
        return (
            "{"
            + ",".join(_quote(key) + ":" + _serialize(value[key], ordered) for key in _object_keys(value, ordered))
            + "}"
        )
    raise ContractError("invalid_json", "Only finite JSON values are allowed.")


def canonical(value):
    return _serialize(value, True)


def js_json(value):
    return _serialize(value, False)


def clone(value):
    return json.loads(canonical(value))


class _FrozenDict(dict):
    __slots__ = ()

    def _immutable(self, *args, **kwargs):
        raise TypeError("Frozen JSON object")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _immutable

    def __deepcopy__(self, memo):
        return {key: deepcopy(item, memo) for key, item in self.items()}


class _FrozenList(list):
    __slots__ = ()

    def _immutable(self, *args, **kwargs):
        raise TypeError("Frozen JSON array")

    __setitem__ = __delitem__ = append = clear = extend = insert = pop = remove = reverse = sort = __iadd__ = (
        __imul__
    ) = _immutable

    def __deepcopy__(self, memo):
        return [deepcopy(item, memo) for item in self]


def freeze(value):
    if isinstance(value, (_FrozenDict, _FrozenList)):
        return value
    if isinstance(value, dict):
        return _FrozenDict((key, freeze(item)) for key, item in value.items())
    if isinstance(value, list):
        return _FrozenList(freeze(item) for item in value)
    return value


def assertSchema(schema):
    if not isinstance(schema, dict):
        raise ValueError("Invalid schema.")
    for key in schema:
        if key not in KEYWORDS:
            raise ValueError(f"Unsupported schema keyword: {key}")
    if schema.get("type") and schema["type"] not in TYPES:
        raise ValueError(f"Unsupported type: {schema['type']}")
    if schema.get("type") == "object" and schema.get("additionalProperties") is not False:
        raise ValueError("Every contract object must explicitly reject extra properties.")
    for sub in schema.get("properties", {}).values():
        assertSchema(sub)
    if "items" in schema:
        assertSchema(schema["items"])
    for key in ("oneOf", "anyOf"):
        for sub in schema.get(key, []):
            assertSchema(sub)


def validate(schema, value, path="$"):
    def fail(message, details=None):
        raise ContractError("invalid_arguments", f"{path}: {message}", details)

    if "const" in schema and canonical(schema["const"]) != canonical(value):
        fail("unexpected constant")
    if "enum" in schema and not any(canonical(x) == canonical(value) for x in schema["enum"]):
        fail("value is not in the declared domain")
    for key in ("oneOf", "anyOf"):
        if key in schema:
            matches = 0
            for sub in schema[key]:
                try:
                    validate(sub, value, path)
                    matches += 1
                except ContractError:
                    pass
            if (key == "oneOf" and matches != 1) or not matches:
                details = (
                    {"clarification": schema["description"]} if isinstance(schema.get("description"), str) else {}
                )
                fail(f"does not satisfy {key}", details)
    kind = next(
        (
            name
            for cls, name in (
                (dict, "object"),
                (list, "array"),
                (bool, "boolean"),
                (str, "string"),
                (int, "number"),
                (float, "number"),
            )
            if isinstance(value, cls)
        ),
        "null",
    )
    typ = schema.get("type")
    if typ == "integer":
        if kind != "number" or not float(value).is_integer():
            fail("expected integer")
    elif typ and typ != kind:
        fail(f"expected {typ}")
    if kind == "object":
        for key in schema.get("required", []):
            if key not in value:
                fail(f"missing {key}")
        for key, item in value.items():
            if key in DANGEROUS:
                fail("unsafe property")
            if key in schema.get("properties", {}):
                validate(schema["properties"][key], item, f"{path}.{key}")
            elif schema.get("additionalProperties") is False:
                fail(f"unexpected {key}")
    if kind == "array":
        if len(value) < schema.get("minItems", 0):
            fail("too few items")
        if len(value) > schema.get("maxItems", math.inf):
            fail("too many items")
        if schema.get("uniqueItems") and len({canonical(x) for x in value}) != len(value):
            fail("duplicate items")
        if "items" in schema:
            for i, item in enumerate(value):
                validate(schema["items"], item, f"{path}[{i}]")
    if kind == "string":
        if _SURROGATE.search(value):
            value = value.encode("utf-16-le", "surrogatepass").decode("utf-16-le", "surrogatepass")
        if len(value) < schema.get("minLength", 0):
            fail("empty or too short")
        if len(value) > schema.get("maxLength", math.inf):
            fail("too long")
    if kind == "number" and not math.isfinite(value):
        fail("non-finite number")
