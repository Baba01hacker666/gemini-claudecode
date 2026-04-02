from typing import Any

# ─── Edge Case Hardened Schema Sanitizer ──────────────────────────────────────
_GEMINI_UNSUPPORTED_KEYS = {
    "$schema", "$defs", "$ref", "default", "examples", "title",
    "additionalProperties", "unevaluatedProperties",
    "if", "then", "else", "not",
    "contentEncoding", "contentMediaType",
    "propertyNames", "exclusiveMinimum", "exclusiveMaximum",
    "minimum", "maximum", "minLength", "maxLength",
    "pattern", "patternProperties", "minItems", "maxItems",
    "uniqueItems", "multipleOf", "minProperties", "maxProperties",
    "const", "format"
}

_GEMINI_ALLOWED_TYPES = {"string", "number", "integer", "boolean", "array", "object"}

def sanitise_schema(schema: Any, depth: int = 0) -> Any:
    if not isinstance(schema, dict):
        return schema

    schema = {k: v for k, v in schema.items() if k not in _GEMINI_UNSUPPORTED_KEYS}

    for kw in ("anyOf", "oneOf"):
        if kw in schema:
            variants = schema.pop(kw)
            if not isinstance(variants, list):
                continue
            non_null = [v for v in variants if not (isinstance(v, dict) and v.get("type") == "null")]
            if non_null:
                schema.update({k: v for k, v in non_null[0].items() if k not in schema})

    if "allOf" in schema:
        for sub in schema.pop("allOf"):
            if isinstance(sub, dict):
                schema.update({k: v for k, v in sub.items() if k not in schema})

    if "type" in schema:
        t = schema["type"]
        if isinstance(t, list):
            non_null = [x for x in t if x != "null"]
            schema["type"] = non_null[0] if non_null else "string"
        if schema.get("type") not in _GEMINI_ALLOWED_TYPES:
            schema["type"] = "string"

    if "properties" in schema:
        schema["properties"] = {
            k: sanitise_schema(v, depth + 1)
            for k, v in schema["properties"].items()
            if isinstance(v, dict)
        }
    elif schema.get("type") == "object":
        schema["properties"] = {}

    if "items" in schema:
        schema["items"] = sanitise_schema(schema["items"], depth + 1)
    elif schema.get("type") == "array":
        schema["items"] = {"type": "string"}

    return schema

def validate_anthropic_request(body: dict) -> None:
    """
    Validates the incoming Anthropic request body.
    Raises ValueError if validation fails.
    """
    if not isinstance(body, dict):
        raise ValueError("Request body must be a JSON object")

    if "model" not in body:
        raise ValueError("Missing 'model' field")

    if "messages" not in body:
        raise ValueError("Missing 'messages' field")

    messages = body["messages"]
    if not isinstance(messages, list):
        raise ValueError("'messages' must be a list")

    if not messages:
        raise ValueError("'messages' list cannot be empty")

    for msg in messages:
        if not isinstance(msg, dict):
            raise ValueError("Each message must be an object")
        if "role" not in msg:
            raise ValueError("Message missing 'role' field")
        if "content" not in msg:
            raise ValueError("Message missing 'content' field")
        if msg["role"] not in ("user", "assistant"):
            raise ValueError(f"Invalid role: {msg['role']}")
