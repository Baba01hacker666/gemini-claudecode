import uuid
import copy
from typing import Any
from app.models import resolve_model
from app.schema import sanitise_schema
from app.utils import normalise_system

# ─── Tool Metadata Mapping (v2.5: id → name + id → thoughtSignature) ─────────
def build_tool_metadata(messages: list[dict]) -> tuple[dict[str, str], dict[str, str]]:
    """Returns (tool_id -> name, tool_id -> thought_signature) for full round-trip"""
    id_to_name: dict[str, str] = {}
    id_to_sig:  dict[str, str] = {}
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                tid = block.get("id", "")
                name = block.get("name", "")
                sig = block.get("thought_signature")  # captured from previous Gemini response
                if tid and name:
                    id_to_name[tid] = name
                if tid and sig:
                    id_to_sig[tid] = sig
    return id_to_name, id_to_sig

def _tool_result_to_text(block: dict) -> str:
    inner = block.get("content", "")
    is_error = block.get("is_error", False)

    if isinstance(inner, str):
        text = inner
    elif isinstance(inner, list):
        parts = []
        for b in inner:
            if isinstance(b, dict):
                if b.get("type") == "text":
                    parts.append(b.get("text", ""))
                elif b.get("type") == "image":
                    parts.append("[image content stripped]")
            elif isinstance(b, str):
                parts.append(b)
        text = "\n".join(p for p in parts if p)
    else:
        text = str(inner) if inner else ""

    if not text.strip():
        text = "Success (no output)" if not is_error else "Error (no details provided)"
    elif is_error:
        text = f"[ERROR] {text}"

    return text

def _image_block_to_gemini_part(block: dict) -> dict | None:
    source = block.get("source", {})
    src_type = source.get("type", "")
    if src_type == "base64":
        return {
            "inlineData": {
                "mimeType": source.get("media_type", "image/jpeg"),
                "data":     source.get("data", ""),
            }
        }
    if src_type == "url":
        url = source.get("url", "")
        return {"fileData": {"mimeType": source.get("media_type", "image/jpeg"), "fileUri": url}}
    return None

def _msg_to_gemini_parts(msg: dict, id_to_name: dict[str, str], id_to_sig: dict[str, str]) -> tuple[str, list[dict]]:
    role    = msg.get("role", "user")
    content = msg.get("content", "")
    g_role  = "model" if role == "assistant" else "user"
    parts: list[dict] = []

    if isinstance(content, str):
        if content.strip():
            parts.append({"text": content})
        return g_role, parts

    if not isinstance(content, list):
        return g_role, parts

    for block in content:
        if not isinstance(block, dict):
            if isinstance(block, str) and block.strip():
                parts.append({"text": block})
            continue

        btype = block.get("type", "")

        if btype == "text":
            text = block.get("text", "")
            if text:
                parts.append({"text": text})

        elif btype in ("thinking", "redacted_thinking"):
            continue

        elif btype == "image":
            part = _image_block_to_gemini_part(block)
            if part:
                parts.append(part)

        elif btype == "tool_use":
            tid = block.get("id", "")
            fc = {
                "id":   tid,
                "name": block.get("name", ""),
                "args": block.get("input", {}),
            }
            part = {"functionCall": fc}
            if tid and tid in id_to_sig:
                part["thoughtSignature"] = id_to_sig[tid]
            parts.append(part)

        elif btype == "tool_result":
            tid       = block.get("tool_use_id", "")
            func_name = id_to_name.get(tid, tid) or "unknown_tool"
            result_text = _tool_result_to_text(block)
            parts.append({
                "functionResponse": {
                    "id":       tid,
                    "name":     func_name,
                    "response": {"output": result_text},
                }
            })

    return g_role, parts

def anthropic_messages_to_gemini(messages: list[dict], id_to_name: dict[str, str], id_to_sig: dict[str, str]) -> list[dict]:
    raw: list[tuple[str, list[dict]]] = []

    for msg in messages:
        g_role, parts = _msg_to_gemini_parts(msg, id_to_name, id_to_sig)
        if parts:
            raw.append((g_role, parts))

    if not raw:
        return []

    merged: list[dict] = []
    for g_role, parts in raw:
        if merged and merged[-1]["role"] == g_role:
            merged[-1]["parts"].extend(parts)
        else:
            merged.append({"role": g_role, "parts": parts})

    if merged and merged[0]["role"] != "user":
        merged.insert(0, {"role": "user", "parts": [{"text": " "}]})

    return merged

def anthropic_tools_to_gemini(tools: list[dict]) -> list[dict]:
    declarations = []
    for t in tools:
        raw_schema = copy.deepcopy(t.get("input_schema", {"type": "object", "properties": {}}))
        schema     = sanitise_schema(raw_schema)
        decl = {
            "name":        t.get("name", ""),
            "description": t.get("description", "") or "",
            "parameters":  schema,
        }
        declarations.append(decl)
    return [{"functionDeclarations": declarations}]

def anthropic_tool_choice_to_gemini(tool_choice: Any) -> dict | None:
    if tool_choice is None or tool_choice == "auto":
        return {"functionCallingConfig": {"mode": "AUTO"}}
    if tool_choice == "any":
        return {"functionCallingConfig": {"mode": "ANY"}}
    if isinstance(tool_choice, dict):
        tc_type = tool_choice.get("type", "")
        if tc_type == "tool":
            return {
                "functionCallingConfig": {
                    "mode":                 "ANY",
                    "allowedFunctionNames": [tool_choice["name"]],
                }
            }
        if tc_type == "none":
            return {"functionCallingConfig": {"mode": "NONE"}}
    return {"functionCallingConfig": {"mode": "AUTO"}}

def build_gemini_payload(body: dict) -> tuple[dict, str, str]:
    original_model = body.get("model", "claude-sonnet-4-6")
    gemini_model   = resolve_model(original_model)

    messages = body.get("messages", [])
    id_to_name, id_to_sig = build_tool_metadata(messages)
    contents = anthropic_messages_to_gemini(messages, id_to_name, id_to_sig)

    payload: dict[str, Any] = {"contents": contents}

    system_text = normalise_system(body.get("system"))
    if system_text:
        payload["systemInstruction"] = {"parts": [{"text": system_text}]}

    gen_cfg: dict[str, Any] = {}
    if body.get("max_tokens"):
        gen_cfg["maxOutputTokens"] = body["max_tokens"]
    if body.get("temperature") is not None:
        gen_cfg["temperature"] = body["temperature"]
    if body.get("top_p") is not None:
        gen_cfg["topP"] = body["top_p"]
    if body.get("stop_sequences"):
        gen_cfg["stopSequences"] = body["stop_sequences"]
    if gen_cfg:
        payload["generationConfig"] = gen_cfg

    if body.get("tools"):
        payload["tools"] = anthropic_tools_to_gemini(body["tools"])
    if body.get("tool_choice"):
        tc = anthropic_tool_choice_to_gemini(body["tool_choice"])
        if tc:
            payload["toolConfig"] = tc

    return payload, original_model, gemini_model

# ─── Gemini → Anthropic Conversion ───────────────────────────────────────────
_FINISH_MAP: dict[str | None, str] = {
    "STOP":                       "end_turn",
    "MAX_TOKENS":                 "max_tokens",
    "SAFETY":                     "end_turn",
    "RECITATION":                 "end_turn",
    "OTHER":                      "end_turn",
    "BLOCKLIST":                  "end_turn",
    "PROHIBITED_CONTENT":         "end_turn",
    "SPII":                       "end_turn",
    "MALFORMED_FUNCTION_CALL":    "end_turn",
    "TOOL_CODE":                  "tool_use",
    "FUNCTION_CALL":              "tool_use",
    None:                         "end_turn",
}

def gemini_finish_to_anthropic(finish_reason: str | None) -> str:
    return _FINISH_MAP.get(finish_reason, "end_turn")

def _parse_gemini_parts(parts: list[dict]) -> tuple[list[dict], str]:
    blocks: list[dict] = []
    stop_override = ""

    for part in parts:
        if not isinstance(part, dict):
            continue
        if "text" in part:
            text = part["text"]
            if text:
                blocks.append({"type": "text", "text": text})
        elif "functionCall" in part:
            fc = part["functionCall"]
            tool_id = fc.get("id", f"toolu_{uuid.uuid4().hex[:16]}")
            stop_override = "tool_use"
            block = {
                "type":  "tool_use",
                "id":    tool_id,
                "name":  fc.get("name", ""),
                "input": fc.get("args", {}),
            }
            if "thoughtSignature" in part:
                block["thought_signature"] = part["thoughtSignature"]
            blocks.append(block)

    return blocks, stop_override

def gemini_response_to_anthropic(gem: dict, original_model: str) -> dict:
    candidates = gem.get("candidates", [])

    if not candidates:
        block_reason = gem.get("promptFeedback", {}).get("blockReason", "UNKNOWN")
        return {
            "id":            f"msg_{uuid.uuid4().hex}",
            "type":          "message",
            "role":          "assistant",
            "model":         original_model,
            "content":       [{"type": "text", "text": f"[Blocked by Gemini safety filter: {block_reason}]"}],
            "stop_reason":   "end_turn",
            "stop_sequence": None,
            "usage":         extract_usage(gem),
        }

    candidate   = candidates[0]
    parts       = candidate.get("content", {}).get("parts", [])
    finish      = candidate.get("finishReason")
    stop_reason = gemini_finish_to_anthropic(finish)

    blocks, stop_override = _parse_gemini_parts(parts)
    if stop_override:
        stop_reason = stop_override

    if not blocks:
        safety_ratings = candidate.get("safetyRatings", [])
        blocked = [r for r in safety_ratings if r.get("blocked")]
        if blocked:
            categories = ", ".join(r.get("category", "?") for r in blocked)
            blocks = [{"type": "text", "text": f"[Blocked by safety filter: {categories}]"}]
        else:
            blocks = [{"type": "text", "text": " "}]

    return {
        "id":            f"msg_{uuid.uuid4().hex}",
        "type":          "message",
        "role":          "assistant",
        "model":         original_model,
        "content":       blocks,
        "stop_reason":   stop_reason,
        "stop_sequence": None,
        "usage":         extract_usage(gem),
    }

def extract_usage(gem: dict) -> dict:
    u = gem.get("usageMetadata", {})
    return {
        "input_tokens":  u.get("promptTokenCount", 0),
        "output_tokens": u.get("candidatesTokenCount", 0),
    }
