"""
DCT Claude Code → Gemini Proxy v2.5
Author: baba01hacker / Doraemon Cyber Team
Translates Anthropic /v1/messages API → Google Gemini generateContent API

Hardened Edge Cases (v2.5):
  - FULL thoughtSignature preservation (Gemini 3.1+ strict function calling)
  - Real signature captured from Gemini response → echoed as sibling to functionCall in Part
  - Fixes "Unknown name thoughtSignature at 'contents[...].parts[0].function_call'" 400s
  - Handles tool_use + tool_result multi-turn chains (len(messages)=5+)
  - Interleaved text + functionCall streaming (correct block ordering)
  - Optional PROXY_API_KEY zero-trust auth
  - Deep schema sanitization, empty tool_result injection, Gemini 3.1+ compliance
"""

import os
import json
import uuid
import copy
import logging
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any

import httpx

# ─── Load .env ────────────────────────────────────────────────────────────────
def _load_dotenv(dotenv_path: str = ".env") -> None:
    p = Path(dotenv_path)
    if not p.exists():
        return
    loaded = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val
            loaded.append(key)
    if loaded:
        print(f"[.env] loaded: {', '.join(loaded)}")

_load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
PROXY_API_KEY   = os.getenv("PROXY_API_KEY")          # ← red-team zero-trust auth (optional)
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
PROXY_PORT      = int(os.getenv("PROXY_PORT", 8083))
LOG_LEVEL       = os.getenv("LOG_LEVEL", "INFO")

DEFAULT_MODEL = os.getenv("GEMINI_MODEL",       "gemini-3.1-pro-preview")
SMALL_MODEL   = os.getenv("GEMINI_SMALL_MODEL", "gemini-3-flash-preview")

MODEL_MAP: dict[str, str] = {
    "claude-opus-4-6":              DEFAULT_MODEL,
    "claude-sonnet-4-6":            DEFAULT_MODEL,
    "claude-opus-4-5":              DEFAULT_MODEL,
    "claude-opus-4-5-20250929":     DEFAULT_MODEL,
    "claude-sonnet-4-5":            DEFAULT_MODEL,
    "claude-sonnet-4-5-20250929":   DEFAULT_MODEL,
    "claude-haiku-4-5":             SMALL_MODEL,
    "claude-haiku-4-5-20251001":    SMALL_MODEL,
    "claude-3-opus-20240229":       DEFAULT_MODEL,
    "claude-3-5-sonnet-20241022":   DEFAULT_MODEL,
    "claude-3-5-haiku-20241022":    SMALL_MODEL,
    "claude-3-haiku-20240307":      SMALL_MODEL,
    "default":                      DEFAULT_MODEL,
}

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log_req = logging.getLogger("dct-gemini.request")
log_tr  = logging.getLogger("dct-gemini.translate")

# ─── Helpers ──────────────────────────────────────────────────────────────────
def resolve_model(anthropic_model: str) -> str:
    mapped = MODEL_MAP.get(anthropic_model, MODEL_MAP["default"])
    if anthropic_model not in MODEL_MAP:
        log_tr.warning("Unknown model %r — falling back to %s", anthropic_model, mapped)
    else:
        log_tr.debug("Model map: %s → %s", anthropic_model, mapped)
    return mapped

def normalise_system(raw: Any) -> str:
    if not raw:
        return ""
    if isinstance(raw, str):
        return raw
    if isinstance(raw, list):
        parts = [b.get("text", "") for b in raw if isinstance(b, dict) and b.get("type") == "text"]
        return "\n".join(p for p in parts if p)
    return str(raw)

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

def _sanitise_schema(schema: Any, depth: int = 0) -> Any:
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
            k: _sanitise_schema(v, depth + 1)
            for k, v in schema["properties"].items()
            if isinstance(v, dict)
        }
    elif schema.get("type") == "object":
        schema["properties"] = {}

    if "items" in schema:
        schema["items"] = _sanitise_schema(schema["items"], depth + 1)
    elif schema.get("type") == "array":
        schema["items"] = {"type": "string"}

    return schema

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

# ─── Turn Generation (v2.5: thoughtSignature as sibling to functionCall) ─────
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
            # Gemini 3.1+ requires real signature as sibling (not inside functionCall)
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
        schema     = _sanitise_schema(raw_schema)
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
    id_to_name, id_to_sig = build_tool_metadata(messages)   # v2.5 metadata
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
            # Capture real signature (sibling in Gemini Part)
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
            "usage":         _extract_usage(gem),
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
        "usage":         _extract_usage(gem),
    }

def _extract_usage(gem: dict) -> dict:
    u = gem.get("usageMetadata", {})
    return {
        "input_tokens":  u.get("promptTokenCount", 0),
        "output_tokens": u.get("candidatesTokenCount", 0),
    }

# ─── SSE Streaming Tools ──────────────────────────────────────────────────────
def sse(event: str, data: dict) -> bytes:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()

class ProxyHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _read_json(self) -> dict:
        length = self.headers.get("Content-Length")
        raw = self.rfile.read(int(length)) if length else self.rfile.read()
        return json.loads(raw)

    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, status: int, message: str):
        log_req.error("→ %d ERROR: %.300s", status, message)
        self._send_json(status, {"error": {"message": message, "type": "proxy_error"}})

    def _gemini_headers(self) -> dict:
        return {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}

    def _gemini_url(self, model: str, stream: bool) -> str:
        method = "streamGenerateContent" if stream else "generateContent"
        url = f"{GEMINI_BASE_URL}/models/{model}:{method}"
        return url + "?alt=sse" if stream else url

    def do_GET(self):
        if self.path.split("?")[0] in ("/", "/health"):
            self._send_json(200, {
                "status":        "ok",
                "service":       "DCT Claude→Gemini Proxy v2.5",
                "default_model": DEFAULT_MODEL,
                "small_model":   SMALL_MODEL,
                "upstream":      GEMINI_BASE_URL,
            })
        else:
            self._send_error(404, "not found")

    def do_POST(self):
        if self.path.split("?")[0] != "/v1/messages":
            self._send_error(404, "only /v1/messages is supported")
            return

        # Red-team proxy auth
        if PROXY_API_KEY:
            provided = (self.headers.get("x-api-key") or
                        self.headers.get("Authorization", "").replace("Bearer ", "").strip())
            if provided != PROXY_API_KEY:
                self._send_error(401, "invalid or missing API key")
                return

        try:
            body = self._read_json()
        except Exception as e:
            self._send_error(400, f"invalid JSON: {e}")
            return

        is_streaming = body.get("stream", False)
        original_model = body.get("model", "claude-sonnet-4-6")
        log_req.info("→ %s | model=%s | stream=%s | len(messages)=%d",
                     "STREAM" if is_streaming else "SYNC",
                     original_model, is_streaming, len(body.get("messages", [])))

        try:
            gem_payload, original_model, gemini_model = build_gemini_payload(body)
        except Exception as e:
            log_req.error("Payload build failed: %s", e, exc_info=True)
            self._send_error(400, f"payload build error: {e}")
            return

        t_start = time.monotonic()
        if is_streaming:
            self._handle_stream(gem_payload, original_model, gemini_model, t_start)
        else:
            self._handle_sync(gem_payload, original_model, gemini_model, t_start)

    def _handle_sync(self, gem_payload: dict, original_model: str, gemini_model: str, t_start: float):
        url = self._gemini_url(gemini_model, stream=False)
        try:
            with httpx.Client(timeout=120.0) as client:
                resp = client.post(url, headers=self._gemini_headers(), json=gem_payload)
        except Exception as e:
            self._send_error(502, f"Network Error: {e}")
            return

        if resp.status_code != 200:
            self._send_error(resp.status_code, resp.text)
            return

        try:
            gem_json = resp.json()
            anthropic_resp = gemini_response_to_anthropic(gem_json, original_model)
        except Exception as e:
            self._send_error(500, f"Translation Error: {e}")
            return

        self._send_json(200, anthropic_resp)

    def _handle_stream(self, gem_payload: dict, original_model: str, gemini_model: str, t_start: float):
        url = self._gemini_url(gemini_model, stream=True)

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def write_sse(event: str, data: dict):
            try:
                self.wfile.write(sse(event, data))
                self.wfile.flush()
            except Exception:
                raise

        def send_error_event(message: str):
            try:
                self.wfile.write(f"event: error\ndata: {json.dumps({'type':'error','error':{'type':'api_error','message':message}})}\n\n".encode())
                self.wfile.flush()
            except Exception:
                pass

        msg_id = f"msg_{uuid.uuid4().hex}"
        write_sse("message_start", {
            "type": "message_start",
            "message": {
                "id": msg_id, "type": "message", "role": "assistant", "model": original_model,
                "content": [], "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        })

        text_block_open = False
        text_block_idx = 0
        next_block_idx = 0
        stop_reason = "end_turn"
        final_usage = {"input_tokens": 0, "output_tokens": 0}

        try:
            with httpx.Client(timeout=180.0) as client:
                with client.stream("POST", url, headers=self._gemini_headers(), json=gem_payload) as resp:
                    if resp.status_code != 200:
                        send_error_event(f"Gemini API Error {resp.status_code}: {resp.read().decode(errors='replace')[:200]}")
                        return

                    for line in resp.iter_lines():
                        if not line.startswith(b"data: "): continue
                        raw = line[6:].strip()
                        if not raw: continue
                        
                        try:
                            chunk = json.loads(raw)
                        except json.JSONDecodeError:
                            continue

                        if chunk.get("usageMetadata"):
                            final_usage = _extract_usage(chunk)

                        candidates = chunk.get("candidates", [])
                        if not candidates:
                            br = chunk.get("promptFeedback", {}).get("blockReason")
                            if br:
                                if not text_block_open:
                                    text_block_idx, next_block_idx = next_block_idx, next_block_idx + 1
                                    text_block_open = True
                                    write_sse("content_block_start", {"type": "content_block_start", "index": text_block_idx, "content_block": {"type": "text", "text": ""}})
                                write_sse("content_block_delta", {"type": "content_block_delta", "index": text_block_idx, "delta": {"type": "text_delta", "text": f"[Blocked: {br}]"}})
                            continue

                        candidate = candidates[0]
                        parts = candidate.get("content", {}).get("parts", [])
                        finish = candidate.get("finishReason")

                        for part in parts:
                            if "text" in part:
                                text_chunk = part["text"]
                                if not text_chunk: continue
                                if not text_block_open:
                                    text_block_idx, next_block_idx = next_block_idx, next_block_idx + 1
                                    text_block_open = True
                                    write_sse("content_block_start", {"type": "content_block_start", "index": text_block_idx, "content_block": {"type": "text", "text": ""}})
                                write_sse("content_block_delta", {"type": "content_block_delta", "index": text_block_idx, "delta": {"type": "text_delta", "text": text_chunk}})

                            elif "functionCall" in part:
                                # close pending text block before tool_use
                                if text_block_open:
                                    write_sse("content_block_stop", {"type": "content_block_stop", "index": text_block_idx})
                                    text_block_open = False

                                fc = part["functionCall"]
                                tool_id = fc.get("id", f"toolu_{uuid.uuid4().hex[:16]}")
                                blk_idx = next_block_idx
                                next_block_idx += 1
                                stop_reason = "tool_use"

                                # v2.5: include captured signature in the Anthropic block
                                content_block = {
                                    "type": "tool_use",
                                    "id": tool_id,
                                    "name": fc.get("name", ""),
                                    "input": {}
                                }
                                if "thoughtSignature" in part:
                                    content_block["thought_signature"] = part["thoughtSignature"]

                                write_sse("content_block_start", {"type": "content_block_start", "index": blk_idx, "content_block": content_block})
                                write_sse("content_block_delta", {"type": "content_block_delta", "index": blk_idx, "delta": {"type": "input_json_delta", "partial_json": json.dumps(fc.get("args", {}))}})
                                write_sse("content_block_stop", {"type": "content_block_stop", "index": blk_idx})

                        if finish and finish not in ("", "FINISH_REASON_UNSPECIFIED"):
                            sr = gemini_finish_to_anthropic(finish)
                            if sr == "tool_use" or stop_reason != "tool_use":
                                stop_reason = sr

        except Exception as e:
            send_error_event(f"Stream Failed: {e}")
            return

        try:
            if text_block_open:
                write_sse("content_block_stop", {"type": "content_block_stop", "index": text_block_idx})
            write_sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None}, "usage": final_usage})
            write_sse("message_stop", {"type": "message_stop"})
        except Exception:
            pass

class ThreadedHTTPServer(HTTPServer):
    def process_request(self, request, client_address):
        t = threading.Thread(target=self._finish_request_thread, args=(request, client_address), daemon=True)
        t.start()
    def _finish_request_thread(self, request, client_address):
        try: self.finish_request(request, client_address)
        except Exception: self.handle_error(request, client_address)
        finally: self.shutdown_request(request)

if __name__ == "__main__":
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set — add it to .env or export it")
    
    server = ThreadedHTTPServer(("0.0.0.0", PROXY_PORT), ProxyHandler)
    print(f"[DCT Proxy v2.5] listening on http://0.0.0.0:{PROXY_PORT} | GEMINI_MODEL={DEFAULT_MODEL} | PROXY_API_KEY={'SET' if PROXY_API_KEY else 'DISABLED'}")
    try: server.serve_forever()
    except KeyboardInterrupt: server.server_close()
