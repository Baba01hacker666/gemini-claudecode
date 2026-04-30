import json
import time
import uuid
import httpx
from http.server import BaseHTTPRequestHandler
from app.config import GEMINI_API_KEY, PROXY_API_KEY, GEMINI_BASE_URL, OLLAMA_BASE_URL, UPSTREAM_PROVIDER
from app.logger import log_req
from app.schema import validate_anthropic_request
from app.translator import (
    build_gemini_payload,
    build_ollama_payload,
    gemini_response_to_anthropic,
    ollama_response_to_anthropic,
    gemini_finish_to_anthropic,
    extract_usage
)
from app.utils import sse
from app.errors import format_error_response

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

    def _send_error(self, status: int, message: str, error_type: str = "proxy_error"):
        error_payload = format_error_response(status, message, error_type)
        self._send_json(status, error_payload)

    def _gemini_headers(self) -> dict:
        if UPSTREAM_PROVIDER == "ollama":
            return {"Content-Type": "application/json"}
        return {"Content-Type": "application/json", "x-goog-api-key": GEMINI_API_KEY}

    def _gemini_url(self, model: str, stream: bool) -> str:
        if UPSTREAM_PROVIDER == "ollama":
            return f"{OLLAMA_BASE_URL}/api/chat"
        method = "streamGenerateContent" if stream else "generateContent"
        url = f"{GEMINI_BASE_URL}/models/{model}:{method}"
        return url + "?alt=sse" if stream else url

    def do_GET(self):
        from app.config import DEFAULT_MODEL, SMALL_MODEL
        if self.path.split("?")[0] in ("/", "/health"):
            self._send_json(200, {
                "status":        "ok",
                "service":       f"DCT Claude→{UPSTREAM_PROVIDER.capitalize()} Proxy v2.5",
                "default_model": DEFAULT_MODEL,
                "small_model":   SMALL_MODEL,
                "upstream":      OLLAMA_BASE_URL if UPSTREAM_PROVIDER == "ollama" else GEMINI_BASE_URL,
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
                self._send_error(401, "invalid or missing API key", "authentication_error")
                return

        try:
            body = self._read_json()
        except Exception as e:
            self._send_error(400, f"invalid JSON: {e}")
            return

        # Input Validation
        try:
            validate_anthropic_request(body)
        except ValueError as e:
            self._send_error(400, str(e), "invalid_request_error")
            return

        is_streaming = body.get("stream", False)
        original_model = body.get("model", "claude-sonnet-4-6")
        log_req.info("→ %s | model=%s | stream=%s | len(messages)=%d",
                     "STREAM" if is_streaming else "SYNC",
                     original_model, is_streaming, len(body.get("messages", [])))

        try:
            if UPSTREAM_PROVIDER == "ollama":
                gem_payload, original_model, gemini_model = build_ollama_payload(body)
            else:
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
            self._send_error(resp.status_code, resp.text, "api_error")
            return

        try:
            gem_json = resp.json()
            anthropic_resp = (
                ollama_response_to_anthropic(gem_json, original_model)
                if UPSTREAM_PROVIDER == "ollama"
                else gemini_response_to_anthropic(gem_json, original_model)
            )
        except Exception as e:
            self._send_error(500, f"Translation Error: {e}")
            return

        self._send_json(200, anthropic_resp)

    def _handle_stream(self, gem_payload: dict, original_model: str, gemini_model: str, t_start: float):
        url = self._gemini_url(gemini_model, stream=True)

        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()

            def write_sse(event: str, data: dict):
                self.wfile.write(sse(event, data))
                self.wfile.flush()

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
        except (BrokenPipeError, ConnectionError):
            log_req.warning("Client disconnected before stream start")
            return

        text_block_open = False
        text_block_idx = 0
        next_block_idx = 0
        stop_reason = "end_turn"
        final_usage = {"input_tokens": 0, "output_tokens": 0}

        try:
            if UPSTREAM_PROVIDER == "ollama":
                with httpx.Client(timeout=180.0) as client:
                    with client.stream("POST", url, headers=self._gemini_headers(), json=gem_payload) as resp:
                        if resp.status_code != 200:
                            send_error_event(f"Ollama API Error {resp.status_code}: {resp.read().decode(errors='replace')[:200]}")
                            return
                        for line in resp.iter_lines():
                            if not line:
                                continue
                            if isinstance(line, bytes):
                                line = line.decode(errors="replace")
                            try:
                                chunk = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            text_chunk = ((chunk.get("message") or {}).get("content")) or ""
                            if text_chunk:
                                if not text_block_open:
                                    text_block_idx, next_block_idx = next_block_idx, next_block_idx + 1
                                    text_block_open = True
                                    write_sse("content_block_start", {"type": "content_block_start", "index": text_block_idx, "content_block": {"type": "text", "text": ""}})
                                write_sse("content_block_delta", {"type": "content_block_delta", "index": text_block_idx, "delta": {"type": "text_delta", "text": text_chunk}})
                            if chunk.get("done") is True:
                                break
                if text_block_open:
                    write_sse("content_block_stop", {"type": "content_block_stop", "index": text_block_idx})
                write_sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": final_usage})
                write_sse("message_stop", {"type": "message_stop"})
                return

            with httpx.Client(timeout=180.0) as client:
                with client.stream("POST", url, headers=self._gemini_headers(), json=gem_payload) as resp:
                    if resp.status_code != 200:
                        send_error_event(f"Gemini API Error {resp.status_code}: {resp.read().decode(errors='replace')[:200]}")
                        return

                    for line in resp.iter_lines():
                        if isinstance(line, bytes):
                            line = line.decode(errors="replace")
                        if not line.startswith("data: "): continue
                        raw = line[6:].strip()
                        if not raw: continue

                        try:
                            chunk = json.loads(raw)
                        except json.JSONDecodeError:
                            continue

                        if chunk.get("usageMetadata"):
                            final_usage = extract_usage(chunk)

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

        except (BrokenPipeError, ConnectionError):
            log_req.warning("Client disconnected during stream")
            return
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
