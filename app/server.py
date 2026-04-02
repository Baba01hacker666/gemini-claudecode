import threading
from http.server import HTTPServer
from app.config import PROXY_PORT, GEMINI_API_KEY, DEFAULT_MODEL, PROXY_API_KEY
from app.proxy import ProxyHandler

class ThreadedHTTPServer(HTTPServer):
    def process_request(self, request, client_address):
        t = threading.Thread(target=self._finish_request_thread, args=(request, client_address), daemon=True)
        t.start()
    def _finish_request_thread(self, request, client_address):
        try: self.finish_request(request, client_address)
        except Exception: self.handle_error(request, client_address)
        finally: self.shutdown_request(request)

def run_server():
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set — add it to .env or export it")

    server = ThreadedHTTPServer(("0.0.0.0", PROXY_PORT), ProxyHandler)
    print(f"[DCT Proxy v2.5] listening on http://0.0.0.0:{PROXY_PORT} | GEMINI_MODEL={DEFAULT_MODEL} | PROXY_API_KEY={'SET' if PROXY_API_KEY else 'DISABLED'}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()
