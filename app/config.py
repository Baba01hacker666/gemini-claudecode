import os
from pathlib import Path

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

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
PROXY_API_KEY   = os.getenv("PROXY_API_KEY")
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
PROXY_PORT      = int(os.getenv("PROXY_PORT", 8083))
LOG_LEVEL       = os.getenv("LOG_LEVEL", "INFO")

DEFAULT_MODEL = os.getenv("GEMINI_MODEL",       "gemini-3.1-pro-preview")
SMALL_MODEL   = os.getenv("GEMINI_SMALL_MODEL", "gemini-3-flash-preview")
