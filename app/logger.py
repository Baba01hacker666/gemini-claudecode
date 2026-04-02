import logging
from app.config import LOG_LEVEL

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)

log_req = logging.getLogger("dct-gemini.request")
log_tr  = logging.getLogger("dct-gemini.translate")
