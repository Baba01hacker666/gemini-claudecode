import json
from app.logger import log_req

def format_error_response(status: int, message: str, error_type: str = "proxy_error") -> dict:
    log_req.error("→ %d ERROR: %.300s", status, message)
    return {
        "error": {
            "message": message,
            "type": error_type
        }
    }
