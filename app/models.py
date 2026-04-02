from app.config import DEFAULT_MODEL, SMALL_MODEL
from app.logger import log_tr

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

def resolve_model(anthropic_model: str) -> str:
    mapped = MODEL_MAP.get(anthropic_model, MODEL_MAP["default"])
    if anthropic_model not in MODEL_MAP:
        log_tr.warning("Unknown model %r — falling back to %s", anthropic_model, mapped)
    else:
        log_tr.debug("Model map: %s → %s", anthropic_model, mapped)
    return mapped
