"""Constants and utilities for the OpenAI-Compatible provider.

Defines the mapping between OpenAI API concepts and Atlas domain models.
"""

from __future__ import annotations

from app.provider.models import StopReason


# ---------------------------------------------------------------------------
# Stop reason mapping
# ---------------------------------------------------------------------------

STOP_REASON_MAP: dict[str, StopReason] = {
    "stop": StopReason.STOP,
    "length": StopReason.LENGTH,
    "max_tokens": StopReason.LENGTH,
    "tool_calls": StopReason.TOOL_CALL,
    "content_filter": StopReason.CONTENT_FILTER,
    "error": StopReason.ERROR,
    "timeout": StopReason.TIMEOUT,
    "cancelled": StopReason.CANCELLED,
}


def map_stop_reason(reason: str | None) -> StopReason:
    """Map an OpenAI finish_reason to an Atlas ``StopReason``."""
    if reason is None:
        return StopReason.UNKNOWN
    if reason == "":
        return StopReason.UNKNOWN
    return STOP_REASON_MAP.get(reason, StopReason.UNKNOWN)


# ---------------------------------------------------------------------------
# Role mapping
# ---------------------------------------------------------------------------

ROLE_MAP: dict[str, str] = {
    "system": "system",
    "user": "user",
    "assistant": "assistant",
    "tool": "tool",
}


# ---------------------------------------------------------------------------
# API constants
# ---------------------------------------------------------------------------

DEFAULT_API_PATH = "/chat/completions"
DEFAULT_STREAM_PATH = "/chat/completions"
CONTENT_TYPE_HEADER = "application/json"
