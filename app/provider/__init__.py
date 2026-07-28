"""ATLAS Universal Provider Runtime.

Every LLM provider in the system implements the ``Provider`` ABC defined here.
The core never knows whether it is talking to Claude, GPT, Gemini, Ollama,
or any other provider — everything is simply a ``Provider``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.provider.provider import Provider
    from app.provider.models import (
        ProviderRequest,
        ProviderResponse,
        ProviderMessage,
        ProviderUsage,
        ProviderCapability,
        ProviderMetadata,
        ProviderInfo,
        StreamingChunk,
        StopReason,
        FinishReason,
        ToolCallRequest,
        ToolCallResponse,
    )
    from app.provider.registry import ProviderRegistry
    from app.provider.factory import ProviderFactory

__all__ = [
    "Provider",
    "ProviderRequest",
    "ProviderResponse",
    "ProviderMessage",
    "ProviderUsage",
    "ProviderCapability",
    "ProviderMetadata",
    "ProviderInfo",
    "StreamingChunk",
    "StopReason",
    "FinishReason",
    "ToolCallRequest",
    "ToolCallResponse",
    "ProviderRegistry",
    "ProviderFactory",
]
