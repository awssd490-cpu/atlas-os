"""Provider streaming models and helpers.

Defines ``StreamChunkType`` for categorising streamed content and
provides a unified interface for consuming provider token streams.

The existing ``StreamingChunk`` and ``StreamAccumulator`` in
``app/provider/stream.py`` remain the standard for non-streaming
accumulation.  This module adds higher-level streaming types for
the Agent Runtime.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

from app.provider.models import StreamingChunk, ToolCallRequest


class StreamChunkType(str, enum.Enum):
    """Categorises a streamed chunk from a provider.

    Every chunk from a provider stream maps to one of these types.
    """

    TEXT = "text"
    THINKING = "thinking"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    MESSAGE_END = "message_end"
    ERROR = "error"
    USAGE = "usage"


@dataclass(frozen=True)
class ProviderStreamResult:
    """The aggregated result of a streaming provider request.

    Combines accumulated content, tool calls, usage, and final stop
    reason into a single immutable object.
    """

    content: str = ""
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    stop_reason: str = "stop"
    metadata: dict[str, Any] = field(default_factory=dict)


async def aggregate_stream(
    stream: Any,
) -> ProviderStreamResult:
    """Aggregate a provider stream into a ``ProviderStreamResult``.

    Args:
        stream: An async iterator of ``StreamingChunk`` objects.

    Returns:
        An aggregated ``ProviderStreamResult``.
    """
    content_parts: list[str] = []
    tool_calls: list[ToolCallRequest] = []
    usage: dict[str, Any] = {}
    stop_reason: str = "stop"
    metadata: dict[str, Any] = {}

    async for chunk in stream:
        if isinstance(chunk, StreamingChunk):
            if chunk.content:
                content_parts.append(chunk.content)
            if chunk.tool_call is not None:
                tool_calls.append(chunk.tool_call)
            if chunk.stop_reason is not None:
                stop_reason = chunk.stop_reason.value
            if chunk.usage is not None:
                usage = {
                    "prompt_tokens": chunk.usage.prompt_tokens,
                    "completion_tokens": chunk.usage.completion_tokens,
                    "total_tokens": chunk.usage.total_tokens,
                }
            metadata.update(chunk.metadata)

    return ProviderStreamResult(
        content="".join(content_parts),
        tool_calls=tool_calls,
        usage=usage,
        stop_reason=stop_reason,
        metadata=metadata,
    )
