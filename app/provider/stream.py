"""Provider-independent streaming helpers.

Provides utilities for accumulating streaming chunks into complete
responses and for transforming streaming responses.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from app.provider.models import (
    ProviderResponse,
    ProviderUsage,
    StopReason,
    StreamingChunk,
    StreamingResult,
    ToolCallRequest,
)


class StreamAccumulator:
    """Accumulates streaming chunks into a ``StreamingResult``.

    Usage::

        async for chunk in provider.stream(request):
            accumulator.add(chunk)
        result = accumulator.result()
    """

    def __init__(self) -> None:
        self._chunks: list[StreamingChunk] = []
        self._content_parts: list[str] = []
        self._stop_reason: StopReason = StopReason.UNKNOWN
        self._usage: ProviderUsage = ProviderUsage()
        self._tool_calls: list[ToolCallRequest] = []

    def add(self, chunk: StreamingChunk) -> None:
        """Add a single chunk to the accumulator."""
        self._chunks.append(chunk)
        if chunk.content:
            self._content_parts.append(chunk.content)
        if chunk.stop_reason is not None:
            self._stop_reason = chunk.stop_reason
        if chunk.usage is not None:
            self._usage = chunk.usage
        if chunk.tool_call is not None:
            self._tool_calls.append(chunk.tool_call)

    def result(self) -> StreamingResult:
        """Return the accumulated result."""
        return StreamingResult(
            full_content="".join(self._content_parts),
            chunks=list(self._chunks),
            stop_reason=self._stop_reason,
            usage=self._usage,
            tool_calls=list(self._tool_calls),
        )

    def to_response(self) -> ProviderResponse:
        """Convert the accumulated result to a ``ProviderResponse``."""
        from app.provider.models import ProviderMessage, Role

        result = self.result()
        return ProviderResponse(
            content=result.full_content,
            message=ProviderMessage(
                role=Role.ASSISTANT,
                content=result.full_content,
                tool_calls=list(result.tool_calls),
            ),
            stop_reason=result.stop_reason,
            usage=result.usage,
            tool_calls=list(result.tool_calls),
        )

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)

    @property
    def is_empty(self) -> bool:
        return not self._chunks

    def reset(self) -> None:
        """Clear all accumulated state."""
        self._chunks.clear()
        self._content_parts.clear()
        self._stop_reason = StopReason.UNKNOWN
        self._usage = ProviderUsage()
        self._tool_calls.clear()


async def collect_stream(
    stream: AsyncIterator[StreamingChunk],
) -> StreamingResult:
    """Collect an entire async stream into a ``StreamingResult``.

    Args:
        stream: An async iterator of ``StreamingChunk`` objects.

    Returns:
        The accumulated ``StreamingResult``.
    """
    acc = StreamAccumulator()
    async for chunk in stream:
        acc.add(chunk)
    return acc.result()
