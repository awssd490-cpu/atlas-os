"""Tests for streaming helpers."""

from __future__ import annotations

import pytest

from app.provider.models import (
    ProviderUsage,
    StopReason,
    StreamingChunk,
    StreamingResult,
    ToolCallRequest,
)
from app.provider.stream import StreamAccumulator, collect_stream


class TestStreamAccumulator:
    def test_empty_accumulator(self) -> None:
        acc = StreamAccumulator()
        assert acc.is_empty is True
        assert acc.chunk_count == 0

    def test_add_chunks(self) -> None:
        acc = StreamAccumulator()
        acc.add(StreamingChunk(content="hello ", index=0))
        acc.add(StreamingChunk(content="world", index=1))
        assert acc.chunk_count == 2
        assert acc.is_empty is False

    def test_result(self) -> None:
        acc = StreamAccumulator()
        acc.add(StreamingChunk(content="hello", index=0))
        acc.add(StreamingChunk(content=" world", index=1))
        result = acc.result()
        assert isinstance(result, StreamingResult)
        assert result.full_content == "hello world"
        assert len(result.chunks) == 2

    def test_stop_reason_captured(self) -> None:
        acc = StreamAccumulator()
        acc.add(StreamingChunk(content="hi", index=0))
        acc.add(StreamingChunk(content="", stop_reason=StopReason.STOP, index=1))
        result = acc.result()
        assert result.stop_reason == StopReason.STOP

    def test_usage_captured(self) -> None:
        acc = StreamAccumulator()
        usage = ProviderUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15)
        acc.add(StreamingChunk(content="done", usage=usage, index=0))
        result = acc.result()
        assert result.usage.prompt_tokens == 10
        assert result.usage.total_tokens == 15

    def test_tool_call_captured(self) -> None:
        acc = StreamAccumulator()
        tc = ToolCallRequest(id="tc1", name="search")
        acc.add(StreamingChunk(content="", tool_call=tc, index=0))
        result = acc.result()
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "search"

    def test_to_response(self) -> None:
        acc = StreamAccumulator()
        acc.add(StreamingChunk(content="hello", index=0))
        response = acc.to_response()
        assert response.content == "hello"
        assert response.stop_reason == StopReason.UNKNOWN

    def test_reset(self) -> None:
        acc = StreamAccumulator()
        acc.add(StreamingChunk(content="hello", index=0))
        acc.reset()
        assert acc.is_empty is True
        assert acc.chunk_count == 0


class TestCollectStream:
    async def test_collect(self) -> None:
        async def _stream():
            yield StreamingChunk(content="a", index=0)
            yield StreamingChunk(content="b", index=1)

        result = await collect_stream(_stream())
        assert result.full_content == "ab"
        assert len(result.chunks) == 2

    async def test_collect_empty(self) -> None:
        async def _empty():
            return
            yield  # pragma: no cover

        result = await collect_stream(_empty())
        assert result.full_content == ""
        assert len(result.chunks) == 0
