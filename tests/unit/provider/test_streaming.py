"""Tests for provider streaming integration."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from app.agent.config import AgentConfig
from app.agent.events import (
    AgentEvent,
    ProviderChunkReceivedEvent,
    ProviderStreamStartedEvent,
    ProviderStreamCompletedEvent,
)
from app.agent.runtime import AgentRuntime
from app.provider.models import (
    ProviderMessage,
    ProviderRequest,
    ProviderResponse,
    ProviderUsage,
    Role,
    StopReason,
    StreamingChunk,
    ToolCallRequest,
)
from app.provider.provider import Provider
from app.provider.streaming import StreamChunkType, ProviderStreamResult, aggregate_stream
from app.provider.stream import StreamAccumulator
from app.tools.models import ToolDefinition, ToolParameter
from app.tools.registry import ToolRegistry
from app.tools.runtime import ToolRuntime


# ---------------------------------------------------------------------------
# Mock streaming provider
# ---------------------------------------------------------------------------


class _MockStreamingProvider(Provider):
    """A mock provider that yields streaming chunks."""

    def __init__(
        self,
        chunks: list[StreamingChunk],
        *,
        fail_after: int = -1,
        stream_fail: bool = False,
        name: str = "mock",
    ) -> None:
        self._chunks = list(chunks)
        self._call_count = 0
        self._fail_after = fail_after
        self._stream_fail = stream_fail
        self._name = name

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        self._call_count += 1
        # Accumulate chunks into a response
        content = "".join(c.content for c in self._chunks)
        return ProviderResponse(
            content=content,
            message=ProviderMessage(role=Role.ASSISTANT, content=content),
            stop_reason=StopReason.STOP,
        )

    async def stream(self, request: ProviderRequest) -> AsyncIterator[StreamingChunk]:
        self._call_count += 1
        for i, chunk in enumerate(self._chunks):
            if self._fail_after >= 0 and i >= self._fail_after:
                raise RuntimeError("Stream failed")
            yield chunk

    async def count_tokens(self, request: ProviderRequest) -> int:
        return 0

    @property
    def provider_info(self) -> Any:
        from app.provider.models import ProviderCapability, ProviderInfo, ProviderMetadata
        return ProviderInfo(
            metadata=ProviderMetadata(name=self._name),
            capabilities=[ProviderCapability(name="streaming"), ProviderCapability(name="tool_calling")],
        )


class _MockNonStreamingProvider(Provider):
    """A provider that doesn't support streaming."""

    def __init__(self) -> None:
        self._call_count = 0

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        self._call_count += 1
        return ProviderResponse(content="non-streaming response")

    def stream(self, request: ProviderRequest) -> AsyncIterator[StreamingChunk]:
        self._call_count += 1
        raise NotImplementedError("Streaming not supported")

    async def count_tokens(self, request: ProviderRequest) -> int:
        return 0

    @property
    def provider_info(self) -> Any:
        from app.provider.models import ProviderCapability, ProviderInfo, ProviderMetadata
        return ProviderInfo(metadata=ProviderMetadata(name="mock"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(content: str, index: int = 0) -> StreamingChunk:
    return StreamingChunk(content=content, index=index)


def _make_tool_chunk(name: str, args: dict[str, Any] | None = None) -> StreamingChunk:
    return StreamingChunk(
        tool_call=ToolCallRequest(id=f"call_{name}", name=name, arguments=args or {}),
    )


def _make_usage_chunk(
    prompt: int = 10,
    completion: int = 5,
) -> StreamingChunk:
    return StreamingChunk(
        usage=ProviderUsage(prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion),
    )


def _create_tool_runtime() -> ToolRuntime:
    registry = ToolRegistry()

    async def calc(expression: str) -> str:
        return str(eval(expression))

    registry.register(ToolDefinition(
        name="calculator",
        parameters=(ToolParameter(name="expression", type="string"),),
        fn=calc,
    ))
    return ToolRuntime(registry)


# ---------------------------------------------------------------------------
# ProviderStreamResult tests
# ---------------------------------------------------------------------------


class TestProviderStreamResult:
    async def test_aggregate_empty_stream(self) -> None:
        """Empty stream produces empty result."""
        async def empty():
            return
            yield  # pragma: no cover

        result = await aggregate_stream(empty())
        assert result.content == ""
        assert result.tool_calls == []

    async def test_aggregate_text_chunks(self) -> None:
        """Text chunks are concatenated in order."""
        async def stream():
            yield _make_chunk("Hello", 0)
            yield _make_chunk(" World", 1)

        result = await aggregate_stream(stream())
        assert result.content == "Hello World"

    async def test_aggregate_tool_calls(self) -> None:
        """Tool call chunks are collected."""
        async def stream():
            yield _make_tool_chunk("calculator", {"expression": "2+2"})

        result = await aggregate_stream(stream())
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "calculator"

    async def test_aggregate_usage(self) -> None:
        """Usage chunks are captured."""
        async def stream():
            yield _make_chunk("text")
            yield _make_usage_chunk(prompt=20, completion=10)

        result = await aggregate_stream(stream())
        assert result.usage.get("prompt_tokens") == 20
        assert result.usage.get("completion_tokens") == 10

    async def test_aggregate_mixed_content(self) -> None:
        """Mixed text and tool chunks are handled."""
        async def stream():
            yield _make_chunk("Thinking...", 0)
            yield _make_tool_chunk("search", {"q": "test"})
            yield _make_chunk(" Done.", 1)

        result = await aggregate_stream(stream())
        assert result.content == "Thinking... Done."
        assert len(result.tool_calls) == 1


# ---------------------------------------------------------------------------
# StreamChunkType tests
# ---------------------------------------------------------------------------


class TestStreamChunkType:
    def test_enum_values(self) -> None:
        assert StreamChunkType.TEXT.value == "text"
        assert StreamChunkType.THINKING.value == "thinking"
        assert StreamChunkType.TOOL_CALL.value == "tool_call"
        assert StreamChunkType.TOOL_RESULT.value == "tool_result"
        assert StreamChunkType.MESSAGE_END.value == "message_end"
        assert StreamChunkType.ERROR.value == "error"
        assert StreamChunkType.USAGE.value == "usage"


# ---------------------------------------------------------------------------
# StreamAccumulator tests
# ---------------------------------------------------------------------------


class TestStreamAccumulator:
    def test_accumulate_single_chunk(self) -> None:
        acc = StreamAccumulator()
        acc.add(_make_chunk("Hello"))
        assert acc.chunk_count == 1
        assert acc.result().full_content == "Hello"

    def test_accumulate_multiple_chunks(self) -> None:
        acc = StreamAccumulator()
        acc.add(_make_chunk("A"))
        acc.add(_make_chunk("B"))
        acc.add(_make_chunk("C"))
        result = acc.result()
        assert result.full_content == "ABC"
        assert len(result.chunks) == 3

    def test_accumulate_stop_reason(self) -> None:
        acc = StreamAccumulator()
        acc.add(StreamingChunk(stop_reason=StopReason.STOP))
        assert acc.result().stop_reason == StopReason.STOP

    def test_accumulate_tool_call(self) -> None:
        acc = StreamAccumulator()
        tc = ToolCallRequest(id="c1", name="test")
        acc.add(StreamingChunk(tool_call=tc))
        assert len(acc.result().tool_calls) == 1
        assert acc.result().tool_calls[0].name == "test"

    def test_to_response(self) -> None:
        acc = StreamAccumulator()
        acc.add(_make_chunk("Hello"))
        response = acc.to_response()
        assert response.content == "Hello"
        assert response.message.content == "Hello"

    def test_reset(self) -> None:
        acc = StreamAccumulator()
        acc.add(_make_chunk("Hello"))
        acc.reset()
        assert acc.is_empty
        assert acc.result().full_content == ""


# ---------------------------------------------------------------------------
# Runtime streaming integration tests
# ---------------------------------------------------------------------------


class TestRuntimeStreamingIntegration:
    async def test_stream_yields_chunks(self) -> None:
        """Runtime.stream() yields chunks from provider stream."""
        chunks = [_make_chunk("Hello"), _make_chunk(" World")]
        provider = _MockStreamingProvider(chunks)
        runtime = AgentRuntime(provider, _create_tool_runtime())
        events: list[Any] = []
        async for event in runtime.stream(
            [ProviderMessage(role=Role.USER, content="Hi")],
        ):
            events.append(event)

        # Should have events + final response
        assert len(events) >= 3  # started + events + response

    async def test_stream_fallback_to_generate(self) -> None:
        """Runtime falls back to generate() when stream() not supported."""
        provider = _MockNonStreamingProvider()
        runtime = AgentRuntime(provider, _create_tool_runtime())
        events: list[Any] = []
        async for event in runtime.stream(
            [ProviderMessage(role=Role.USER, content="Hi")],
        ):
            events.append(event)

        # Should work despite no streaming support
        final_response = events[-1]
        from app.provider.models import ProviderResponse as PR
        assert isinstance(final_response, PR)
        assert "non-streaming" in final_response.content

    async def test_stream_empty_message_list(self) -> None:
        """Stream with empty messages works."""
        chunks = [_make_chunk("hello")]
        provider = _MockStreamingProvider(chunks)
        runtime = AgentRuntime(provider, _create_tool_runtime())
        events: list[Any] = []
        async for event in runtime.stream([]):
            events.append(event)
        assert len(events) >= 2

    async def test_stream_conversation_preserved(self) -> None:
        """Original conversation messages are not modified."""
        chunks = [_make_chunk("Ok")]
        provider = _MockStreamingProvider(chunks)
        runtime = AgentRuntime(provider, _create_tool_runtime())
        original = [ProviderMessage(role=Role.USER, content="Hello")]
        async for _ in runtime.stream(original):
            pass
        assert len(original) == 1
        assert original[0].content == "Hello"


# ---------------------------------------------------------------------------
# Provider streaming events
# ---------------------------------------------------------------------------


class TestProviderStreamingEvents:
    async def test_stream_started_event(self) -> None:
        """ProviderStreamStartedEvent is emitted."""
        chunks = [_make_chunk("hello")]
        provider = _MockStreamingProvider(chunks)
        runtime = AgentRuntime(provider, _create_tool_runtime())
        event_types: list[str] = []

        def listener(event: AgentEvent) -> None:
            event_types.append(event.event_type)

        runtime.dispatcher.add_listener(listener)
        async for _ in runtime.stream(
            [ProviderMessage(role=Role.USER, content="Hi")],
        ):
            pass

        assert "provider_stream_started" in event_types

    async def test_stream_completed_event(self) -> None:
        """ProviderStreamCompletedEvent is emitted."""
        chunks = [_make_chunk("hello")]
        provider = _MockStreamingProvider(chunks)
        runtime = AgentRuntime(provider, _create_tool_runtime())
        event_types: list[str] = []

        def listener(event: AgentEvent) -> None:
            event_types.append(event.event_type)

        runtime.dispatcher.add_listener(listener)
        async for _ in runtime.stream(
            [ProviderMessage(role=Role.USER, content="Hi")],
        ):
            pass

        assert "provider_stream_completed" in event_types


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestStreamingConfig:
    def test_default_streaming_enabled(self) -> None:
        config = AgentConfig.default()
        assert config.streaming_enabled is True

    def test_chunk_buffer_size(self) -> None:
        config = AgentConfig(chunk_buffer_size=5)
        assert config.chunk_buffer_size == 5

    def test_invalid_chunk_buffer_size(self) -> None:
        with pytest.raises(ValueError):
            AgentConfig(chunk_buffer_size=0)

    def test_emit_thinking_chunks(self) -> None:
        config = AgentConfig(emit_thinking_chunks=False)
        assert config.emit_thinking_chunks is False
