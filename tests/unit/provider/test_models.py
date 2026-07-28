"""Tests for provider domain models."""

from __future__ import annotations

from app.provider.models import (
    Capabilities,
    ContentType,
    FinishReason,
    ProviderCapability,
    ProviderInfo,
    ProviderMessage,
    ProviderMetadata,
    ProviderRequest,
    ProviderResponse,
    ProviderUsage,
    Role,
    StopReason,
    StreamingChunk,
    StreamingResult,
    ToolCallRequest,
    ToolCallResponse,
)


class TestStopReason:
    def test_values(self) -> None:
        assert StopReason.STOP.value == "stop"
        assert StopReason.LENGTH.value == "length"
        assert StopReason.TOOL_CALL.value == "tool_call"


class TestFinishReason:
    def test_alias(self) -> None:
        assert FinishReason is StopReason


class TestRole:
    def test_values(self) -> None:
        assert Role.SYSTEM.value == "system"
        assert Role.USER.value == "user"
        assert Role.ASSISTANT.value == "assistant"
        assert Role.TOOL.value == "tool"


class TestContentType:
    def test_values(self) -> None:
        assert ContentType.TEXT.value == "text"
        assert ContentType.IMAGE.value == "image"


class TestProviderCapability:
    def test_create(self) -> None:
        cap = ProviderCapability(name="streaming", version="1.0", description="Streaming support")
        assert cap.name == "streaming"
        assert cap.version == "1.0"


class TestProviderMetadata:
    def test_create(self) -> None:
        meta = ProviderMetadata(name="claude", version="3.5", description="Anthropic Claude")
        assert meta.name == "claude"


class TestProviderInfo:
    def test_has_capability(self) -> None:
        caps = [ProviderCapability(name="streaming"), ProviderCapability(name="vision")]
        info = ProviderInfo(capabilities=caps)
        assert info.has_capability("streaming") is True
        assert info.has_capability("audio") is False

    def test_capability_names(self) -> None:
        caps = [ProviderCapability(name="a"), ProviderCapability(name="b")]
        info = ProviderInfo(capabilities=caps)
        assert info.capability_names == ["a", "b"]


class TestToolCallRequest:
    def test_create(self) -> None:
        tc = ToolCallRequest(id="call-1", name="get_weather", arguments={"city": "NYC"})
        assert tc.name == "get_weather"
        assert tc.arguments["city"] == "NYC"


class TestToolCallResponse:
    def test_create(self) -> None:
        tr = ToolCallResponse(call_id="call-1", output="sunny")
        assert tr.output == "sunny"


class TestProviderMessage:
    def test_create_text(self) -> None:
        msg = ProviderMessage(role=Role.USER, content="hello")
        assert msg.content == "hello"
        assert msg.role == Role.USER

    def test_create_assistant_with_tool_calls(self) -> None:
        tc = ToolCallRequest(id="tc1", name="search")
        msg = ProviderMessage(role=Role.ASSISTANT, content="", tool_calls=[tc])
        assert len(msg.tool_calls) == 1


class TestProviderUsage:
    def test_defaults(self) -> None:
        usage = ProviderUsage()
        assert usage.total_tokens == 0

    def test_ratio(self) -> None:
        usage = ProviderUsage(prompt_tokens=100, completion_tokens=50)
        assert usage.ratio == 0.5

    def test_ratio_zero(self) -> None:
        usage = ProviderUsage()
        assert usage.ratio == 0.0


class TestProviderRequest:
    def test_empty(self) -> None:
        req = ProviderRequest()
        assert req.message_count == 0
        assert req.max_tokens == 4096

    def test_with_messages(self) -> None:
        msg = ProviderMessage(role=Role.USER, content="hi")
        req = ProviderRequest(messages=[msg], temperature=0.5)
        assert req.message_count == 1
        assert req.temperature == 0.5


class TestProviderResponse:
    def test_empty(self) -> None:
        resp = ProviderResponse.empty()
        assert resp.content == ""

    def test_to_dict(self) -> None:
        resp = ProviderResponse(content="hello", usage=ProviderUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15))
        assert resp.content == "hello"
        assert resp.usage.total_tokens == 15


class TestStreamingChunk:
    def test_create(self) -> None:
        chunk = StreamingChunk(content="hello", index=0)
        assert chunk.content == "hello"
        assert chunk.index == 0


class TestStreamingResult:
    def test_empty(self) -> None:
        result = StreamingResult()
        assert result.full_content == ""

    def test_accumulated(self) -> None:
        chunks = [StreamingChunk(content="a"), StreamingChunk(content="b")]
        result = StreamingResult(full_content="ab", chunks=chunks)
        assert result.full_content == "ab"
        assert len(result.chunks) == 2


class TestCapabilities:
    def test_constants(self) -> None:
        assert Capabilities.STREAMING == "streaming"
        assert Capabilities.TOOL_CALLING == "tool_calling"
        assert Capabilities.VISION == "vision"
