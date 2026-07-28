"""Tests for the Provider ABC and a test implementation."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from app.provider.errors import CapabilityNotSupportedError
from app.provider.models import (
    Capabilities,
    ProviderCapability,
    ProviderInfo,
    ProviderMetadata,
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


# ---------------------------------------------------------------------------
# Test provider implementation
# ---------------------------------------------------------------------------


class _EchoProvider(Provider):
    """A test provider that echoes back the input."""

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs
        self._initialized = False
        self._shutdown = False

    async def initialize(self) -> None:
        self._initialized = True

    async def shutdown(self) -> None:
        self._shutdown = True

    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        content = request.messages[-1].content if request.messages else ""
        return ProviderResponse(
            content=f"Echo: {content}",
            message=self._make_assistant(f"Echo: {content}"),
            stop_reason=StopReason.STOP,
            usage=ProviderUsage(prompt_tokens=10, completion_tokens=5),
        )

    def stream(self, request: ProviderRequest) -> AsyncIterator[StreamingChunk]:
        raise NotImplementedError("Streaming not implemented in test")

    async def count_tokens(self, request: ProviderRequest) -> int:
        return sum(len(m.content) for m in request.messages)

    @property
    def provider_info(self) -> ProviderInfo:
        return ProviderInfo(
            metadata=ProviderMetadata(name="echo", version="1.0", description="Test provider"),
            capabilities=[
                ProviderCapability(name=Capabilities.TEMPERATURE),
                ProviderCapability(name=Capabilities.SYSTEM_PROMPTS),
            ],
        )

    @staticmethod
    def _make_assistant(content: str) -> Any:
        from app.provider.models import ProviderMessage
        return ProviderMessage(role=Role.ASSISTANT, content=content)


class _StreamingProvider(Provider):
    """A test provider that supports streaming."""

    def __init__(self, **kwargs: Any) -> None:
        self._kwargs = kwargs

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        return ProviderResponse.empty()

    async def stream(self, request: ProviderRequest) -> AsyncIterator[StreamingChunk]:
        for i, char in enumerate("hello"):
            yield StreamingChunk(content=char, index=i)

    async def count_tokens(self, request: ProviderRequest) -> int:
        return 0

    @property
    def provider_info(self) -> ProviderInfo:
        return ProviderInfo(
            metadata=ProviderMetadata(name="stream-test"),
            capabilities=[ProviderCapability(name=Capabilities.STREAMING)],
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestProviderInterface:
    async def test_generate(self) -> None:
        provider = _EchoProvider()
        req = ProviderRequest(messages=[ProviderMessage(role=Role.USER, content="hello")])
        resp = await provider.generate(req)
        assert resp.content == "Echo: hello"

    async def test_initialize(self) -> None:
        provider = _EchoProvider()
        assert provider._initialized is False
        await provider.initialize()
        assert provider._initialized is True

    async def test_shutdown(self) -> None:
        provider = _EchoProvider()
        await provider.shutdown()
        assert provider._shutdown is True

    async def test_count_tokens(self) -> None:
        provider = _EchoProvider()
        req = ProviderRequest(messages=[ProviderMessage(role=Role.USER, content="hello world")])
        count = await provider.count_tokens(req)
        assert count == 11

    async def test_health_check(self) -> None:
        provider = _EchoProvider()
        healthy = await provider.health_check()
        assert healthy is True

    async def test_supports_capability(self) -> None:
        provider = _EchoProvider()
        assert provider.supports_capability("temperature") is True
        assert provider.supports_capability("vision") is False

    async def test_assert_capability_raises(self) -> None:
        provider = _EchoProvider()
        with pytest.raises(CapabilityNotSupportedError):
            provider.assert_capability("vision")

    async def test_assert_capability_passes(self) -> None:
        provider = _EchoProvider()
        provider.assert_capability("temperature")  # should not raise

    async def test_convenience_properties(self) -> None:
        provider = _EchoProvider()
        assert provider.supports_tool_calling is False
        assert provider.supports_vision is False
        assert provider.supports_streaming is False
        assert provider.supports_audio is False

    async def test_stream_not_implemented(self) -> None:
        provider = _EchoProvider()
        req = ProviderRequest()
        with pytest.raises(NotImplementedError):
            async for _ in provider.stream(req):
                pass

    async def test_provider_info(self) -> None:
        provider = _EchoProvider()
        info = provider.provider_info
        assert info.metadata.name == "echo"
        assert info.metadata.version == "1.0"

    async def test_health_check_unhealthy(self) -> None:
        """Provider that fails health check."""
        class _UnhealthyProvider(_EchoProvider):
            async def count_tokens(self, request: ProviderRequest) -> int:
                raise RuntimeError("not ready")

        provider = _UnhealthyProvider()
        healthy = await provider.health_check()
        assert healthy is False


class TestStreamingProvider:
    async def test_stream(self) -> None:
        provider = _StreamingProvider()
        chunks: list[str] = []
        async for chunk in provider.stream(ProviderRequest()):
            chunks.append(chunk.content)
        assert "".join(chunks) == "hello"
