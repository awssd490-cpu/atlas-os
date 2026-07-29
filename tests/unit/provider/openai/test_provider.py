"""Tests for OpenAICompatibleProvider."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock
from typing import Any

import pytest

from app.provider.config import ProviderConfig
from app.provider.models import (
    Capabilities,
    ProviderMessage,
    ProviderRequest,
    Role,
    StopReason,
)
from app.provider.openai import OpenAICompatibleProvider
from app.transport.models import HttpResponse


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_provider(
    config_dict: dict[str, Any] | None = None,
) -> OpenAICompatibleProvider:
    """Create a provider with the given config dict."""
    cfg = ProviderConfig.from_dict(
        config_dict or {"api_key": "sk-test", "model": "gpt-4"},
        name="openai-test",
    )
    return OpenAICompatibleProvider(config=cfg)


def _mock_transport(
    provider: OpenAICompatibleProvider,
    *,
    json_body: dict[str, Any] | None = None,
) -> MagicMock:
    """Replace the transport with a mock and return the mock.

    The mock's ``send_json`` is an ``AsyncMock`` that records calls.
    """
    transport = MagicMock()
    transport.is_initialized = True

    # AsyncMock for send_json — records calls and returns a default response
    default_body = json_body or {
        "id": "chatcmpl-mock",
        "object": "chat.completion",
        "created": 1677652288,
        "model": "gpt-4",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": "Mock response"},
            "finish_reason": "stop",
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    send_json = AsyncMock(return_value=HttpResponse(
        status_code=200,
        text=json.dumps(default_body),
    ))
    transport.send_json = send_json

    # Mock stream as empty
    async def _empty_stream(*args: Any, **kwargs: Any) -> AsyncIterator[bytes]:
        return
        yield b""  # pragma: no cover

    transport.stream = _empty_stream

    provider._transport = transport
    return transport


# ---------------------------------------------------------------------------
# Provider info tests
# ---------------------------------------------------------------------------


class TestOpenAIProviderInfo:
    def test_provider_info(self) -> None:
        provider = _make_provider()
        info = provider.provider_info
        assert info.metadata.name == "openai-compatible"
        assert info.metadata.version == "1.0.0"
        assert info.has_capability(Capabilities.STREAMING)
        assert info.has_capability(Capabilities.JSON_MODE)
        assert info.has_capability(Capabilities.SYSTEM_PROMPTS)
        assert info.has_capability(Capabilities.TEMPERATURE)
        assert info.has_capability(Capabilities.STOP_SEQUENCES)
        assert info.has_capability(Capabilities.FUNCTION_CALLING)
        assert info.has_capability("vision") is False
        assert info.has_capability("audio") is False
        assert info.has_capability("embedding") is False

    def test_supports_properties(self) -> None:
        provider = _make_provider()
        assert provider.supports_streaming is True
        assert provider.supports_json_mode is True
        assert provider.supports_tool_calling is False
        assert provider.supports_vision is False
        assert provider.supports_audio is False

    def test_capability_checks(self) -> None:
        provider = _make_provider()
        assert provider.supports_capability("streaming") is True
        assert provider.supports_capability("json_mode") is True
        assert provider.supports_capability("vision") is False


# ---------------------------------------------------------------------------
# Lifecycle tests
# ---------------------------------------------------------------------------


class TestOpenAIProviderLifecycle:
    async def test_generate_requires_initialize(self) -> None:
        provider = _make_provider()
        with pytest.raises(RuntimeError, match="not initialized"):
            await provider.generate(ProviderRequest())

    async def test_stream_requires_initialize(self) -> None:
        provider = _make_provider()
        with pytest.raises(RuntimeError, match="not initialized"):
            async for _ in provider.stream(ProviderRequest()):
                pass

    async def test_count_tokens_before_init(self) -> None:
        provider = _make_provider()
        count = await provider.count_tokens(ProviderRequest())
        assert count >= 0

    async def test_health_check_before_init(self) -> None:
        provider = _make_provider()
        result = await provider.health_check()
        assert result is False

    async def test_initialize_and_shutdown(self) -> None:
        provider = _make_provider()
        await provider.initialize()
        assert provider._transport is not None
        assert provider._transport.is_initialized is True
        await provider.shutdown()
        assert provider._transport is None


# ---------------------------------------------------------------------------
# Generate tests
# ---------------------------------------------------------------------------


class TestOpenAIProviderGenerate:
    async def test_generate_success(self) -> None:
        provider = _make_provider()
        _mock_transport(provider)

        response = await provider.generate(
            ProviderRequest(
                messages=[ProviderMessage(role=Role.USER, content="Hello")],
            )
        )
        assert response.content == "Mock response"
        assert response.stop_reason == StopReason.STOP
        assert response.usage.total_tokens == 15

    async def test_generate_with_system_prompt(self) -> None:
        provider = _make_provider()
        transport = _mock_transport(provider)

        await provider.generate(
            ProviderRequest(
                system="Be helpful.",
                messages=[ProviderMessage(role=Role.USER, content="Hi")],
            )
        )

        # Verify the request body had system prompt
        transport.send_json.assert_called_once()
        # send_json(url, body) → positional args in call_args[0]
        body = transport.send_json.call_args[0][1]
        assert len(body["messages"]) == 2
        assert body["messages"][0]["role"] == "system"
        assert body["messages"][0]["content"] == "Be helpful."

    async def test_generate_with_model(self) -> None:
        provider = _make_provider({"api_key": "sk-test", "model": "gpt-4-turbo"})
        transport = _mock_transport(provider)

        await provider.generate(
            ProviderRequest(
                messages=[ProviderMessage(role=Role.USER, content="Hi")],
            )
        )

        transport.send_json.assert_called_once()
        body = transport.send_json.call_args[0][1]  # second positional arg (json_body)
        assert body["model"] == "gpt-4-turbo"

    async def test_generate_with_model_in_metadata(self) -> None:
        provider = _make_provider({"api_key": "sk-test", "model": "gpt-4"})
        transport = _mock_transport(provider)

        await provider.generate(
            ProviderRequest(
                messages=[ProviderMessage(role=Role.USER, content="Hi")],
                metadata={"model": "gpt-4o"},
            )
        )

        transport.send_json.assert_called_once()
        body = transport.send_json.call_args[0][1]  # second positional arg (json_body)
        # Metadata model overrides config model
        assert body["model"] == "gpt-4o"

    async def test_generate_error_400(self) -> None:
        from app.provider.errors import InvalidRequestError

        provider = _make_provider()
        transport = MagicMock()
        provider._transport = transport
        transport.is_initialized = True
        transport.send_json = AsyncMock(return_value=HttpResponse(
            status_code=400,
            text=json.dumps({
                "error": {"code": "invalid_prompt", "message": "Prompt too long"},
            }),
        ))

        with pytest.raises(InvalidRequestError):
            await provider.generate(
                ProviderRequest(messages=[ProviderMessage(role=Role.USER, content="Hi")])
            )

    async def test_generate_error_401(self) -> None:
        from app.provider.errors import AuthenticationError

        provider = _make_provider()
        transport = MagicMock()
        provider._transport = transport
        transport.is_initialized = True
        transport.send_json = AsyncMock(return_value=HttpResponse(
            status_code=401, text="Unauthorized",
        ))

        with pytest.raises(AuthenticationError):
            await provider.generate(
                ProviderRequest(messages=[ProviderMessage(role=Role.USER, content="Hi")])
            )

    async def test_generate_error_403(self) -> None:
        from app.provider.errors import AuthenticationError

        provider = _make_provider()
        transport = MagicMock()
        provider._transport = transport
        transport.is_initialized = True
        transport.send_json = AsyncMock(return_value=HttpResponse(
            status_code=403, text="Forbidden",
        ))

        with pytest.raises(AuthenticationError):
            await provider.generate(
                ProviderRequest(messages=[ProviderMessage(role=Role.USER, content="Hi")])
            )

    async def test_generate_error_429(self) -> None:
        from app.provider.errors import RateLimitError

        provider = _make_provider()
        transport = MagicMock()
        provider._transport = transport
        transport.is_initialized = True
        transport.send_json = AsyncMock(return_value=HttpResponse(
            status_code=429, text="Rate limited",
        ))

        with pytest.raises(RateLimitError):
            await provider.generate(
                ProviderRequest(messages=[ProviderMessage(role=Role.USER, content="Hi")])
            )

    async def test_generate_error_500(self) -> None:
        from app.provider.errors import ProviderUnavailableError

        provider = _make_provider()
        transport = MagicMock()
        provider._transport = transport
        transport.is_initialized = True
        transport.send_json = AsyncMock(return_value=HttpResponse(
            status_code=500, text="Server Error",
        ))

        with pytest.raises(ProviderUnavailableError):
            await provider.generate(
                ProviderRequest(messages=[ProviderMessage(role=Role.USER, content="Hi")])
            )


# ---------------------------------------------------------------------------
# Streaming tests
# ---------------------------------------------------------------------------


class TestOpenAIProviderStreaming:
    async def _stream_chunks(
        self,
        chunks: list[bytes],
    ) -> list[dict[str, Any]]:
        """Helper: create a mock transport that yields SSE chunks."""
        provider = _make_provider()
        transport = MagicMock()
        provider._transport = transport
        transport.is_initialized = True

        async def _mock_stream(*args: Any, **kwargs: Any) -> AsyncIterator[bytes]:
            for chunk in chunks:
                yield chunk

        transport.stream = _mock_stream

        results: list[dict[str, Any]] = []
        async for chunk in provider.stream(
            ProviderRequest(messages=[ProviderMessage(role=Role.USER, content="Hi")])
        ):
            results.append({
                "content": chunk.content,
                "stop_reason": chunk.stop_reason,
                "usage": chunk.usage,
            })
        return results

    async def test_stream_content_chunks(self) -> None:
        """Stream returns content chunks."""
        chunks = [b'data: {"choices":[{"index":0,"delta":{"content":"Hello"}}]}\n\n']
        results = await self._stream_chunks(chunks)
        assert len(results) == 1
        assert results[0]["content"] == "Hello"

    async def test_stream_multiple_chunks(self) -> None:
        """Multiple SSE chunks yield multiple results."""
        chunks = [
            b'data: {"choices":[{"index":0,"delta":{"content":"Hello"}}]}\n\n',
            b'data: {"choices":[{"index":0,"delta":{"content":" World"}}]}\n\n',
            b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ]
        results = await self._stream_chunks(chunks)
        assert len(results) == 3
        assert results[0]["content"] == "Hello"
        assert results[1]["content"] == " World"
        assert results[2]["stop_reason"] == StopReason.STOP

    async def test_stream_done_signal(self) -> None:
        """[DONE] signal stops streaming."""
        chunks = [
            b'data: {"choices":[{"index":0,"delta":{"content":"Hello"}}]}\n\n',
            b"data: [DONE]\n\n",
            b'data: {"choices":[{"index":0,"delta":{"content":"IGNORED"}}]}\n\n',
        ]
        results = await self._stream_chunks(chunks)
        assert len(results) == 1
        assert results[0]["content"] == "Hello"

    async def test_stream_empty_chunks_skipped(self) -> None:
        """Empty data lines are skipped."""
        chunks = [
            b"data: \n\n",
            b'data: {"choices":[{"index":0,"delta":{"content":"Hello"}}]}\n\n',
        ]
        results = await self._stream_chunks(chunks)
        assert len(results) == 1
        assert results[0]["content"] == "Hello"

    async def test_stream_no_events(self) -> None:
        """No matching events yields no results."""
        chunks = [b"event: ping\n\n"]
        results = await self._stream_chunks(chunks)
        assert len(results) == 0

    async def test_stream_usage_chunk(self) -> None:
        """Final usage chunk is captured."""
        chunks = [
            b'data: {"choices":[{"index":0,"delta":{"content":"Done"}}]}\n\n',
            b'data: {"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}\n\n',
            b"data: [DONE]\n\n",
        ]
        results = await self._stream_chunks(chunks)
        assert len(results) == 2
        assert results[0]["content"] == "Done"
        assert results[1]["usage"].total_tokens == 15

    async def test_stream_json_decode_error(self) -> None:
        """Malformed JSON in a chunk is skipped."""
        chunks = [
            b"data: {invalid json}\n\n",
            b'data: {"choices":[{"index":0,"delta":{"content":"Hello"}}]}\n\n',
        ]
        results = await self._stream_chunks(chunks)
        assert len(results) == 1
        assert results[0]["content"] == "Hello"

    async def test_stream_split_sse_lines(self) -> None:
        """Multiple SSE events in a single transport chunk."""
        chunks = [
            b'data: {"choices":[{"index":0,"delta":{"content":"Hello"}}]}\n'
            b'data: {"choices":[{"index":0,"delta":{"content":" World"}}]}\n'
            b"data: [DONE]\n\n",
        ]
        results = await self._stream_chunks(chunks)
        assert len(results) == 2
        assert results[0]["content"] == "Hello"
        assert results[1]["content"] == " World"

    async def test_stream_cross_boundary(self) -> None:
        """SSE line split across transport chunks."""
        chunks = [
            b'data: {"choices":[{"index":0',
            b',"delta":{"content":"Hello"}}]}\n\n',
        ]
        results = await self._stream_chunks(chunks)
        assert len(results) == 1
        assert results[0]["content"] == "Hello"


# ---------------------------------------------------------------------------
# JSON mode tests
# ---------------------------------------------------------------------------


class TestOpenAIProviderJsonMode:
    async def test_json_object_mode(self) -> None:
        """JSON object response format."""
        provider = _make_provider()
        transport = _mock_transport(provider)

        await provider.generate(
            ProviderRequest(
                messages=[ProviderMessage(role=Role.USER, content="Return JSON")],
                metadata={"response_format": {"type": "json_object"}},
            )
        )
        transport.send_json.assert_called_once()
        body = transport.send_json.call_args[0][1]  # second positional arg (json_body)
        assert body["response_format"] == {"type": "json_object"}

    async def test_json_schema_mode(self) -> None:
        """JSON schema response format."""
        provider = _make_provider()
        transport = _mock_transport(provider)
        schema = {
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
            },
        }

        await provider.generate(
            ProviderRequest(
                messages=[ProviderMessage(role=Role.USER, content="Return schema")],
                metadata={"response_format": schema},
            )
        )
        transport.send_json.assert_called_once()
        body = transport.send_json.call_args[0][1]  # second positional arg (json_body)
        assert body["response_format"]["type"] == "json_schema"


# ---------------------------------------------------------------------------
# Configuration tests
# ---------------------------------------------------------------------------


class TestOpenAIProviderConfig:
    def test_default_config(self) -> None:
        provider = _make_provider({"api_key": "sk-test"})
        assert provider._config.credentials.api_key == "sk-test"
        assert provider._config.model == ""
        assert provider._config.generation.max_tokens == 4096

    def test_configured_model(self) -> None:
        provider = _make_provider({"api_key": "sk-test", "model": "gpt-4o"})
        assert provider._config.model == "gpt-4o"

    def test_generation_defaults(self) -> None:
        provider = _make_provider({
            "api_key": "sk-test",
            "temperature": 0.3,
            "max_tokens": 2048,
            "top_p": 0.9,
        })
        assert provider._config.generation.temperature == 0.3
        assert provider._config.generation.max_tokens == 2048
        assert provider._config.generation.top_p == 0.9

    def test_frequency_penalty_config(self) -> None:
        provider = _make_provider({
            "api_key": "sk-test",
            "frequency_penalty": 0.5,
        })
        assert provider._config.generation.frequency_penalty == 0.5

    def test_presence_penalty_config(self) -> None:
        provider = _make_provider({
            "api_key": "sk-test",
            "presence_penalty": 0.3,
        })
        assert provider._config.generation.presence_penalty == 0.3

    def test_openai_endpoint(self) -> None:
        """OpenAI base URL."""
        provider = _make_provider({
            "api_key": "sk-test",
            "endpoint.base_url": "https://api.openai.com/v1",
        })
        assert provider._request_url == "https://api.openai.com/v1/chat/completions"

    def test_openrouter_endpoint(self) -> None:
        """OpenRouter base URL."""
        provider = _make_provider({
            "api_key": "sk-test",
            "endpoint.base_url": "https://openrouter.ai/api/v1",
        })
        assert provider._request_url == "https://openrouter.ai/api/v1/chat/completions"

    def test_groq_endpoint(self) -> None:
        """Groq base URL."""
        provider = _make_provider({
            "api_key": "gsk-test",
            "endpoint.base_url": "https://api.groq.com/openai/v1",
        })
        assert provider._request_url == "https://api.groq.com/openai/v1/chat/completions"

    def test_localhost_endpoint(self) -> None:
        """Local LLM (LM Studio, Ollama, etc.) base URL."""
        provider = _make_provider({
            "api_key": "",
            "endpoint.base_url": "http://localhost:11434/v1",
        })
        assert provider._request_url == "http://localhost:11434/v1/chat/completions"

    def test_together_endpoint(self) -> None:
        """Together AI base URL."""
        provider = _make_provider({
            "api_key": "sk-test",
            "endpoint.base_url": "https://api.together.xyz/v1",
        })
        assert provider._request_url == "https://api.together.xyz/v1/chat/completions"

    def test_custom_api_path(self) -> None:
        """Custom api_path does not get overridden."""
        provider = _make_provider({
            "api_key": "sk-test",
            "endpoint.base_url": "https://custom.example.com",
            "endpoint.api_path": "/custom/chat",
        })
        assert provider._request_url == "https://custom.example.com/custom/chat"


# ---------------------------------------------------------------------------
# Token counting tests
# ---------------------------------------------------------------------------


class TestOpenAIProviderTokenCount:
    async def test_count_tokens_empty(self) -> None:
        provider = _make_provider()
        count = await provider.count_tokens(ProviderRequest())
        assert count == 1  # minimum

    async def test_count_tokens_with_content(self) -> None:
        provider = _make_provider()
        req = ProviderRequest(
            system="Hello",
            messages=[ProviderMessage(role=Role.USER, content="World")],
        )
        count = await provider.count_tokens(req)
        # "Hello" + "World" = 10 chars / 4 ≈ 2
        assert count == 2

    async def test_count_tokens_with_long_content(self) -> None:
        provider = _make_provider()
        req = ProviderRequest(
            messages=[ProviderMessage(role=Role.USER, content="A" * 100)],
        )
        count = await provider.count_tokens(req)
        assert count == 25  # 100 / 4


# ---------------------------------------------------------------------------
# Health check tests
# ---------------------------------------------------------------------------


class TestOpenAIProviderHealthCheck:
    async def test_health_check_uninitialized(self) -> None:
        provider = _make_provider()
        result = await provider.health_check()
        assert result is False
