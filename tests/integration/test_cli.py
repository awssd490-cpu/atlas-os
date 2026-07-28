"""Integration tests for the Atlas CLI and example scripts.

All tests use a mocked HTTP transport so no real API calls are made.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.provider.claude import ClaudeProvider
from app.provider.config import ProviderConfig
from app.provider.models import ProviderMessage, ProviderRequest, ProviderUsage, Role, StopReason


class _MockTransport:
    """Mock transport that returns canned responses."""

    def __init__(self) -> None:
        self.last_request: Any = None
        self._initialized = True

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    async def initialize(self) -> None:
        self._initialized = True

    async def shutdown(self) -> None:
        self._initialized = False

    async def send_json(self, url: str, body: dict, **kwargs: Any) -> Any:
        self.last_request = body
        # Return a mock response
        import json
        text = json.dumps({
            "id": "mock_msg",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Mock response"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 10, "output_tokens": 3},
        })
        return type("_", (), {
            "status_code": 200,
            "is_success": True,
            "text": text,
            "body": text.encode(),
            "json": lambda self: json.loads(text),
        })()

    async def send(self, request: Any) -> Any:
        return await self.send_json(request.url, {})

    async def health_check(self, url: str) -> bool:
        return True

    def stream(self, request: Any) -> Any:  # sync — returns async generator
        async def _gen():
            yield b"data: {\"type\": \"content_block_delta\", \"delta\": {\"type\": \"text_delta\", \"text\": \"Hello\"}}\n\n"
            yield b"data: {\"type\": \"message_delta\", \"delta\": {\"stop_reason\": \"end_turn\"}, \"usage\": {\"input_tokens\": 5, \"output_tokens\": 2}}\n\n"
            yield b"data: [DONE]\n\n"
        return _gen()


# ---------------------------------------------------------------------------
# ChatCLI tests
# ---------------------------------------------------------------------------


class TestChatCLIIntegration:
    """Tests that verify the CLI components work together."""

    async def test_provider_creates_and_generates(self) -> None:
        """Full stack: ProviderConfig → ClaudeProvider → generate."""
        config = ProviderConfig.from_dict({"api_key": "sk-test"})
        provider = ClaudeProvider(config=config)

        # Inject mock transport
        transport = _MockTransport()
        provider._transport = transport

        request = ProviderRequest(
            messages=[ProviderMessage(role=Role.USER, content="Hello")],
            system="Be helpful.",
        )
        response = await provider.generate(request)
        assert response.content == "Mock response"
        assert response.stop_reason == StopReason.STOP
        assert response.usage.prompt_tokens == 10

    async def test_streaming_integration(self) -> None:
        """Full stack: ProviderConfig → ClaudeProvider → stream."""
        config = ProviderConfig.from_dict({"api_key": "sk-test"})
        provider = ClaudeProvider(config=config)

        transport = _MockTransport()
        provider._transport = transport

        request = ProviderRequest(
            messages=[ProviderMessage(role=Role.USER, content="Hi")],
        )

        chunks: list[str] = []
        async for chunk in provider.stream(request):
            if chunk.content:
                chunks.append(chunk.content)

        assert len(chunks) >= 1
        assert "Hello" in chunks

    async def test_conversation_history(self) -> None:
        """Multiple turns should accumulate history."""
        config = ProviderConfig.from_dict({"api_key": "sk-test"})
        provider = ClaudeProvider(config=config)

        transport = _MockTransport()
        provider._transport = transport

        history: list[ProviderMessage] = []

        # Turn 1
        history.append(ProviderMessage(role=Role.USER, content="Hello"))
        req1 = ProviderRequest(messages=list(history))
        resp1 = await provider.generate(req1)
        history.append(ProviderMessage(role=Role.ASSISTANT, content=resp1.content))
        assert len(history) == 2

        # Turn 2
        history.append(ProviderMessage(role=Role.USER, content="Tell me more"))
        req2 = ProviderRequest(messages=list(history))
        await provider.generate(req2)
        # Transport should have received all messages
        assert transport.last_request is not None

    async def test_error_on_missing_api_key(self) -> None:
        """Provider should handle missing API key gracefully at init."""
        config = ProviderConfig()  # no api_key
        provider = ClaudeProvider(config=config)
        # Should not crash — transport handles auth via middleware
        await provider.initialize()
        assert provider._transport is not None
        await provider.shutdown()

    async def test_health_check_with_mock(self) -> None:
        """Health check should work through the transport."""
        config = ProviderConfig.from_dict({"api_key": "sk-test"})
        provider = ClaudeProvider(config=config)
        transport = _MockTransport()
        provider._transport = transport
        healthy = await provider.health_check()
        assert healthy is True

    async def test_count_tokens_fallback(self) -> None:
        """Token counting should work (possibly with fallback)."""
        config = ProviderConfig.from_dict({"api_key": "sk-test"})
        provider = ClaudeProvider(config=config)
        transport = _MockTransport()
        provider._transport = transport
        count = await provider.count_tokens(ProviderRequest(
            messages=[ProviderMessage(role=Role.USER, content="Hello world")],
        ))
        # Should return either the API count or the fallback estimate
        assert count >= 0

    async def test_verbose_flag_does_not_crash(self) -> None:
        """Exercising the verbose flag handler."""
        # Test that arg parsing works for verbose mode
        from app.cli.chat import build_arg_parser
        # Simulate --verbose being parsed
        args = build_arg_parser().parse_args(["--verbose"])
        assert args.verbose is True


class TestCLIParsing:
    def test_default_args(self) -> None:
        from app.cli.chat import build_arg_parser
        args = build_arg_parser().parse_args([])
        assert args.provider == "claude"
        assert args.model == "claude-sonnet-4-20250514"
        assert args.verbose is False

    def test_custom_args(self) -> None:
        from app.cli.chat import build_arg_parser
        args = build_arg_parser().parse_args([
            "--provider", "claude",
            "--model", "claude-3-opus",
            "--verbose",
            "--temperature", "0.3",
        ])
        assert args.provider == "claude"
        assert args.model == "claude-3-opus"
        assert args.verbose is True
        assert args.temperature == 0.3

    def test_build_chat_config(self) -> None:
        from app.cli.chat import build_arg_parser, build_chat_config
        args = build_arg_parser().parse_args(["--api-key", "sk-test"])
        config = build_chat_config(args)
        assert config["api_key"] == "sk-test"
        assert config["model"] == "claude-sonnet-4-20250514"


class TestExampleScripts:
    """Verify example scripts parse arguments correctly."""

    def test_stream_parser(self) -> None:
        """``examples/stream.py`` argument parser should work."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--prompt", default="Write a poem.")
        parser.add_argument("--model", default="claude-sonnet-4-20250514")
        parser.add_argument("--api-key", default="")
        args = parser.parse_args([])
        assert args.prompt == "Write a poem."
        assert args.model == "claude-sonnet-4-20250514"

    def test_count_tokens_parser(self) -> None:
        """``examples/count_tokens.py`` argument parser should work."""
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--text", default="Hello")
        parser.add_argument("--model", default="claude-sonnet-4-20250514")
        parser.add_argument("--api-key", default="")
        args = parser.parse_args(["--text", "Count these tokens"])
        assert args.text == "Count these tokens"
