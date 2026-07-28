"""Tests for ClaudeProvider."""

from __future__ import annotations

import json
from typing import Any

import pytest

from app.provider.claude import ClaudeProvider
from app.provider.config import ProviderConfig
from app.provider.models import (
    Capabilities,
    ProviderMessage,
    ProviderRequest,
    Role,
    StopReason,
)


class TestClaudeProvider:
    def test_provider_info(self) -> None:
        config = ProviderConfig.from_dict({"api_key": "sk-test"})
        provider = ClaudeProvider(config=config)
        info = provider.provider_info
        assert info.metadata.name == "claude"
        assert info.has_capability(Capabilities.STREAMING)
        assert info.has_capability(Capabilities.SYSTEM_PROMPTS)
        assert info.has_capability(Capabilities.TEMPERATURE)
        assert info.has_capability("vision") is False
        assert info.has_capability("audio") is False
        assert info.has_capability("tool_calling") is False

    def test_supports_streaming_property(self) -> None:
        config = ProviderConfig.from_dict({"api_key": "sk-test"})
        provider = ClaudeProvider(config=config)
        assert provider.supports_streaming is True
        assert provider.supports_vision is False

    async def test_generate_requires_initialize(self) -> None:
        config = ProviderConfig.from_dict({"api_key": "sk-test"})
        provider = ClaudeProvider(config=config)
        with pytest.raises(RuntimeError, match="not initialized"):
            await provider.generate(ProviderRequest())

    async def test_stream_requires_initialize(self) -> None:
        config = ProviderConfig.from_dict({"api_key": "sk-test"})
        provider = ClaudeProvider(config=config)
        with pytest.raises(RuntimeError, match="not initialized"):
            async for _ in provider.stream(ProviderRequest()):
                pass

    async def test_count_tokens_before_init(self) -> None:
        config = ProviderConfig.from_dict({"api_key": "sk-test"})
        provider = ClaudeProvider(config=config)
        count = await provider.count_tokens(ProviderRequest())
        assert count >= 0

    async def test_health_check_before_init(self) -> None:
        config = ProviderConfig.from_dict({"api_key": "sk-test"})
        provider = ClaudeProvider(config=config)
        result = await provider.health_check()
        assert result is False

    def test_capability_checks(self) -> None:
        config = ProviderConfig.from_dict({"api_key": "sk-test"})
        provider = ClaudeProvider(config=config)
        assert provider.supports_capability("streaming") is True
        assert provider.supports_capability("vision") is False
        assert provider.supports_capability("audio") is False


class TestClaudeProviderInit:
    async def test_initialize_and_shutdown(self) -> None:
        config = ProviderConfig.from_dict({"api_key": "sk-test"})
        provider = ClaudeProvider(config=config)
        await provider.initialize()
        # Transport should be created
        assert provider._transport is not None
        assert provider._transport.is_initialized is True
        await provider.shutdown()
        assert provider._transport is None
