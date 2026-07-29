"""Tests for OpenAI provider __init__ exports."""
from __future__ import annotations

from app.provider.openai import OpenAICompatibleProvider


class TestOpenAIInit:
    def test_exports(self) -> None:
        assert OpenAICompatibleProvider is not None
