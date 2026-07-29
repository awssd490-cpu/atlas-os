"""Tests for AgentConfig."""

from __future__ import annotations

import pytest

from app.agent.config import AgentConfig


class TestAgentConfig:
    def test_default_config(self) -> None:
        config = AgentConfig.default()
        assert config.max_iterations == 10
        assert config.max_tool_calls == 100
        assert config.raise_on_iteration_limit is True
        assert config.return_partial_response is False

    def test_custom_config(self) -> None:
        config = AgentConfig(
            max_iterations=5,
            max_tool_calls=20,
            raise_on_iteration_limit=False,
            return_partial_response=True,
        )
        assert config.max_iterations == 5
        assert config.max_tool_calls == 20
        assert config.raise_on_iteration_limit is False
        assert config.return_partial_response is True

    def test_invalid_max_iterations_raises(self) -> None:
        with pytest.raises(ValueError, match="max_iterations"):
            AgentConfig(max_iterations=0)

    def test_invalid_max_tool_calls_raises(self) -> None:
        with pytest.raises(ValueError, match="max_tool_calls"):
            AgentConfig(max_tool_calls=0)

    def test_to_dict(self) -> None:
        config = AgentConfig(max_iterations=3, max_tool_calls=10)
        d = config.to_dict()
        assert d["max_iterations"] == 3
        assert d["max_tool_calls"] == 10
        assert d["raise_on_iteration_limit"] is True

    def test_frozen(self) -> None:
        config = AgentConfig()
        with pytest.raises(AttributeError):
            config.max_iterations = 20  # type: ignore[misc]
