"""Tests for agent error types."""

from __future__ import annotations

import pytest

from app.agent.errors import (
    AgentError,
    IterationLimitExceeded,
    ProviderExecutionError,
    ToolCallLimitExceeded,
)


class TestAgentErrorHierarchy:
    def test_agent_error_is_base(self) -> None:
        assert issubclass(IterationLimitExceeded, AgentError)
        assert issubclass(ToolCallLimitExceeded, AgentError)
        assert issubclass(ProviderExecutionError, AgentError)

    def test_agent_error_has_code(self) -> None:
        error = AgentError("test")
        assert error.code == "AGENT_ERROR"

    def test_iteration_limit_exceeded(self) -> None:
        error = IterationLimitExceeded(max_iterations=10)
        assert "10" in str(error)
        assert error.code == "AGENT_ITERATION_LIMIT_EXCEEDED"
        assert error.details["max_iterations"] == 10

    def test_iteration_limit_exceeded_default(self) -> None:
        error = IterationLimitExceeded()
        assert "maximum" in str(error).lower()

    def test_tool_call_limit_exceeded(self) -> None:
        error = ToolCallLimitExceeded(max_tool_calls=50)
        assert "50" in str(error)
        assert error.code == "AGENT_TOOL_CALL_LIMIT_EXCEEDED"
        assert error.details["max_tool_calls"] == 50

    def test_tool_call_limit_exceeded_default(self) -> None:
        error = ToolCallLimitExceeded()
        assert "maximum" in str(error).lower()

    def test_provider_execution_error(self) -> None:
        original = ValueError("API returned 500")
        error = ProviderExecutionError(
            message="Provider failed",
            original_exception=original,
            details={"iteration": 1},
        )
        assert error.code == "AGENT_PROVIDER_EXECUTION_ERROR"
        assert error.details["iteration"] == 1
        assert "ValueError" in error.details["original_error"]

    def test_provider_execution_error_default_message(self) -> None:
        error = ProviderExecutionError()
        assert "execution failed" in str(error).lower()
