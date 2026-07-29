"""Agent error hierarchy.

Every agent-related error derives from ``AgentError`` so callers
can catch all agent issues distinctly from provider, tool, or
platform errors.
"""

from __future__ import annotations

from typing import Any

from app.core.errors import AtlasError


class AgentError(AtlasError):
    """Base class for all agent errors."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "AGENT_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class IterationLimitExceeded(AgentError):
    """Raised when the agent exceeds the maximum number of reasoning iterations."""

    def __init__(
        self,
        max_iterations: int = 0,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        msg = (
            f"Agent exceeded maximum iterations ({max_iterations})"
            if max_iterations
            else "Agent exceeded maximum iterations"
        )
        merged_details = dict(details or {})
        merged_details["max_iterations"] = max_iterations
        super().__init__(msg, code="AGENT_ITERATION_LIMIT_EXCEEDED", details=merged_details)


class ToolCallLimitExceeded(AgentError):
    """Raised when the agent exceeds the maximum number of tool calls."""

    def __init__(
        self,
        max_tool_calls: int = 0,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        msg = (
            f"Agent exceeded maximum tool calls ({max_tool_calls})"
            if max_tool_calls
            else "Agent exceeded maximum tool calls"
        )
        merged_details = dict(details or {})
        merged_details["max_tool_calls"] = max_tool_calls
        super().__init__(msg, code="AGENT_TOOL_CALL_LIMIT_EXCEEDED", details=merged_details)


class ProviderExecutionError(AgentError):
    """Raised when the provider fails during an agent reasoning step.

    Wraps the original provider exception while preserving context.
    """

    def __init__(
        self,
        message: str = "Provider execution failed during agent loop",
        *,
        original_exception: Exception | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        merged_details = dict(details or {})
        if original_exception is not None:
            merged_details["original_error"] = (
                f"{type(original_exception).__name__}: {original_exception}"
            )
        super().__init__(
            message,
            code="AGENT_PROVIDER_EXECUTION_ERROR",
            details=merged_details,
        )
