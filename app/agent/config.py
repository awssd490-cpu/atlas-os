"""Agent configuration.

Validated configuration for the ``AgentRuntime``.

Controls loop limits and error behaviour.

Contains no provider-specific settings — provider identity is derived
from the provider instance itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AgentConfig:
    """Configuration for the ``AgentRuntime`` reasoning loop.

    Attributes:
        max_iterations: Maximum number of reasoning iterations before
            raising ``IterationLimitExceeded``.  Default 10.
        max_tool_calls: Maximum total tool calls across all iterations
            before raising ``ToolCallLimitExceeded``.  Default 100.
        raise_on_iteration_limit: If ``True`` (default), raise
            ``IterationLimitExceeded`` when the iteration limit is hit.
            If ``False``, return the partial response.
        return_partial_response: If ``True`` and the iteration limit is
            hit (with ``raise_on_iteration_limit=False``), return the last
            provider response even if it contains tool calls.  Default
            ``False``.
    """

    max_iterations: int = 10
    max_tool_calls: int = 100
    raise_on_iteration_limit: bool = True
    return_partial_response: bool = False

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.max_iterations < 1:
            raise ValueError(
                f"max_iterations must be >= 1, got {self.max_iterations}"
            )
        if self.max_tool_calls < 1:
            raise ValueError(
                f"max_tool_calls must be >= 1, got {self.max_tool_calls}"
            )

    @classmethod
    def default(cls) -> "AgentConfig":
        """Return the default configuration."""
        return cls()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a dictionary."""
        return {
            "max_iterations": self.max_iterations,
            "max_tool_calls": self.max_tool_calls,
            "raise_on_iteration_limit": self.raise_on_iteration_limit,
            "return_partial_response": self.return_partial_response,
        }
