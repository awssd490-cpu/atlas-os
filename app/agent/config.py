"""Agent configuration.

Validated configuration for the ``AgentRuntime``.

Controls loop limits, error behaviour, memory settings, and planning.

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
        memory_enabled: If ``True`` (default), retrieve and inject
            relevant memories before every provider request.
        memory_limit: Maximum number of memories to retrieve per
            provider request.  Default 10.
        minimum_memory_score: Minimum importance score for a memory
            to be retrieved (0.0 = no minimum).  Default 0.0.
        inject_memory_as: How to inject memories into the provider
            request.  ``"system"`` (default) prepends memories to the
            system prompt.  ``"assistant"`` appends an assistant message.
            ``"hidden_context"`` stores in request metadata.
        planning_enabled: If ``True`` (default), create and maintain a
            structured execution plan throughout each run.
        max_plan_steps: Maximum number of plan steps allowed.  Default 100.
        allow_replanning: If ``True``, the plan can be extended during
            execution.  Default ``False``.
        streaming_enabled: If ``True`` (default), use provider streaming
            for incremental token output when using ``stream()``.
        chunk_buffer_size: Number of chunks to buffer before yielding.
            Default 1 (yield every chunk).
        emit_thinking_chunks: If ``True``, emit thinking chunks (if the
            provider supports them).  Default ``True``.
        checkpoint_enabled: If ``True`` (default), enable checkpoint
            creation during agent execution.
        checkpoint_frequency: When to create checkpoints automatically.
            ``"manual"`` (default) — only on explicit calls.
            ``"iteration"`` — after each reasoning iteration.
            ``"provider_response"`` — after each provider response.
        parallel_tools_enabled: If ``True`` (default), execute independent
            tool calls concurrently.
        max_parallel_tools: Maximum concurrent tool calls.  Default 8.
        execution_strategy: Strategy for tool execution.
            ``"auto"`` (default) — parallel when independent.
            ``"parallel"`` — always parallel.
            ``"sequential"`` — always sequential (backward compatible).
    """

    max_iterations: int = 10
    max_tool_calls: int = 100
    raise_on_iteration_limit: bool = True
    return_partial_response: bool = False
    memory_enabled: bool = True
    memory_limit: int = 10
    minimum_memory_score: float = 0.0
    inject_memory_as: str = "system"
    planning_enabled: bool = True
    max_plan_steps: int = 100
    allow_replanning: bool = False
    streaming_enabled: bool = True
    chunk_buffer_size: int = 1
    emit_thinking_chunks: bool = True
    checkpoint_enabled: bool = True
    checkpoint_frequency: str = "manual"
    parallel_tools_enabled: bool = True
    max_parallel_tools: int = 8
    execution_strategy: str = "auto"

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
        if self.memory_limit < 1:
            raise ValueError(
                f"memory_limit must be >= 1, got {self.memory_limit}"
            )
        if self.minimum_memory_score < 0.0 or self.minimum_memory_score > 1.0:
            raise ValueError(
                f"minimum_memory_score must be in [0.0, 1.0], got {self.minimum_memory_score}"
            )
        valid_inject = ("system", "assistant", "hidden_context")
        if self.inject_memory_as not in valid_inject:
            raise ValueError(
                f"inject_memory_as must be one of {valid_inject}, "
                f"got {self.inject_memory_as!r}"
            )
        if self.max_plan_steps < 1:
            raise ValueError(
                f"max_plan_steps must be >= 1, got {self.max_plan_steps}"
            )
        if self.chunk_buffer_size < 1:
            raise ValueError(
                f"chunk_buffer_size must be >= 1, got {self.chunk_buffer_size}"
            )
        if self.max_parallel_tools < 1:
            raise ValueError(
                f"max_parallel_tools must be >= 1, got {self.max_parallel_tools}"
            )
        valid_strategies = ("auto", "parallel", "sequential")
        if self.execution_strategy not in valid_strategies:
            raise ValueError(
                f"execution_strategy must be one of {valid_strategies}, "
                f"got {self.execution_strategy!r}"
            )
        valid_freq = ("manual", "iteration", "provider_response")
        if self.checkpoint_frequency not in valid_freq:
            raise ValueError(
                f"checkpoint_frequency must be one of {valid_freq}, "
                f"got {self.checkpoint_frequency!r}"
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
            "memory_enabled": self.memory_enabled,
            "memory_limit": self.memory_limit,
            "minimum_memory_score": self.minimum_memory_score,
            "inject_memory_as": self.inject_memory_as,
            "planning_enabled": self.planning_enabled,
            "max_plan_steps": self.max_plan_steps,
            "allow_replanning": self.allow_replanning,
            "streaming_enabled": self.streaming_enabled,
            "chunk_buffer_size": self.chunk_buffer_size,
            "emit_thinking_chunks": self.emit_thinking_chunks,
            "checkpoint_enabled": self.checkpoint_enabled,
            "checkpoint_frequency": self.checkpoint_frequency,
        }
