"""Agent event models and dispatcher.

Provides a lightweight event system for observing ``AgentRuntime``
execution in real time.  Applications can register listeners to
receive events as the reasoning loop progresses.

The dispatcher is an optional dependency — ``AgentRuntime`` works
perfectly without it.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from app.tools.models import ToolCall, ToolResult


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------

# Event type string constants
AGENT_STARTED = "agent_started"
ITERATION_STARTED = "iteration_started"
PROVIDER_REQUEST_STARTED = "provider_request_started"
PROVIDER_RESPONSE_RECEIVED = "provider_response_received"
TOOL_EXECUTION_STARTED = "tool_execution_started"
TOOL_EXECUTION_FINISHED = "tool_execution_finished"
ITERATION_COMPLETED = "iteration_completed"
AGENT_COMPLETED = "agent_completed"
AGENT_FAILED = "agent_failed"
AGENT_TOOL_CALL_LIMIT = "agent_tool_call_limit"
AGENT_ITERATION_LIMIT = "agent_iteration_limit"


# ---------------------------------------------------------------------------
# Event models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AgentEvent:
    """Base event emitted by the ``AgentRuntime``.

    Attributes:
        event_type: A string identifying the event type.
        timestamp: Unix timestamp when the event was created.
        iteration: The current iteration number (0-indexed base).
        metadata: Optional structured data attached to the event.
    """

    event_type: str = ""
    timestamp: float = 0.0
    iteration: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Set timestamp to current time if not provided."""
        # Handle the case where timestamp is explicitly 0.0 (default)
        # but we want to set it to time.time() if not explicitly provided.
        # We use object.__setattr__ because the dataclass is frozen.
        if self.timestamp == 0.0:
            object.__setattr__(self, "timestamp", __import__("time").time())


@dataclass(frozen=True)
class AgentStartedEvent(AgentEvent):
    """Emitted when ``run()`` or ``stream()`` begins."""

    event_type: str = AGENT_STARTED

    def __init__(
        self,
        *,
        iteration: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            event_type=AGENT_STARTED,
            timestamp=time.time(),
            iteration=iteration,
            metadata=metadata or {},
        )


@dataclass(frozen=True)
class IterationStartedEvent(AgentEvent):
    """Emitted at the start of each reasoning iteration."""

    event_type: str = ITERATION_STARTED

    def __init__(
        self,
        *,
        iteration: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            event_type=ITERATION_STARTED,
            timestamp=time.time(),
            iteration=iteration,
            metadata=metadata or {},
        )


@dataclass(frozen=True)
class ProviderRequestStartedEvent(AgentEvent):
    """Emitted before a provider request is made."""

    event_type: str = PROVIDER_REQUEST_STARTED
    message_count: int = 0

    def __init__(
        self,
        *,
        iteration: int,
        message_count: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            event_type=PROVIDER_REQUEST_STARTED,
            timestamp=time.time(),
            iteration=iteration,
            metadata=metadata or {},
        )
        object.__setattr__(self, "message_count", message_count)


@dataclass(frozen=True)
class ProviderResponseReceivedEvent(AgentEvent):
    """Emitted when a provider response is received.

    Contains the response content and tool call info (not the full
    response object, to keep events lightweight).
    """

    event_type: str = PROVIDER_RESPONSE_RECEIVED
    content_length: int = 0
    tool_call_count: int = 0

    def __init__(
        self,
        *,
        iteration: int,
        content_length: int,
        tool_call_count: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            event_type=PROVIDER_RESPONSE_RECEIVED,
            timestamp=time.time(),
            iteration=iteration,
            metadata=metadata or {},
        )
        object.__setattr__(self, "content_length", content_length)
        object.__setattr__(self, "tool_call_count", tool_call_count)


@dataclass(frozen=True)
class ToolExecutionStartedEvent(AgentEvent):
    """Emitted before a tool is executed."""

    event_type: str = TOOL_EXECUTION_STARTED
    tool_name: str = ""
    tool_call_id: str = ""

    def __init__(
        self,
        *,
        iteration: int,
        tool_name: str,
        tool_call_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            event_type=TOOL_EXECUTION_STARTED,
            timestamp=time.time(),
            iteration=iteration,
            metadata=metadata or {},
        )
        object.__setattr__(self, "tool_name", tool_name)
        object.__setattr__(self, "tool_call_id", tool_call_id)


@dataclass(frozen=True)
class ToolExecutionFinishedEvent(AgentEvent):
    """Emitted after a tool finishes execution."""

    event_type: str = TOOL_EXECUTION_FINISHED
    tool_name: str = ""
    tool_call_id: str = ""
    duration_ms: float = 0.0
    success: bool = True

    def __init__(
        self,
        *,
        iteration: int,
        tool_name: str,
        tool_call_id: str,
        duration_ms: float = 0.0,
        success: bool = True,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            event_type=TOOL_EXECUTION_FINISHED,
            timestamp=time.time(),
            iteration=iteration,
            metadata=metadata or {},
        )
        object.__setattr__(self, "tool_name", tool_name)
        object.__setattr__(self, "tool_call_id", tool_call_id)
        object.__setattr__(self, "duration_ms", duration_ms)
        object.__setattr__(self, "success", success)


@dataclass(frozen=True)
class IterationCompletedEvent(AgentEvent):
    """Emitted after an iteration completes (tool results appended)."""

    event_type: str = ITERATION_COMPLETED
    tool_calls_in_iteration: int = 0

    def __init__(
        self,
        *,
        iteration: int,
        tool_calls_in_iteration: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            event_type=ITERATION_COMPLETED,
            timestamp=time.time(),
            iteration=iteration,
            metadata=metadata or {},
        )
        object.__setattr__(self, "tool_calls_in_iteration", tool_calls_in_iteration)


@dataclass(frozen=True)
class AgentCompletedEvent(AgentEvent):
    """Emitted when the agent run completes successfully."""

    event_type: str = AGENT_COMPLETED
    total_iterations: int = 0
    total_tool_calls: int = 0
    total_provider_requests: int = 0

    def __init__(
        self,
        *,
        iteration: int,
        total_iterations: int,
        total_tool_calls: int,
        total_provider_requests: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            event_type=AGENT_COMPLETED,
            timestamp=time.time(),
            iteration=iteration,
            metadata=metadata or {},
        )
        object.__setattr__(self, "total_iterations", total_iterations)
        object.__setattr__(self, "total_tool_calls", total_tool_calls)
        object.__setattr__(self, "total_provider_requests", total_provider_requests)


@dataclass(frozen=True)
class AgentFailedEvent(AgentEvent):
    """Emitted when the agent run fails with an exception."""

    event_type: str = AGENT_FAILED
    error: str = ""
    error_type: str = ""

    def __init__(
        self,
        *,
        error: str,
        error_type: str = "",
        iteration: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            event_type=AGENT_FAILED,
            timestamp=time.time(),
            iteration=iteration,
            metadata=metadata or {},
        )
        object.__setattr__(self, "error", error)
        object.__setattr__(self, "error_type", error_type)


# ---------------------------------------------------------------------------
# Event dispatcher
# ---------------------------------------------------------------------------

Listener = Callable[[AgentEvent], Any]


class AgentEventDispatcher:
    """Dispatches agent events to registered listeners.

    Supports both sync and async listeners.  Listener order is preserved.

    The dispatcher is an optional component — ``AgentRuntime`` works
    without it.

    Usage::

        dispatcher = AgentEventDispatcher()

        def on_event(event: AgentEvent) -> None:
            print(event.event_type, event.iteration)

        async def on_event_async(event: AgentEvent) -> None:
            await log_event(event)

        dispatcher.add_listener(on_event)
        dispatcher.add_listener(on_event_async)

        dispatcher.emit(event)
    """

    def __init__(self) -> None:
        self._listeners: list[Listener] = []

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add_listener(self, listener: Listener) -> None:
        """Register an event listener.

        Args:
            listener: A callable that accepts an ``AgentEvent``.
                Can be sync or async.

        Raises:
            ValueError: If *listener* is already registered.
        """
        if listener in self._listeners:
            raise ValueError(f"Listener {listener!r} is already registered")
        self._listeners.append(listener)

    def remove_listener(self, listener: Listener) -> None:
        """Unregister an event listener.

        Args:
            listener: The listener to remove.

        Raises:
            ValueError: If *listener* is not registered.
        """
        if listener not in self._listeners:
            raise ValueError(f"Listener {listener!r} is not registered")
        self._listeners.remove(listener)

    def has_listener(self, listener: Listener) -> bool:
        """Check if a listener is registered."""
        return listener in self._listeners

    @property
    def listener_count(self) -> int:
        """Return the number of registered listeners."""
        return len(self._listeners)

    def clear(self) -> None:
        """Remove all registered listeners."""
        self._listeners.clear()

    # ------------------------------------------------------------------
    # Emission
    # ------------------------------------------------------------------

    async def emit(self, event: AgentEvent) -> None:
        """Emit an event to all registered listeners.

        Sync listeners are called with ``await`` if they are coroutine
        functions, or called directly otherwise.  Async listeners are
        always awaited.

        Listener order is preserved.  If a listener raises, the
        exception propagates to the caller.

        Args:
            event: The ``AgentEvent`` to emit.
        """
        for listener in self._listeners:
            if hasattr(listener, "__call__"):
                result = listener(event)
                if hasattr(result, "__await__"):
                    await result
            else:
                listener(event)
