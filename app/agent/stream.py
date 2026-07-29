"""Agent streaming interface.

Provides an async generator that yields ``AgentEvent`` objects as the
reasoning loop progresses, then returns the final ``ProviderResponse``.

Usage::

    async for event in runtime.stream(messages):
        if event.event_type == "provider_response_received":
            print(f"Got response: {event.content_length} chars")

    # After the loop, the final response is available on runtime
    # (or you can capture it by observing AgentCompletedEvent).
"""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from app.agent.config import AgentConfig
from app.agent.events import AgentEvent
from app.provider.models import ProviderMessage, ProviderResponse
from app.provider.provider import Provider
from app.tools.runtime import ToolRuntime


class AgentStream:
    """Wraps an ``AgentRuntime`` to yield events as an async generator.

    The stream yields ``AgentEvent`` objects as they occur during the
    reasoning loop.  After iteration completes, the final
    ``ProviderResponse`` is also accessible.

    Usage::

        stream = AgentStream(runtime)
        async for event in stream.run(messages):
            print(event.event_type, event.iteration)
    """

    def __init__(
        self,
        runtime: Any,
    ) -> None:
        self._runtime = runtime
        self._final_response: ProviderResponse | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        messages: list[ProviderMessage],
        *,
        config: AgentConfig | None = None,
        system: str = "",
        **request_kwargs: Any,
    ) -> AsyncIterator[AgentEvent]:
        """Run the agent loop as an event stream.

        Yields ``AgentEvent`` objects for each phase of execution.
        After the generator exits, ``final_response`` is set to the
        provider's final response.

        Args:
            messages: The initial conversation messages.
            config: Optional per-run configuration override.
            system: Optional system prompt.
            **request_kwargs: Additional keyword arguments forwarded to
                ``ProviderRequest``.

        Yields:
            ``AgentEvent`` objects as the loop progresses.
        """
        # Delegate to the runtime's stream method
        async for event in self._runtime.stream(
            messages,
            config=config,
            system=system,
            **request_kwargs,
        ):
            if isinstance(event, AgentEvent):
                yield event

        self._final_response = self._runtime._last_stream_response

    @property
    def final_response(self) -> ProviderResponse | None:
        """The final ``ProviderResponse`` after streaming completes.

        ``None`` if ``run()`` has not been called or completed.
        """
        return self._final_response
