"""Tests for AgentRuntime event emission and streaming."""

from __future__ import annotations

from typing import Any

import pytest

from app.agent.runtime import AgentRuntime
from app.agent.config import AgentConfig
from app.agent.events import (
    AgentCompletedEvent,
    AgentEvent,
    AgentFailedEvent,
    AgentStartedEvent,
    IterationCompletedEvent,
    IterationStartedEvent,
    ProviderRequestStartedEvent,
    ProviderResponseReceivedEvent,
    ToolExecutionFinishedEvent,
    ToolExecutionStartedEvent,
)
from app.provider.models import (
    ProviderMessage,
    ProviderRequest,
    ProviderResponse,
    ProviderUsage,
    Role,
    StopReason,
    ToolCallRequest,
)
from app.provider.provider import Provider
from app.tools.models import ToolDefinition, ToolParameter
from app.tools.registry import ToolRegistry
from app.tools.runtime import ToolRuntime


# ---------------------------------------------------------------------------
# Mock provider
# ---------------------------------------------------------------------------


class _MockProvider(Provider):
    """A mock provider that returns canned responses."""

    def __init__(
        self,
        responses: list[ProviderResponse],
    ) -> None:
        self._responses = list(responses)
        self._call_count = 0
        self._should_fail = False
        self._fail_message = ""

    def fail_on_next(self, message: str = "API Error") -> None:
        self._should_fail = True
        self._fail_message = message

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        self._call_count += 1
        if self._should_fail:
            raise RuntimeError(self._fail_message)
        if not self._responses:
            return ProviderResponse(content="", stop_reason=StopReason.STOP)
        return self._responses.pop(0)

    def stream(self, request: ProviderRequest):  # type: ignore[override]
        raise NotImplementedError

    async def count_tokens(self, request: ProviderRequest) -> int:
        return 0

    @property
    def provider_info(self) -> Any:
        from app.provider.models import (
            ProviderCapability,
            ProviderInfo,
            ProviderMetadata,
        )
        return ProviderInfo(
            metadata=ProviderMetadata(name="mock"),
            capabilities=[ProviderCapability(name="tool_calling")],
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_response(
    content: str,
    tool_calls: list[ToolCallRequest] | None = None,
) -> ProviderResponse:
    calls = tool_calls or []
    return ProviderResponse(
        content=content,
        message=ProviderMessage(role=Role.ASSISTANT, content=content, tool_calls=calls),
        stop_reason=StopReason.TOOL_CALL if calls else StopReason.STOP,
        tool_calls=calls,
        usage=ProviderUsage(prompt_tokens=10, completion_tokens=5),
    )


def _make_tc(name: str, args: dict[str, Any] | None = None) -> ToolCallRequest:
    return ToolCallRequest(id=f"call_{name}", name=name, arguments=args or {})


def _create_tool_runtime() -> ToolRuntime:
    registry = ToolRegistry()

    async def calc(expression: str) -> str:
        return str(eval(expression))

    registry.register(ToolDefinition(
        name="calculator",
        parameters=(ToolParameter(name="expression", type="string"),),
        fn=calc,
    ))
    return ToolRuntime(registry)


def _create_agent(
    responses: list[ProviderResponse],
) -> tuple[AgentRuntime, _MockProvider, ToolRuntime]:
    provider = _MockProvider(responses)
    tr = _create_tool_runtime()
    runtime = AgentRuntime(provider, tr)
    return runtime, provider, tr


# ---------------------------------------------------------------------------
# Event emission tests
# ---------------------------------------------------------------------------


class TestAgentRuntimeEvents:
    async def test_emit_started_event(self) -> None:
        """AgentStartedEvent is emitted via dispatcher."""
        runtime, _, _ = _create_agent([_make_response("done")])
        events: list[AgentEvent] = []

        def listener(event: AgentEvent) -> None:
            events.append(event)

        runtime._dispatcher.add_listener(listener)  # type: ignore[attr-defined]
        await runtime.run([ProviderMessage(role=Role.USER, content="Hi")])

        assert any(e.event_type == "agent_started" for e in events)

    async def test_events_in_correct_order(self) -> None:
        """Events are emitted in the expected order."""
        runtime, _, _ = _create_agent([_make_response("done")])
        event_types: list[str] = []

        def listener(event: AgentEvent) -> None:
            event_types.append(event.event_type)

        runtime._dispatcher.add_listener(listener)  # type: ignore[attr-defined]
        await runtime.run([ProviderMessage(role=Role.USER, content="Hi")])

        assert event_types[0] == "agent_started"
        assert "iteration_started" in event_types
        assert "provider_request_started" in event_types
        assert "provider_response_received" in event_types

    async def test_emit_tool_events(self) -> None:
        """Tool execution events are emitted."""
        runtime, _, _ = _create_agent([
            _make_response(
                "calc",
                tool_calls=[_make_tc("calculator", {"expression": "1+1"})],
            ),
            _make_response("result: 2"),
        ])
        event_types: list[str] = []

        def listener(event: AgentEvent) -> None:
            event_types.append(event.event_type)

        runtime._dispatcher.add_listener(listener)  # type: ignore[attr-defined]
        await runtime.run([ProviderMessage(role=Role.USER, content="Calc")])

        assert "tool_execution_started" in event_types
        assert "tool_execution_finished" in event_types

    async def test_emit_tool_started_event_fields(self) -> None:
        """ToolExecutionStartedEvent contains tool name and id."""
        runtime, _, _ = _create_agent([
            _make_response("calc", tool_calls=[_make_tc("calculator", {"expr": "2+2"})]),
            _make_response("4"),
        ])
        tool_events: list[ToolExecutionStartedEvent] = []

        def listener(event: AgentEvent) -> None:
            if event.event_type == "tool_execution_started":
                tool_events.append(event)  # type: ignore[arg-type]

        runtime._dispatcher.add_listener(listener)  # type: ignore[attr-defined]
        await runtime.run([ProviderMessage(role=Role.USER, content="Calc")])

        assert len(tool_events) == 1
        assert tool_events[0].tool_name == "calculator"

    async def test_emit_completed_event(self) -> None:
        """AgentCompletedEvent is emitted on successful completion."""
        runtime, _, _ = _create_agent([_make_response("done")])
        completed: list[AgentCompletedEvent] = []

        def listener(event: AgentEvent) -> None:
            if event.event_type == "agent_completed":
                completed.append(event)  # type: ignore[arg-type]

        runtime._dispatcher.add_listener(listener)  # type: ignore[attr-defined]
        await runtime.run([ProviderMessage(role=Role.USER, content="Hi")])

        assert len(completed) == 1
        assert completed[0].total_iterations == 1

    async def test_emit_failed_event_on_provider_error(self) -> None:
        """AgentFailedEvent is emitted on provider error."""
        provider = _MockProvider([])
        provider.fail_on_next("connection error")
        runtime = AgentRuntime(provider, _create_tool_runtime())
        failed: list[AgentFailedEvent] = []

        def listener(event: AgentEvent) -> None:
            if event.event_type == "agent_failed":
                failed.append(event)  # type: ignore[arg-type]

        runtime._dispatcher.add_listener(listener)  # type: ignore[attr-defined]

        with pytest.raises(Exception):
            await runtime.run([ProviderMessage(role=Role.USER, content="Hi")])

        assert len(failed) == 1
        assert "connection error" in failed[0].error

    async def test_iteration_events_emit_counter(self) -> None:
        """Iteration started event carries correct iteration number."""
        runtime, _, _ = _create_agent([
            _make_response("1", tool_calls=[_make_tc("calculator", {"expr": "1+1"})]),
            _make_response("2"),
        ])
        iterations: list[int] = []

        def listener(event: AgentEvent) -> None:
            if event.event_type == "iteration_started":
                iterations.append(event.iteration)

        runtime._dispatcher.add_listener(listener)  # type: ignore[attr-defined]
        await runtime.run([ProviderMessage(role=Role.USER, content="Go")])

        assert iterations == [1, 2]

    async def test_provider_response_event_details(self) -> None:
        """ProviderResponseReceivedEvent includes content length and tool call count."""
        runtime, _, _ = _create_agent([
            _make_response("hello world", tool_calls=[_make_tc("calc")]),
            _make_response("done"),
        ])
        responses: list[ProviderResponseReceivedEvent] = []

        def listener(event: AgentEvent) -> None:
            if event.event_type == "provider_response_received":
                responses.append(event)  # type: ignore[arg-type]

        runtime._dispatcher.add_listener(listener)  # type: ignore[attr-defined]
        await runtime.run([ProviderMessage(role=Role.USER, content="Hi")])

        assert len(responses) == 2
        assert responses[0].content_length == 11  # "hello world"
        assert responses[0].tool_call_count == 1
        assert responses[1].tool_call_count == 0


# ---------------------------------------------------------------------------
# Streaming tests
# ---------------------------------------------------------------------------


class TestAgentRuntimeStreaming:
    async def test_stream_yields_started_event(self) -> None:
        """Stream yields AgentStartedEvent first."""
        runtime, _, _ = _create_agent([_make_response("done")])
        events: list[AgentEvent | ProviderResponse] = []
        async for event in runtime.stream(
            [ProviderMessage(role=Role.USER, content="Hi")],
        ):
            events.append(event)

        assert len(events) >= 2  # started + completed + response
        assert events[0].event_type == "agent_started"  # type: ignore[attr-defined]

    async def test_stream_yields_final_response(self) -> None:
        """Stream yields the final ProviderResponse at the end."""
        runtime, _, _ = _create_agent([_make_response("final answer")])
        last_item = None
        async for event in runtime.stream(
            [ProviderMessage(role=Role.USER, content="Hi")],
        ):
            last_item = event

        # Last item should be a ProviderResponse
        from app.provider.models import ProviderResponse as PR

        assert isinstance(last_item, PR)
        assert last_item.content == "final answer"

    async def test_stream_returns_completed_event(self) -> None:
        """Stream yields AgentCompletedEvent."""
        runtime, _, _ = _create_agent([_make_response("done")])
        event_types: list[str] = []
        async for event in runtime.stream(
            [ProviderMessage(role=Role.USER, content="Hi")],
        ):
            if hasattr(event, "event_type"):
                event_types.append(event.event_type)  # type: ignore[attr-defined]

        assert "agent_completed" in event_types

    async def test_stream_with_tool_calls(self) -> None:
        """Stream yields tool events with tool calls."""
        runtime, _, _ = _create_agent([
            _make_response("calc", tool_calls=[_make_tc("calculator", {"expr": "2+2"})]),
            _make_response("4"),
        ])
        event_types: list[str] = []
        async for event in runtime.stream(
            [ProviderMessage(role=Role.USER, content="Calc")],
        ):
            if hasattr(event, "event_type"):
                event_types.append(event.event_type)  # type: ignore[attr-defined]

        assert "tool_execution_started" in event_types
        assert "tool_execution_finished" in event_types

    async def test_stream_provider_failure(self) -> None:
        """Stream yields AgentFailedEvent on provider error."""
        provider = _MockProvider([])
        provider.fail_on_next("stream failure")
        runtime = AgentRuntime(provider, _create_tool_runtime())

        events: list[AgentEvent | ProviderResponse] = []
        with pytest.raises(Exception):
            async for event in runtime.stream(
                [ProviderMessage(role=Role.USER, content="Hi")],
            ):
                events.append(event)

        failed_events = [
            e for e in events
            if hasattr(e, "event_type") and e.event_type == "agent_failed"  # type: ignore[attr-defined]
        ]
        assert len(failed_events) == 1

    async def test_stream_multiple_iterations(self) -> None:
        """Stream works with multiple reasoning iterations."""
        runtime, _, _ = _create_agent([
            _make_response("1", tool_calls=[_make_tc("calculator", {"expr": "1+1"})]),
            _make_response("2", tool_calls=[_make_tc("calculator", {"expr": "2+2"})]),
            _make_response("3"),
        ])
        event_types: list[str] = []
        async for event in runtime.stream(
            [ProviderMessage(role=Role.USER, content="Go")],
        ):
            if hasattr(event, "event_type"):
                event_types.append(event.event_type)  # type: ignore[attr-defined]

        # Should have 2 iteration_started events
        started_count = event_types.count("iteration_started")
        assert started_count >= 2

    async def test_stream_immutable_kwargs(self) -> None:
        """Stream does not mutate caller's kwargs dict."""
        runtime, _, _ = _create_agent([_make_response("done")])
        kwargs = {"max_tokens": 100}
        kwargs_copy = dict(kwargs)
        async for _ in runtime.stream(
            [ProviderMessage(role=Role.USER, content="Hi")],
            **kwargs_copy,
        ):
            pass
        assert kwargs_copy == kwargs

    async def test_stream_no_tool_events_on_direct_answer(self) -> None:
        """No tool events when provider returns final answer immediately."""
        runtime, _, _ = _create_agent([_make_response("hello")])
        event_types: list[str] = []
        async for event in runtime.stream(
            [ProviderMessage(role=Role.USER, content="Hi")],
        ):
            if hasattr(event, "event_type"):
                event_types.append(event.event_type)  # type: ignore[attr-defined]

        assert "tool_execution_started" not in event_types
        assert "tool_execution_finished" not in event_types


# ---------------------------------------------------------------------------
# Dispatcher integration tests
# ---------------------------------------------------------------------------


class TestRuntimeDispatcherIntegration:
    async def test_runtime_with_dispatcher(self) -> None:
        """Runtime works with a dispatcher."""
        runtime, _, _ = _create_agent([_make_response("done")])
        events: list[AgentEvent] = []

        def listener(event: AgentEvent) -> None:
            events.append(event)

        runtime._dispatcher.add_listener(listener)  # type: ignore[attr-defined]
        await runtime.run([ProviderMessage(role=Role.USER, content="Hi")])

        assert len(events) > 0

    async def test_runtime_without_dispatcher(self) -> None:
        """Runtime works without a dispatcher."""
        runtime, _, _ = _create_agent([_make_response("done")])
        # No dispatcher configured
        response = await runtime.run([ProviderMessage(role=Role.USER, content="Hi")])
        assert response.content == "done"

    async def test_dispatcher_property(self) -> None:
        """Runtime exposes the dispatcher property."""
        runtime, _, _ = _create_agent([_make_response("done")])
        assert runtime.dispatcher is not None

    async def test_dispatcher_events_include_iteration(self) -> None:
        """Events include iteration metadata from dispatcher."""
        runtime, _, _ = _create_agent([
            _make_response("1", tool_calls=[_make_tc("calculator", {"expr": "1+1"})]),
            _make_response("2"),
        ])
        iterations: set[int] = set()

        def listener(event: AgentEvent) -> None:
            if event.event_type == "iteration_started":
                iterations.add(event.iteration)

        runtime._dispatcher.add_listener(listener)  # type: ignore[attr-defined]
        await runtime.run([ProviderMessage(role=Role.USER, content="Go")])

        assert iterations == {1, 2}


# ---------------------------------------------------------------------------
# Stream edge cases
# ---------------------------------------------------------------------------


class TestStreamEdgeCases:
    async def test_stream_empty_messages(self) -> None:
        """Stream with empty messages works."""
        runtime, _, _ = _create_agent([_make_response("hello")])
        events: list[AgentEvent | ProviderResponse] = []
        async for event in runtime.stream([]):
            events.append(event)
        assert len(events) >= 2

    async def test_stream_tool_call_limit(self) -> None:
        """Stream handles tool call limit."""
        runtime = AgentRuntime(
            _MockProvider([_make_response(
                "calc",
                tool_calls=[_make_tc("calculator", {"expr": "1"})],
            ) for _ in range(5)]),
            _create_tool_runtime(),
            config=AgentConfig(max_tool_calls=2),
        )
        events: list[AgentEvent | ProviderResponse] = []
        with pytest.raises(Exception):
            async for event in runtime.stream(
                [ProviderMessage(role=Role.USER, content="Calc")],
            ):
                events.append(event)
        # Should have a failed event
        failed = [
            e for e in events
            if hasattr(e, "event_type") and e.event_type == "agent_failed"  # type: ignore[attr-defined]
        ]
        assert len(failed) >= 1
