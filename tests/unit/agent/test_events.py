"""Tests for agent event models."""

from __future__ import annotations

import pytest

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


class TestAgentEvent:
    def test_base_event(self) -> None:
        event = AgentEvent(event_type="test", iteration=1)
        assert event.event_type == "test"
        assert event.iteration == 1
        assert event.timestamp > 0
        assert event.metadata == {}

    def test_event_immutable(self) -> None:
        event = AgentEvent(event_type="test")
        with pytest.raises(AttributeError):
            event.event_type = "changed"  # type: ignore[misc]

    def test_event_with_metadata(self) -> None:
        event = AgentEvent(
            event_type="test",
            iteration=2,
            metadata={"key": "value"},
        )
        assert event.metadata["key"] == "value"


class TestAgentStartedEvent:
    def test_create(self) -> None:
        event = AgentStartedEvent()
        assert event.event_type == "agent_started"
        assert event.iteration == 0
        assert event.timestamp > 0

    def test_with_metadata(self) -> None:
        event = AgentStartedEvent(metadata={"msg_count": 5})
        assert event.metadata["msg_count"] == 5


class TestIterationStartedEvent:
    def test_create(self) -> None:
        event = IterationStartedEvent(iteration=1)
        assert event.event_type == "iteration_started"
        assert event.iteration == 1

    def test_defaults(self) -> None:
        event = IterationStartedEvent(iteration=1)
        assert event.metadata == {}


class TestProviderRequestStartedEvent:
    def test_create(self) -> None:
        event = ProviderRequestStartedEvent(iteration=1, message_count=10)
        assert event.event_type == "provider_request_started"
        assert event.iteration == 1
        assert event.message_count == 10

    def test_with_metadata(self) -> None:
        event = ProviderRequestStartedEvent(
            iteration=1,
            message_count=5,
            metadata={"model": "gpt-4"},
        )
        assert event.metadata["model"] == "gpt-4"


class TestProviderResponseReceivedEvent:
    def test_create(self) -> None:
        event = ProviderResponseReceivedEvent(
            iteration=1,
            content_length=42,
            tool_call_count=2,
        )
        assert event.event_type == "provider_response_received"
        assert event.content_length == 42
        assert event.tool_call_count == 2

    def test_no_tool_calls(self) -> None:
        event = ProviderResponseReceivedEvent(
            iteration=1,
            content_length=10,
            tool_call_count=0,
        )
        assert event.tool_call_count == 0


class TestToolExecutionStartedEvent:
    def test_create(self) -> None:
        event = ToolExecutionStartedEvent(
            iteration=1,
            tool_name="calculator",
            tool_call_id="call_1",
        )
        assert event.event_type == "tool_execution_started"
        assert event.tool_name == "calculator"
        assert event.tool_call_id == "call_1"
        assert event.iteration == 1


class TestToolExecutionFinishedEvent:
    def test_success(self) -> None:
        event = ToolExecutionFinishedEvent(
            iteration=1,
            tool_name="calculator",
            tool_call_id="call_1",
            duration_ms=5.0,
            success=True,
        )
        assert event.event_type == "tool_execution_finished"
        assert event.success is True
        assert event.duration_ms == 5.0

    def test_failure(self) -> None:
        event = ToolExecutionFinishedEvent(
            iteration=1,
            tool_name="failing",
            tool_call_id="call_2",
            success=False,
        )
        assert event.success is False


class TestIterationCompletedEvent:
    def test_create(self) -> None:
        event = IterationCompletedEvent(iteration=1, tool_calls_in_iteration=2)
        assert event.event_type == "iteration_completed"
        assert event.tool_calls_in_iteration == 2

    def test_no_tool_calls(self) -> None:
        event = IterationCompletedEvent(iteration=1, tool_calls_in_iteration=0)
        assert event.tool_calls_in_iteration == 0


class TestAgentCompletedEvent:
    def test_create(self) -> None:
        event = AgentCompletedEvent(
            iteration=3,
            total_iterations=3,
            total_tool_calls=5,
            total_provider_requests=4,
        )
        assert event.event_type == "agent_completed"
        assert event.total_iterations == 3
        assert event.total_tool_calls == 5
        assert event.total_provider_requests == 4


class TestAgentFailedEvent:
    def test_create(self) -> None:
        event = AgentFailedEvent(
            error="Something went wrong",
            error_type="ValueError",
            iteration=2,
        )
        assert event.event_type == "agent_failed"
        assert event.error == "Something went wrong"
        assert event.error_type == "ValueError"
        assert event.iteration == 2

    def test_default_error_type(self) -> None:
        event = AgentFailedEvent(error="unknown error")
        assert event.error_type == ""
