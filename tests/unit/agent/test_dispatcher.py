"""Tests for AgentEventDispatcher."""

from __future__ import annotations

from typing import Any

import pytest

from app.agent.events import AgentEvent, AgentEventDispatcher, AgentStartedEvent


class TestAgentEventDispatcher:
    def setup_method(self) -> None:
        self.dispatcher = AgentEventDispatcher()
        self.events: list[AgentEvent] = []

    def _sync_listener(self, event: AgentEvent) -> None:
        self.events.append(event)

    async def _async_listener(self, event: AgentEvent) -> None:
        self.events.append(event)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def test_add_listener(self) -> None:
        self.dispatcher.add_listener(self._sync_listener)
        assert self.dispatcher.listener_count == 1

    def test_add_async_listener(self) -> None:
        self.dispatcher.add_listener(self._async_listener)
        assert self.dispatcher.listener_count == 1

    def test_add_duplicate_raises(self) -> None:
        self.dispatcher.add_listener(self._sync_listener)
        with pytest.raises(ValueError, match="already registered"):
            self.dispatcher.add_listener(self._sync_listener)

    def test_remove_listener(self) -> None:
        self.dispatcher.add_listener(self._sync_listener)
        self.dispatcher.remove_listener(self._sync_listener)
        assert self.dispatcher.listener_count == 0

    def test_remove_nonexistent_raises(self) -> None:
        with pytest.raises(ValueError, match="not registered"):
            self.dispatcher.remove_listener(self._sync_listener)

    def test_has_listener(self) -> None:
        self.dispatcher.add_listener(self._sync_listener)
        assert self.dispatcher.has_listener(self._sync_listener) is True
        assert self.dispatcher.has_listener(self._async_listener) is False

    def test_clear(self) -> None:
        self.dispatcher.add_listener(self._sync_listener)
        self.dispatcher.add_listener(self._async_listener)
        assert self.dispatcher.listener_count == 2
        self.dispatcher.clear()
        assert self.dispatcher.listener_count == 0

    def test_empty_dispatcher(self) -> None:
        assert self.dispatcher.listener_count == 0

    # ------------------------------------------------------------------
    # Emission — sync
    # ------------------------------------------------------------------

    async def test_emit_sync_listener(self) -> None:
        received: list[AgentEvent] = []

        def listener(event: AgentEvent) -> None:
            received.append(event)

        self.dispatcher.add_listener(listener)
        event = AgentStartedEvent()
        await self.dispatcher.emit(event)
        assert len(received) == 1
        assert received[0].event_type == "agent_started"

    async def test_emit_multiple_sync_listeners(self) -> None:
        received1: list[AgentEvent] = []
        received2: list[AgentEvent] = []

        def listener1(event: AgentEvent) -> None:
            received1.append(event)

        def listener2(event: AgentEvent) -> None:
            received2.append(event)

        self.dispatcher.add_listener(listener1)
        self.dispatcher.add_listener(listener2)
        event = AgentStartedEvent()
        await self.dispatcher.emit(event)
        assert len(received1) == 1
        assert len(received2) == 1

    # ------------------------------------------------------------------
    # Emission — async
    # ------------------------------------------------------------------

    async def test_emit_async_listener(self) -> None:
        received: list[AgentEvent] = []

        async def listener(event: AgentEvent) -> None:
            received.append(event)

        self.dispatcher.add_listener(listener)
        await self.dispatcher.emit(AgentStartedEvent())
        assert len(received) == 1

    async def test_emit_mixed_listeners(self) -> None:
        sync_events: list[AgentEvent] = []
        async_events: list[AgentEvent] = []

        def sync_listener(event: AgentEvent) -> None:
            sync_events.append(event)

        async def async_listener(event: AgentEvent) -> None:
            async_events.append(event)

        self.dispatcher.add_listener(sync_listener)
        self.dispatcher.add_listener(async_listener)
        await self.dispatcher.emit(AgentStartedEvent())

        assert len(sync_events) == 1
        assert len(async_events) == 1

    # ------------------------------------------------------------------
    # Listener order
    # ------------------------------------------------------------------

    async def test_listener_order_preserved(self) -> None:
        order: list[int] = []

        def first(event: AgentEvent) -> None:
            order.append(1)

        def second(event: AgentEvent) -> None:
            order.append(2)

        def third(event: AgentEvent) -> None:
            order.append(3)

        self.dispatcher.add_listener(first)
        self.dispatcher.add_listener(second)
        self.dispatcher.add_listener(third)
        await self.dispatcher.emit(AgentStartedEvent())

        assert order == [1, 2, 3]

    # ------------------------------------------------------------------
    # Error propagation
    # ------------------------------------------------------------------

    async def test_listener_exception_propagates(self) -> None:
        def failing_listener(event: AgentEvent) -> None:
            raise ValueError("listener failed")

        self.dispatcher.add_listener(failing_listener)
        with pytest.raises(ValueError, match="listener failed"):
            await self.dispatcher.emit(AgentStartedEvent())

    async def test_listener_exception_stops_chain(self) -> None:
        """Exception in one listener stops subsequent listeners."""
        order: list[int] = []

        def first(event: AgentEvent) -> None:
            order.append(1)

        def failing(event: AgentEvent) -> None:
            raise RuntimeError("fail")

        def third(event: AgentEvent) -> None:
            order.append(3)

        self.dispatcher.add_listener(first)
        self.dispatcher.add_listener(failing)
        self.dispatcher.add_listener(third)

        with pytest.raises(RuntimeError):
            await self.dispatcher.emit(AgentStartedEvent())

        assert order == [1]  # third never called

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    async def test_emit_no_listeners(self) -> None:
        """Emitting with no listeners does nothing."""
        await self.dispatcher.emit(AgentStartedEvent())  # should not raise

    async def test_emit_multiple_events(self) -> None:
        """Multiple emit calls each deliver to all listeners."""
        count: int = 0

        def listener(event: AgentEvent) -> None:
            nonlocal count
            count += 1

        self.dispatcher.add_listener(listener)
        await self.dispatcher.emit(AgentStartedEvent())
        await self.dispatcher.emit(AgentStartedEvent())
        await self.dispatcher.emit(AgentStartedEvent())

        assert count == 3

    async def test_listener_called_with_correct_event(self) -> None:
        """Listener receives the exact event emitted."""
        received: list[AgentEvent] = []

        def listener(event: AgentEvent) -> None:
            received.append(event)

        self.dispatcher.add_listener(listener)
        original = AgentStartedEvent(iteration=5)
        await self.dispatcher.emit(original)

        assert received[0].iteration == 5
        assert received[0].event_type == "agent_started"
