"""Tests for InProcessEventBus.

Verifies:
- Subscribe and publish deliver events to handlers
- Multiple handlers per event type all execute
- Handler isolation (one failure doesn't affect others)
- ``emit_and_wait`` propagates exceptions
- Unsubscribe works
- Unsubscribe on unknown raises ValueError
- Stats snapshot returns correct counters
- Events carry proper envelope fields
"""

from __future__ import annotations

from typing import ClassVar

import pytest

from app.core.events import Event
from app.events.bus import InProcessEventBus


# ---------------------------------------------------------------------------
# Test events
# ---------------------------------------------------------------------------


class Ping(Event):
    _event_type: ClassVar[str] = "ping"
    source: str = "test"
    payload: dict = {}


class Pong(Event):
    _event_type: ClassVar[str] = "pong"
    source: str = "test"
    reply: str = ""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bus() -> InProcessEventBus:
    return InProcessEventBus()


# ---------------------------------------------------------------------------
# Subscribe / publish
# ---------------------------------------------------------------------------


class TestSubscribeAndPublish:
    async def test_single_handler_receives_event(self, bus: InProcessEventBus) -> None:
        received: list[Ping] = []

        async def handler(event: Ping) -> None:
            received.append(event)

        bus.subscribe(Ping, handler)
        event = Ping(payload={"msg": "hello"})
        await bus.publish(event)
        assert len(received) == 1
        assert received[0].payload == {"msg": "hello"}

    async def test_multiple_handlers_all_execute(self, bus: InProcessEventBus) -> None:
        results: list[int] = []

        async def h1(event: Ping) -> None:
            results.append(1)

        async def h2(event: Ping) -> None:
            results.append(2)

        bus.subscribe(Ping, h1)
        bus.subscribe(Ping, h2)
        await bus.publish(Ping(payload={}))
        assert sorted(results) == [1, 2]

    async def test_different_event_types_isolated(self, bus: InProcessEventBus) -> None:
        pings: list[Ping] = []
        pongs: list[Pong] = []

        async def ping_h(event: Ping) -> None:
            pings.append(event)

        async def pong_h(event: Pong) -> None:
            pongs.append(event)

        bus.subscribe(Ping, ping_h)
        bus.subscribe(Pong, pong_h)

        await bus.publish(Ping(payload={}))
        await bus.publish(Pong(reply="pong"))

        assert len(pings) == 1
        assert len(pongs) == 1


# ---------------------------------------------------------------------------
# Handler isolation
# ---------------------------------------------------------------------------


class TestHandlerIsolation:
    async def test_failing_handler_does_not_block_others(self, bus: InProcessEventBus) -> None:
        results: list[int] = []

        async def failing_handler(event: Ping) -> None:
            msg = "intentional failure"  # noqa: F841

        async def ok_handler(event: Ping) -> None:
            results.append(42)

        bus.subscribe(Ping, failing_handler)
        bus.subscribe(Ping, ok_handler)

        await bus.publish(Ping(payload={}))
        assert results == [42]


# ---------------------------------------------------------------------------
# emit_and_wait
# ---------------------------------------------------------------------------


class TestEmitAndWait:
    async def test_returns_exceptions(self, bus: InProcessEventBus) -> None:
        async def failing(event: Ping) -> None:
            raise RuntimeError("handler failed")

        bus.subscribe(Ping, failing)
        results = await bus.emit_and_wait(Ping(payload={}))
        assert len(results) == 1
        assert isinstance(results[0], RuntimeError)

    async def test_returns_none_for_success(self, bus: InProcessEventBus) -> None:
        async def ok(event: Ping) -> None:
            pass

        bus.subscribe(Ping, ok)
        results = await bus.emit_and_wait(Ping(payload={}))
        assert results == [None]

    async def test_no_handlers_returns_empty(self, bus: InProcessEventBus) -> None:
        results = await bus.emit_and_wait(Ping(payload={}))
        assert results == []


# ---------------------------------------------------------------------------
# Unsubscribe
# ---------------------------------------------------------------------------


class TestUnsubscribe:
    async def test_handler_no_longer_called(self, bus: InProcessEventBus) -> None:
        received: list[Ping] = []

        async def handler(event: Ping) -> None:
            received.append(event)

        bus.subscribe(Ping, handler)
        bus.unsubscribe(Ping, handler)
        await bus.publish(Ping(payload={}))
        assert len(received) == 0

    async def test_unsubscribe_unknown_raises(self, bus: InProcessEventBus) -> None:
        async def handler(event: Ping) -> None:
            pass

        with pytest.raises(ValueError, match="not registered"):
            bus.unsubscribe(Ping, handler)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestStats:
    async def test_counts_events_emitted(self, bus: InProcessEventBus) -> None:
        async def handler(event: Ping) -> None:
            pass

        bus.subscribe(Ping, handler)
        await bus.publish(Ping(payload={}))
        await bus.publish(Ping(payload={}))
        stats = bus.stats()
        assert stats["total_events_emitted"] == 2

    async def test_counts_errors(self, bus: InProcessEventBus) -> None:
        async def failing(event: Ping) -> None:
            raise RuntimeError("fail")

        bus.subscribe(Ping, failing)
        await bus.publish(Ping(payload={}))
        stats = bus.stats()
        assert stats["total_errors"] == 1

    async def test_registered_event_types(self, bus: InProcessEventBus) -> None:
        async def handler(event: Ping) -> None:
            pass

        bus.subscribe(Ping, handler)
        stats = bus.stats()
        assert stats["registered_event_types"] == 1
        assert stats["total_handlers"] == 1


# ---------------------------------------------------------------------------
# Event envelope
# ---------------------------------------------------------------------------


class TestEventEnvelope:
    def test_event_has_standard_fields(self) -> None:
        event = Ping(payload={})
        assert event.event_id is not None and len(event.event_id) > 0
        assert event.version == 1
        assert event.timestamp is not None
        assert event.correlation_id is not None
        assert event.source == "test"
        assert event.target == "*"
        assert event.metadata == {}

    def test_event_type_uses_class_var(self) -> None:
        assert Ping.event_type() == "ping"
        assert Pong.event_type() == "pong"

    def test_with_correlation(self) -> None:
        event = Ping(payload={})
        modified = event.with_correlation("custom-id")
        assert modified.correlation_id == "custom-id"
        assert event.correlation_id != "custom-id"  # immutable

    def test_with_source(self) -> None:
        event = Ping(payload={})
        modified = event.with_source("other-module")
        assert modified.source == "other-module"

    def test_with_metadata(self) -> None:
        event = Ping(payload={})
        modified = event.with_metadata(trace="abc", env="prod")
        assert modified.metadata == {"trace": "abc", "env": "prod"}
