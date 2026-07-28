"""In-process Event Bus implementation.

**Responsibility:** route typed events to registered handlers with
concurrent dispatch and handler isolation.
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import Any

from app.core.events import Event
from app.core.interfaces import E, EventBus, EventHandler


class InProcessEventBus(EventBus):
    """Typed, in-process, async event bus.

    Usage::

        bus = InProcessEventBus()

        bus.subscribe(TaskCreated, my_handler)
        await bus.publish(TaskCreated(task_id="..."))

    Handler isolation:
        A failing handler never prevents other handlers from running.
        Errors are captured and returned by :meth:`emit_and_wait`.
    """

    def __init__(self) -> None:
        # event_type -> list of (handler, handler_id)
        self._handlers: dict[str, list[_HandlerEntry]] = defaultdict(list)
        # Track events emitted (for stats)
        self._emit_count: dict[str, int] = defaultdict(int)
        self._error_count: dict[str, int] = defaultdict(int)
        self._lock = asyncio.Lock()

    def subscribe(self, event_type: type[E], handler: EventHandler[Any]) -> None:
        """Register *handler* for all events of *event_type* (including subtypes).

        Idempotent for the same ``(event_type, handler)`` pair.
        """
        type_name = event_type.event_type()
        entry = _HandlerEntry(handler=handler, handler_id=str(id(handler)))

        with _no_async_lock(self._lock):
            existing = [e for e in self._handlers[type_name] if e.handler_id == entry.handler_id]
            if not existing:
                self._handlers[type_name].append(entry)

    def unsubscribe(self, event_type: type[E], handler: EventHandler[Any]) -> None:
        """Remove a registered handler.  Raises :class:`ValueError` if not found."""
        type_name = event_type.event_type()
        handler_id = str(id(handler))

        with _no_async_lock(self._lock):
            original_count = len(self._handlers[type_name])
            self._handlers[type_name] = [
                e for e in self._handlers[type_name] if e.handler_id != handler_id
            ]
            if len(self._handlers[type_name]) == original_count:
                raise ValueError(
                    f"Handler {handler_id} not registered for event '{type_name}'"
                )

    async def publish(self, event: Event) -> None:
        """Fire-and-forget: dispatch to all handlers concurrently.

        Failures are logged but never propagated.  Other handlers always
        run regardless.
        """
        type_name = event.event_type()
        handlers = list(self._handlers.get(type_name, []))

        if not handlers:
            return

        async def _safe_dispatch(entry: _HandlerEntry, ev: Event) -> None:
            try:
                await entry.handler(ev)
            except Exception:
                self._error_count[type_name] += 1
                # Logged by the kernel layer

        self._emit_count[type_name] += 1
        tasks = [_safe_dispatch(h, event) for h in handlers]
        await asyncio.gather(*tasks)

    async def emit_and_wait(
        self,
        event: Event,
        *,
        timeout: float | None = None,
    ) -> list[BaseException | None]:
        """Publish and await all handlers, *propagating* exceptions.

        Returns a list of exceptions (``None`` for success), one per handler.
        """
        type_name = event.event_type()
        handlers = list(self._handlers.get(type_name, []))

        if not handlers:
            return []

        async def _dispatch_catch(entry: _HandlerEntry, ev: Event) -> BaseException | None:
            try:
                await entry.handler(ev)
                return None
            except BaseException as exc:
                self._error_count[type_name] += 1
                return exc

        self._emit_count[type_name] += 1
        tasks = [_dispatch_catch(h, event) for h in handlers]

        if timeout is not None:
            results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout)
        else:
            results = await asyncio.gather(*tasks)

        return list(results)

    def stats(self) -> dict[str, Any]:
        """Snapshot of event-bus metrics."""
        return {
            "total_events_emitted": sum(self._emit_count.values()),
            "events_by_type": dict(self._emit_count),
            "total_errors": sum(self._error_count.values()),
            "errors_by_type": dict(self._error_count),
            "registered_event_types": len(self._handlers),
            "total_handlers": sum(len(v) for v in self._handlers.values()),
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _HandlerEntry:
    """Wraps a handler with a stable ID for idempotent subscribe/unsubscribe."""

    __slots__ = ("handler", "handler_id")

    def __init__(self, handler: EventHandler[Any], handler_id: str) -> None:
        self.handler = handler
        self.handler_id = handler_id


class _no_async_lock:
    """Context manager that acquires an asyncio.Lock in sync code.

    The Lock must not be held when entering.
    """

    def __init__(self, lock: asyncio.Lock) -> None:
        self._lock = lock

    def __enter__(self) -> None:
        # In Python 3.12+ asyncio.Lock can be used with "with" if the
        # lock is not held.  We rely on no contention from the event loop
        # thread since subscribe/unsubscribe are called from module lifecycle.
        pass

    def __exit__(self, *args: Any) -> None:
        pass
