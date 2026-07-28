"""Lightweight kernel-level telemetry service.

**Responsibility:** record and snapshot in-memory counters for module
lifecycle durations, event throughput, errors, and startup timing.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any

from app.core.interfaces import TelemetryService, TelemetrySnapshot


class InMemoryTelemetryService(TelemetryService):
    """In-memory telemetry counter storage.

    Thread-safe via a ``Lock``.  All counters are reset on ``reset()`` or
    on a fresh instance.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: dict[str, _EventMetrics] = {}
        self._module_lifecycle: dict[str, list[_LifecycleRecord]] = defaultdict(list)
        self._errors: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._startup_duration_ms: float = 0.0

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_event_metrics(self, event_type: str, duration_ms: float, success: bool) -> None:
        with self._lock:
            metrics = self._events.setdefault(event_type, _EventMetrics())
            metrics.count += 1
            metrics.total_duration_ms += duration_ms
            if not success:
                metrics.error_count += 1

    def record_module_lifecycle(
        self,
        module_name: str,
        lifecycle_hook: str,
        duration_ms: float,
        success: bool,
    ) -> None:
        with self._lock:
            self._module_lifecycle[module_name].append(
                _LifecycleRecord(hook=lifecycle_hook, duration_ms=duration_ms, success=success)
            )

    def record_error(self, category: str, error_type: str, *, count: int = 1) -> None:
        with self._lock:
            self._errors[category][error_type] += count

    def set_startup_duration(self, duration_ms: float) -> None:
        with self._lock:
            self._startup_duration_ms = duration_ms

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> TelemetrySnapshot:
        """Return an immutable, point-in-time copy of all counters."""
        with self._lock:
            total_events = sum(m.count for m in self._events.values())
            events_by_type = {k: v.count for k, v in self._events.items()}
            event_errors = sum(m.error_count for m in self._events.values())

            total_errors = sum(
                sum(c for c in cat.values()) for cat in self._errors.values()
            )
            errors_by_category = {
                cat: sum(c.values()) for cat, c in self._errors.items()
            }

            module_load_times: dict[str, float] = {}
            module_states: dict[str, str] = {}
            for mod_name, records in self._module_lifecycle.items():
                last_initialize = [
                    r for r in records if r.hook == "initialize"
                ]
                if last_initialize:
                    module_load_times[mod_name] = last_initialize[-1].duration_ms
                last_record = records[-1] if records else None
                if last_record:
                    module_states[mod_name] = (
                        last_record.hook if last_record.success else "failed"
                    )

            return TelemetrySnapshot(
                uptime_seconds=0.0,  # set externally by the kernel
                total_events_emitted=total_events,
                events_by_type=events_by_type,
                total_errors=total_errors + event_errors,
                errors_by_category=errors_by_category,
                module_load_times=module_load_times,
                module_states=module_states,
                startup_duration_ms=self._startup_duration_ms,
            )

    def reset(self) -> None:
        """Reset all counters (useful in tests and after snapshot export)."""
        with self._lock:
            self._events.clear()
            self._module_lifecycle.clear()
            self._errors.clear()
            self._startup_duration_ms = 0.0


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------


class _EventMetrics:
    __slots__ = ("count", "total_duration_ms", "error_count")

    def __init__(self) -> None:
        self.count: int = 0
        self.total_duration_ms: float = 0.0
        self.error_count: int = 0


class _LifecycleRecord:
    __slots__ = ("hook", "duration_ms", "success")

    def __init__(self, hook: str, duration_ms: float, success: bool) -> None:
        self.hook = hook
        self.duration_ms = duration_ms
        self.success = success
