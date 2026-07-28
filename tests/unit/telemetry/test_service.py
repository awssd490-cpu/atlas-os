"""Tests for InMemoryTelemetryService.

Verifies:
- Event metrics recording
- Module lifecycle recording
- Error recording
- Startup duration recording
- Snapshot returns correct aggregated data
- Reset clears all counters
"""

from __future__ import annotations

from app.telemetry.service import InMemoryTelemetryService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def telemetry() -> InMemoryTelemetryService:
    return InMemoryTelemetryService()


# ---------------------------------------------------------------------------
# Event metrics
# ---------------------------------------------------------------------------


class TestEventMetrics:
    def test_records_event_count(self) -> None:
        svc = telemetry()
        svc.record_event_metrics("task.created", 10.5, True)
        svc.record_event_metrics("task.created", 5.2, True)
        snap = svc.snapshot()
        assert snap.total_events_emitted == 2
        assert snap.events_by_type["task.created"] == 2

    def test_records_event_failure(self) -> None:
        svc = telemetry()
        svc.record_event_metrics("task.created", 10.0, False)
        snap = svc.snapshot()
        assert snap.total_events_emitted == 1
        assert snap.total_errors > 0


# ---------------------------------------------------------------------------
# Module lifecycle
# ---------------------------------------------------------------------------


class TestModuleLifecycle:
    def test_records_lifecycle_hook(self) -> None:
        svc = telemetry()
        svc.record_module_lifecycle("config", "initialize", 15.0, True)
        svc.record_module_lifecycle("config", "start", 5.0, True)
        snap = svc.snapshot()
        assert "config" in snap.module_load_times

    def test_records_failed_lifecycle(self) -> None:
        svc = telemetry()
        svc.record_module_lifecycle("db", "start", 100.0, False)
        snap = svc.snapshot()
        assert snap.module_states.get("db") == "failed"


# ---------------------------------------------------------------------------
# Error recording
# ---------------------------------------------------------------------------


class TestErrorRecording:
    def test_records_error(self) -> None:
        svc = telemetry()
        svc.record_error("database", "connection_refused")
        snap = svc.snapshot()
        assert snap.total_errors == 1
        assert "database" in snap.errors_by_category

    def test_records_multiple_errors(self) -> None:
        svc = telemetry()
        svc.record_error("database", "connection_refused", count=3)
        svc.record_error("event_bus", "timeout", count=2)
        snap = svc.snapshot()
        assert snap.total_errors == 5
        assert snap.errors_by_category["database"] == 3
        assert snap.errors_by_category["event_bus"] == 2


# ---------------------------------------------------------------------------
# Startup duration
# ---------------------------------------------------------------------------


class TestStartupDuration:
    def test_initial_zero(self) -> None:
        svc = telemetry()
        assert svc.snapshot().startup_duration_ms == 0.0

    def test_set_duration(self) -> None:
        svc = telemetry()
        svc.set_startup_duration(1250.5)
        assert svc.snapshot().startup_duration_ms == 1250.5


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


class TestSnapshot:
    def test_snapshot_is_immutable_copy(self) -> None:
        svc = telemetry()
        svc.record_event_metrics("test.event", 1.0, True)
        snap1 = svc.snapshot()
        svc.record_event_metrics("test.event", 2.0, True)
        snap2 = svc.snapshot()
        # snap1 should not have been updated
        assert snap1.total_events_emitted == 1
        assert snap2.total_events_emitted == 2

    def test_snapshot_to_dict(self) -> None:
        svc = telemetry()
        svc.set_startup_duration(500.0)
        d = svc.snapshot().to_dict()
        assert d["startup_duration_ms"] == 500.0
        assert "events_by_type" in d


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_all(self) -> None:
        svc = telemetry()
        svc.record_event_metrics("test", 1.0, True)
        svc.record_error("test", "err")
        svc.set_startup_duration(100.0)
        svc.reset()
        snap = svc.snapshot()
        assert snap.total_events_emitted == 0
        assert snap.total_errors == 0
        assert snap.startup_duration_ms == 0.0
