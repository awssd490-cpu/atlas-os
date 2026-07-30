"""Tests for the Atlas structured logging subsystem."""

from __future__ import annotations

import json
from typing import Any

import pytest

from app.core.log import (
    AtlasLogger,
    InvalidLogLevel,
    JsonFormatter,
    LogRecord,
    LoggingError,
)
from app.core.log.errors import InvalidLogLevel as InvalidLogLevel_Impl
from app.core.log.errors import LoggingError as LoggingError_Impl
from app.core.log.formatter import JsonFormatter as JsonFormatter_Impl
from app.core.log.logger import AtlasLogger as AtlasLogger_Impl
from app.core.log.models import LogRecord as LogRecord_Impl
from app.core.errors import AtlasError


# ======================================================================
# Imports
# ======================================================================


class TestImports:
    def test_log_record_imported(self) -> None:
        assert LogRecord is LogRecord_Impl

    def test_json_formatter_imported(self) -> None:
        assert JsonFormatter is JsonFormatter_Impl

    def test_atlas_logger_imported(self) -> None:
        assert AtlasLogger is AtlasLogger_Impl

    def test_logging_error_imported(self) -> None:
        assert LoggingError is LoggingError_Impl

    def test_invalid_log_level_imported(self) -> None:
        assert InvalidLogLevel is InvalidLogLevel_Impl

    def test_error_hierarchy(self) -> None:
        assert issubclass(LoggingError, AtlasError)
        assert issubclass(InvalidLogLevel, LoggingError)


# ======================================================================
# LogRecord
# ======================================================================


class TestLogRecord:
    """LogRecord frozen dataclass."""

    def test_default_values(self) -> None:
        r = LogRecord()
        assert r.timestamp == ""
        assert r.level == ""
        assert r.logger == ""
        assert r.message == ""
        assert r.metadata == {}

    def test_custom_values(self) -> None:
        r = LogRecord(
            timestamp="2026-07-30T12:00:00+00:00",
            level="INFO",
            logger="test",
            message="hello",
            metadata={"key": "val"},
        )
        assert r.timestamp == "2026-07-30T12:00:00+00:00"
        assert r.level == "INFO"
        assert r.logger == "test"
        assert r.message == "hello"
        assert r.metadata == {"key": "val"}

    def test_immutable(self) -> None:
        r = LogRecord()
        with pytest.raises(AttributeError):
            r.message = "changed"  # type: ignore[misc]


# ======================================================================
# JsonFormatter
# ======================================================================


class TestJsonFormatter:
    """JsonFormatter tests."""

    @pytest.fixture
    def formatter(self) -> JsonFormatter:
        return JsonFormatter()

    def test_format(self, formatter: JsonFormatter) -> None:
        r = LogRecord(
            timestamp="2026-01-01T00:00:00+00:00",
            level="INFO",
            logger="test",
            message="hello",
            metadata={"key": "val"},
        )
        line = formatter.format(r)
        data = json.loads(line)
        assert data["timestamp"] == "2026-01-01T00:00:00+00:00"
        assert data["level"] == "INFO"
        assert data["logger"] == "test"
        assert data["message"] == "hello"
        assert data["metadata"] == {"key": "val"}

    def test_deterministic(self, formatter: JsonFormatter) -> None:
        r = LogRecord(timestamp="t", level="WARN", logger="x", message="m", metadata={"z": 1, "a": 2})
        a = formatter.format(r)
        b = formatter.format(r)
        assert a == b

    def test_sorted_keys(self, formatter: JsonFormatter) -> None:
        r = LogRecord(timestamp="t", level="WARN", logger="x", message="m")
        line = formatter.format(r)
        data = json.loads(line)
        assert list(data.keys()) == sorted(data.keys())

    def test_unicode(self, formatter: JsonFormatter) -> None:
        r = LogRecord(
            timestamp="t",
            level="INFO",
            logger="test",
            message="東京の首都",
            metadata={"city": "東京"},
        )
        line = formatter.format(r)
        assert "東京" in line
        data = json.loads(line)
        assert data["message"] == "東京の首都"
        assert data["metadata"]["city"] == "東京"

    def test_empty_metadata(self, formatter: JsonFormatter) -> None:
        r = LogRecord(timestamp="t", level="INFO", logger="x", message="m")
        line = formatter.format(r)
        data = json.loads(line)
        assert data["metadata"] == {}


# ======================================================================
# AtlasLogger — construction
# ======================================================================


class TestAtlasLoggerConstruction:
    """AtlasLogger construction tests."""

    def test_default_name(self) -> None:
        log = AtlasLogger()
        assert log.name == "atlas"
        assert log.level == "INFO"

    def test_custom_name(self) -> None:
        log = AtlasLogger("my.service")
        assert log.name == "my.service"

    def test_custom_level(self) -> None:
        log = AtlasLogger("test", level="DEBUG")
        assert log.level == "DEBUG"

    def test_invalid_level(self) -> None:
        with pytest.raises(InvalidLogLevel):
            AtlasLogger("test", level="TRACE")

    def test_case_insensitive_level(self) -> None:
        log = AtlasLogger("test", level="warning")
        assert log.level == "WARNING"


# ======================================================================
# AtlasLogger — log level filtering
# ======================================================================


class TestAtlasLoggerLevelFilter:
    """AtlasLogger level filtering."""

    def test_debug_below_info(self) -> None:
        """INFO logger should not emit DEBUG."""
        log = AtlasLogger("test", level="INFO")
        assert log.is_enabled_for("DEBUG") is False

    def test_info_at_info(self) -> None:
        log = AtlasLogger("test", level="INFO")
        assert log.is_enabled_for("INFO") is True

    def test_debug_at_debug(self) -> None:
        log = AtlasLogger("test", level="DEBUG")
        assert log.is_enabled_for("DEBUG") is True

    def test_info_at_debug(self) -> None:
        log = AtlasLogger("test", level="DEBUG")
        assert log.is_enabled_for("INFO") is True


# ======================================================================
# AtlasLogger — log methods (return values)
# ======================================================================


class TestAtlasLoggerMethods:
    """AtlasLogger log methods return immutable LogRecords."""

    @pytest.fixture
    def log(self) -> AtlasLogger:
        return AtlasLogger("test-logger", level="DEBUG")

    def test_debug_returns_record(self, log: AtlasLogger) -> None:
        r = log.debug("debug message", count=1)
        assert isinstance(r, LogRecord)
        assert r.level == "DEBUG"
        assert r.logger == "test-logger"
        assert r.message == "debug message"
        assert r.metadata == {"count": 1}

    def test_info_returns_record(self, log: AtlasLogger) -> None:
        r = log.info("info message")
        assert r.level == "INFO"
        assert r.logger == "test-logger"
        assert r.message == "info message"

    def test_warning_returns_record(self, log: AtlasLogger) -> None:
        r = log.warning("warning message")
        assert r.level == "WARNING"

    def test_error_returns_record(self, log: AtlasLogger) -> None:
        r = log.error("error message", code=500)
        assert r.level == "ERROR"
        assert r.metadata == {"code": 500}

    def test_critical_returns_record(self, log: AtlasLogger) -> None:
        r = log.critical("critical message")
        assert r.level == "CRITICAL"

    def test_exception_without_exception(self, log: AtlasLogger) -> None:
        r = log.exception("something failed")
        assert r.level == "ERROR"
        assert "exception" not in r.metadata

    def test_exception_with_exception(self, log: AtlasLogger) -> None:
        try:
            raise ValueError("test error")
        except ValueError as exc:
            r = log.exception("operation failed", exception=exc)
        assert r.level == "ERROR"
        assert r.metadata["exception"]["type"] == "ValueError"
        assert r.metadata["exception"]["message"] == "test error"

    def test_exception_metadata_preserved(self, log: AtlasLogger) -> None:
        try:
            raise RuntimeError("fail")
        except RuntimeError as exc:
            r = log.exception("failed", exception=exc, component="worker", attempt=3)
        assert r.metadata["component"] == "worker"
        assert r.metadata["attempt"] == 3
        assert r.metadata["exception"]["type"] == "RuntimeError"

    def test_record_immutable(self, log: AtlasLogger) -> None:
        r = log.info("test")
        with pytest.raises(AttributeError):
            r.message = "changed"  # type: ignore[misc]

    def test_metadata_unicode(self, log: AtlasLogger) -> None:
        r = log.info("unicode test", city="東京")
        assert r.metadata["city"] == "東京"

    def test_message_unicode(self, log: AtlasLogger) -> None:
        r = log.info("東京の首都")
        assert "東京" in r.message


# ======================================================================
# AtlasLogger — output suppression
# ======================================================================


class TestAtlasLoggerSuppressed:
    """When level is below threshold, returns record without writing."""

    @pytest.fixture
    def log(self) -> AtlasLogger:
        return AtlasLogger("test", level="WARNING")

    def test_debug_suppressed(self, log: AtlasLogger) -> None:
        r = log.debug("should not appear")
        assert r.level == "DEBUG"
        assert r.message == "should not appear"
        # No timestamp means it wasn't emitted
        assert r.timestamp == ""

    def test_info_suppressed(self, log: AtlasLogger) -> None:
        r = log.info("should not appear")
        assert r.timestamp == ""

    def test_warning_emitted(self, log: AtlasLogger) -> None:
        r = log.warning("should appear")
        assert r.timestamp != ""


# ======================================================================
# AtlasLogger — stderr output (capsys)
# ======================================================================


class TestAtlasLoggerStderr:
    """Verify JSON output is written to stderr."""

    def test_info_writes_to_stderr(self, capsys: Any) -> None:
        log = AtlasLogger("test", level="INFO")
        log.info("hello world", key="val")
        captured = capsys.readouterr()
        assert captured.err != ""
        data = json.loads(captured.err.strip())
        assert data["level"] == "INFO"
        assert data["message"] == "hello world"
        assert data["logger"] == "test"
        assert data["metadata"] == {"key": "val"}

    def test_debug_not_written(self, capsys: Any) -> None:
        log = AtlasLogger("test", level="INFO")
        log.debug("debug message")
        captured = capsys.readouterr()
        assert captured.err == ""

    def test_multiple_lines(self, capsys: Any) -> None:
        log = AtlasLogger("test", level="INFO")
        log.info("first")
        log.error("second")
        captured = capsys.readouterr()
        lines = captured.err.strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["message"] == "first"
        assert json.loads(lines[1])["message"] == "second"

    def test_unicode_output(self, capsys: Any) -> None:
        log = AtlasLogger("test", level="INFO")
        log.info("東京の首都")
        captured = capsys.readouterr()
        assert "東京" in captured.err
        data = json.loads(captured.err.strip())
        assert data["message"] == "東京の首都"
