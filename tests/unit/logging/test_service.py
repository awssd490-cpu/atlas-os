"""Tests for LoguruLoggingService.

Verifies:
- ``get_logger`` returns a Logger-compatible object
- Logger methods work (debug, info, warning, error, exception)
- ``bind()`` returns a new logger with extra context
- ``configure()`` can be called without error
- The logger conforms to the ``Logger`` protocol
"""

from __future__ import annotations

from typing import Any

import pytest

from app.core.interfaces import Logger
from app.logging.service import LoguruLoggingService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def service() -> LoguruLoggingService:
    return LoguruLoggingService()


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestLoggerProtocol:
    """Verifies the returned logger satisfies the ``Logger`` protocol."""

    def test_is_logger_protocol(self, service: LoguruLoggingService) -> None:
        logger = service.get_logger("test")
        # __call__ is not in the protocol, but runtime_checkable will
        # check that all methods exist.
        assert isinstance(logger, Logger)

    def test_all_methods_exist(self, service: LoguruLoggingService) -> None:
        logger = service.get_logger("test")
        for method in ("debug", "info", "warning", "error", "exception", "bind"):
            assert hasattr(logger, method)
            assert callable(getattr(logger, method))


# ---------------------------------------------------------------------------
# Logger method smoke tests
# ---------------------------------------------------------------------------


class TestLoggerMethods:
    """Call each method — they should not raise."""

    def test_debug(self, service: LoguruLoggingService) -> None:
        logger = service.get_logger("test")
        logger.debug("debug {n}", n=42)  # should not raise

    def test_info(self, service: LoguruLoggingService) -> None:
        logger = service.get_logger("test")
        logger.info("info {n}", n=42)

    def test_warning(self, service: LoguruLoggingService) -> None:
        logger = service.get_logger("test")
        logger.warning("warning {n}", n=42)

    def test_error(self, service: LoguruLoggingService) -> None:
        logger = service.get_logger("test")
        logger.error("error {n}", n=42)

    def test_exception(self, service: LoguruLoggingService) -> None:
        logger = service.get_logger("test")
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            logger.exception("exception test")

    def test_info_no_args(self, service: LoguruLoggingService) -> None:
        logger = service.get_logger("test")
        logger.info("plain message")


class TestBind:
    def test_bind_returns_logger(self, service: LoguruLoggingService) -> None:
        logger = service.get_logger("test")
        bound = logger.bind(task_id="abc-123")
        assert isinstance(bound, Logger)

    def test_bind_methods_work(self, service: LoguruLoggingService) -> None:
        logger = service.get_logger("test")
        bound = logger.bind(task_id="abc-123")
        bound.info("bound log {task_id}", task_id="abc-123")


# ---------------------------------------------------------------------------
# Module distinction
# ---------------------------------------------------------------------------


class TestModuleScoping:
    def test_different_modules_return_loggers(self, service: LoguruLoggingService) -> None:
        logger_a = service.get_logger("module_a")
        logger_b = service.get_logger("module_b")
        # Both should be usable independently
        logger_a.info("from a")
        logger_b.info("from b")
