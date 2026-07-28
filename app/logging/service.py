"""Logging service — Loguru facade for the ATLAS kernel.

**Responsibility:** configure Loguru sinks from settings, then provide
per-module loggers through the :class:`LoggingService` interface.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import loguru

from app.core.interfaces import Logger, LoggingService


# ---------------------------------------------------------------------------
# Intercept handler: routes stdlib ``logging`` records into Loguru
# ---------------------------------------------------------------------------


class _InterceptHandler(logging.Handler):
    """Routing handler that feeds stdlib ``logging`` messages into Loguru.

    Register this via ``logging.basicConfig(handlers=[...], ...)`` so that
    third-party libraries (Uvicorn, SQLAlchemy, httpx) emit their logs
    through our configured Loguru sinks with correct formatting.
    """

    def emit(self, record: logging.LogRecord) -> None:
        loguru_level: str | int
        try:
            loguru_level = loguru.logger.level(record.levelname).name
        except ValueError:
            loguru_level = record.levelno

        frame, depth = logging.currentframe(), 6  # noqa: F841
        loguru.logger.opt(depth=depth, exception=record.exc_info).log(
            loguru_level,
            record.getMessage(),
        )


# ---------------------------------------------------------------------------
# Loguru wrapper — conforms to the Logger protocol
# ---------------------------------------------------------------------------


class _LoguruLogger:
    """Thin wrapper that delegates to a Loguru logger bound to a module name.

    This exists purely so we can produce loggers *without* requiring that
    callers import ``loguru``.  The wrapper conforms to the :class:`Logger`
    protocol.
    """

    def __init__(self, module: str) -> None:
        self._module = module
        self._logger = loguru.logger.bind(module=module)

    def debug(self, _message: str, *args: Any, **kwargs: Any) -> None:
        self._logger.opt(depth=1).debug(_message, *args, **kwargs)

    def info(self, _message: str, *args: Any, **kwargs: Any) -> None:
        self._logger.opt(depth=1).info(_message, *args, **kwargs)

    def warning(self, _message: str, *args: Any, **kwargs: Any) -> None:
        self._logger.opt(depth=1).warning(_message, *args, **kwargs)

    def error(self, _message: str, *args: Any, **kwargs: Any) -> None:
        self._logger.opt(depth=1).error(_message, *args, **kwargs)

    def exception(self, _message: str, *args: Any, **kwargs: Any) -> None:
        self._logger.opt(depth=1, exception=True).error(_message, *args, **kwargs)

    def bind(self, **kwargs: Any) -> "Logger":
        new = _LoguruLogger.__new__(_LoguruLogger)
        new._module = self._module
        new._logger = self._logger.bind(**kwargs)
        return new  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Public facade
# ---------------------------------------------------------------------------


# Cached proxy to avoid allocating a wrapper on every call
_loguru_logger_proxy = _LoguruLogger("system")


class LoguruLoggingService(LoggingService):
    """Logging service implementation backed by Loguru.

    Usage::

        service = LoguruLoggingService()
        service.configure()                     # one-shot at startup
        logger = service.get_logger("my_module")
        logger.info("hello {place}", place="world")
    """

    def __init__(self) -> None:
        self._configured = False

    def configure(self) -> None:
        """Set up Loguru sinks based on the current configuration.

        Must be called at least once before any logger is used.  Safe to
        call multiple times — it removes existing handlers first.
        """
        # Remove the default sink and any previously configured ones
        loguru.logger.remove()

        # Determine mode from config (import lazily to avoid circular deps)
        from app.config.settings import AtlasSettings

        try:
            # When running via DI, the config service holds the settings
            from app.core.interfaces import ConfigService

            container = None  # would come from DI
        except ImportError:
            settings = AtlasSettings()
        else:
            settings = AtlasSettings()

        log_cfg = settings.logging

        # Console sink
        if "console" in log_cfg.sinks or not log_cfg.sinks:
            self._add_console_sink(log_cfg)

        # File sink
        if "file" in log_cfg.sinks and log_cfg.file_path:
            self._add_file_sink(log_cfg)

        # Route stdlib logging into Loguru
        self._intercept_stdlib(log_cfg.level)

        self._configured = True

    def get_logger(self, module: str) -> Logger:
        """Return a :class:`Logger` bound to the given *module* name.

        The returned logger is a lightweight wrapper that adds ``module``
        to every log record's context.
        """
        return _LoguruLogger(module)

    # ------------------------------------------------------------------
    # Internal sink helpers
    # ------------------------------------------------------------------

    def _add_console_sink(self, log_cfg: Any) -> None:
        """Add a console sink with environment-appropriate formatting."""
        if log_cfg.format == "json" or log_cfg.serialize:
            loguru.logger.add(
                sys.stderr,
                level=log_cfg.level,
                serialize=True,
                enqueue=True,
                backtrace=True,
                diagnose=False,
            )
        else:
            loguru.logger.add(
                sys.stderr,
                level=log_cfg.level,
                format=(
                    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                    "<level>{level: <8}</level> | "
                    "<cyan>{extra[module]: <16}</cyan> | "
                    "<level>{message}</level>"
                ),
                colorize=True,
                enqueue=True,
                backtrace=True,
                diagnose=False,
            )

    def _add_file_sink(self, log_cfg: Any) -> None:
        """Add a JSON file sink."""
        path = Path(log_cfg.file_path)  # type: ignore[arg-type]
        path.parent.mkdir(parents=True, exist_ok=True)
        loguru.logger.add(
            str(path),
            level=log_cfg.level,
            serialize=True,
            enqueue=True,
            retention="30 days",
            rotation="100 MB",
        )

    @staticmethod
    def _intercept_stdlib(level: str) -> None:
        """Route stdlib logging into Loguru."""
        logging.basicConfig(handlers=[_InterceptHandler()], level=level, force=True)
