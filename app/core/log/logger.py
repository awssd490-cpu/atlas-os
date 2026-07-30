"""AtlasLogger — structured logger producing immutable ``LogRecord`` objects.

Log output is written to stderr with ISO-8601 timestamps.  Each call
returns an immutable ``LogRecord`` for downstream processing.
"""

from __future__ import annotations

import datetime
import sys
from typing import Any

from app.core.log.errors import InvalidLogLevel
from app.core.log.formatter import JsonFormatter
from app.core.log.models import LogRecord

_VALID_LEVELS: dict[str, int] = {
    "DEBUG": 10,
    "INFO": 20,
    "WARNING": 30,
    "ERROR": 40,
    "CRITICAL": 50,
}


class AtlasLogger:
    """Structured logger that produces immutable ``LogRecord`` objects.

    Each log method writes a single-line JSON record to stderr and
    returns the ``LogRecord`` for optional inspection.

    Usage::

        log = AtlasLogger("my.component")
        log.info("Server started", port=8080)
        log.error("Connection failed", remote_addr="10.0.0.1", retry=True)
    """

    def __init__(
        self,
        name: str = "atlas",
        *,
        level: str = "INFO",
        formatter: JsonFormatter | None = None,
    ) -> None:
        if level.upper() not in _VALID_LEVELS:
            raise InvalidLogLevel(level)
        self._name = name
        self._level = level.upper()
        self._level_num = _VALID_LEVELS[self._level]
        self._formatter = formatter or JsonFormatter()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Return the logger name."""
        return self._name

    @property
    def level(self) -> str:
        """Return the current log level."""
        return self._level

    # ------------------------------------------------------------------
    # Logging methods
    # ------------------------------------------------------------------

    def debug(
        self,
        message: str,
        **metadata: Any,
    ) -> LogRecord:
        """Log at DEBUG level.

        Args:
            message: The log message.
            **metadata: Structured key-value metadata.

        Returns:
            An immutable ``LogRecord``.
        """
        return self._log("DEBUG", message, metadata)

    def info(
        self,
        message: str,
        **metadata: Any,
    ) -> LogRecord:
        """Log at INFO level.

        Args:
            message: The log message.
            **metadata: Structured key-value metadata.

        Returns:
            An immutable ``LogRecord``.
        """
        return self._log("INFO", message, metadata)

    def warning(
        self,
        message: str,
        **metadata: Any,
    ) -> LogRecord:
        """Log at WARNING level.

        Args:
            message: The log message.
            **metadata: Structured key-value metadata.

        Returns:
            An immutable ``LogRecord``.
        """
        return self._log("WARNING", message, metadata)

    def error(
        self,
        message: str,
        **metadata: Any,
    ) -> LogRecord:
        """Log at ERROR level.

        Args:
            message: The log message.
            **metadata: Structured key-value metadata.

        Returns:
            An immutable ``LogRecord``.
        """
        return self._log("ERROR", message, metadata)

    def critical(
        self,
        message: str,
        **metadata: Any,
    ) -> LogRecord:
        """Log at CRITICAL level.

        Args:
            message: The log message.
            **metadata: Structured key-value metadata.

        Returns:
            An immutable ``LogRecord``.
        """
        return self._log("CRITICAL", message, metadata)

    def exception(
        self,
        message: str,
        exception: BaseException | None = None,
        **metadata: Any,
    ) -> LogRecord:
        """Log at ERROR level with exception information.

        The exception details (type, message, args) are automatically
        included in the metadata under ``"exception"``.

        Args:
            message: The log message.
            exception: An optional exception to log.
            **metadata: Structured key-value metadata.

        Returns:
            An immutable ``LogRecord``.
        """
        meta = dict(metadata)
        if exception is not None:
            meta["exception"] = {
                "type": type(exception).__name__,
                "message": str(exception),
            }
        return self._log("ERROR", message, meta)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _log(
        self,
        level: str,
        message: str,
        metadata: dict[str, Any],
    ) -> LogRecord:
        """Create and emit a log record."""
        if _VALID_LEVELS.get(level, 0) < self._level_num:
            # Below threshold — still create a record but don't write
            return LogRecord(
                level=level,
                logger=self._name,
                message=message,
                metadata=metadata,
            )

        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()

        record = LogRecord(
            timestamp=timestamp,
            level=level,
            logger=self._name,
            message=message,
            metadata=metadata,
        )

        # Write JSON to stderr
        line = self._formatter.format(record)
        print(line, file=sys.stderr)

        return record

    def is_enabled_for(self, level: str) -> bool:
        """Check whether the given level would be emitted.

        Args:
            level: The log level to check.

        Returns:
            ``True`` if records at *level* will be written to stderr.
        """
        level_num = _VALID_LEVELS.get(level.upper(), 0)
        return level_num >= self._level_num
