"""Logging domain models.

All models in this module are immutable frozen dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class LogRecord:
    """An immutable log record produced by an ``AtlasLogger``.

    Attributes:
        timestamp: ISO-8601 formatted timestamp string.
        level: Log level name (e.g. ``"INFO"``, ``"ERROR"``).
        logger: Name of the logger that produced this record.
        message: The log message text.
        metadata: Structured metadata attached to the record.
    """

    timestamp: str = ""
    level: str = ""
    logger: str = ""
    message: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
