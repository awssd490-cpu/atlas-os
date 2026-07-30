"""Logging — structured logging for Atlas.

Provides ``AtlasLogger``, a structured logger that produces immutable
``LogRecord`` objects written as single-line JSON to stderr, and
``JsonFormatter`` for deterministic serialisation.
"""

from __future__ import annotations

from app.core.log.errors import InvalidLogLevel, LoggingError
from app.core.log.formatter import JsonFormatter
from app.core.log.logger import AtlasLogger
from app.core.log.models import LogRecord

__all__ = [
    "AtlasLogger",
    "InvalidLogLevel",
    "JsonFormatter",
    "LogRecord",
    "LoggingError",
]
