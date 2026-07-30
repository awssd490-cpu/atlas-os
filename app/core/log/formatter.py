"""JsonFormatter — serialises ``LogRecord`` objects to deterministic JSON.

Output is UTF-8 JSON with sorted keys, suitable for structured logging
pipelines (logstash, fluentd, CloudWatch, etc.).
"""

from __future__ import annotations

import json
from typing import Any

from app.core.log.models import LogRecord


class JsonFormatter:
    """Formats ``LogRecord`` objects as deterministic JSON strings.

    Output is single-line JSON (no indentation), UTF-8 encoded, with
    sorted keys for deterministic output.

    Usage::

        formatter = JsonFormatter()
        record = LogRecord(level="INFO", message="hello", ...)
        line = formatter.format(record)
    """

    def format(self, record: LogRecord) -> str:
        """Format a ``LogRecord`` as a JSON string.

        Args:
            record: The log record to format.

        Returns:
            A JSON string with sorted keys and ``ensure_ascii=False``.
        """
        data: dict[str, Any] = {
            "timestamp": record.timestamp,
            "level": record.level,
            "logger": record.logger,
            "message": record.message,
            "metadata": dict(record.metadata),
        }

        return json.dumps(
            data,
            ensure_ascii=False,
            sort_keys=True,
        )
