"""Health monitoring domain models.

All models in this module are immutable frozen dataclasses.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Mapping


class HealthStatus(enum.Enum):
    """Status of a health check.

    Values are ordered from best to worst for comparison::

        UNKNOWN < HEALTHY < DEGRADED < UNHEALTHY
    """

    UNKNOWN = 0
    HEALTHY = 1
    DEGRADED = 2
    UNHEALTHY = 3


@dataclass(frozen=True)
class HealthCheck:
    """Immutable result of a single health check.

    Attributes:
        name: Name of the health check.
        status: The :class:`HealthStatus` value.
        message: Human-readable description of the check result.
        duration_ms: Execution time of the check in milliseconds.
        metadata: Optional structured metadata about the check.
    """

    name: str = ""
    status: HealthStatus = HealthStatus.UNKNOWN
    message: str = ""
    duration_ms: float = 0.0
    metadata: Mapping[str, Any] = field(default_factory=dict)
