"""Health — lightweight health monitoring for Atlas.

Provides ``HealthMonitor`` for executing registered health checks and
collecting immutable ``HealthCheck`` result objects.  Supports both
sync and async check functions.
"""

from __future__ import annotations

from app.core.health.errors import DuplicateHealthCheck, HealthCheckNotFound, HealthError
from app.core.health.models import HealthCheck, HealthStatus
from app.core.health.monitor import HealthMonitor
from app.core.health.registry import clear_checks, get, list_checks, register, unregister

__all__ = [
    "DuplicateHealthCheck",
    "HealthCheck",
    "HealthCheckNotFound",
    "HealthError",
    "HealthMonitor",
    "HealthStatus",
    "clear_checks",
    "get",
    "list_checks",
    "register",
    "unregister",
]
