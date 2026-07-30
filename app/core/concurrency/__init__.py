"""Concurrency — lightweight concurrency and resource management for Atlas.

Provides ``ConcurrencyLimiter`` for controlling async concurrency and
``ResourceManager`` for lifecycle tracking of named resources.
"""

from __future__ import annotations

from app.core.concurrency.errors import ConcurrencyError, DuplicateResource, ResourceNotFound
from app.core.concurrency.limiter import ConcurrencyLimiter
from app.core.concurrency.models import ManagedResource, ResourceState
from app.core.concurrency.resources import ResourceManager

__all__ = [
    "ConcurrencyError",
    "ConcurrencyLimiter",
    "DuplicateResource",
    "ManagedResource",
    "ResourceManager",
    "ResourceNotFound",
    "ResourceState",
]
