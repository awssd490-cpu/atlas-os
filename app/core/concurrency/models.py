"""Concurrency domain models.

All models in this module are immutable frozen dataclasses and enums.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Mapping


class ResourceState(enum.Enum):
    """State of a managed resource.

    Resources transition: CREATED → OPEN → CLOSED
    """

    CREATED = 0
    OPEN = 1
    CLOSED = 2


@dataclass(frozen=True)
class ManagedResource:
    """Immutable record of a managed resource.

    Attributes:
        name: Unique resource name.
        state: Current :class:`ResourceState`.
        metadata: Optional structured metadata about the resource.
    """

    name: str = ""
    state: ResourceState = ResourceState.CREATED
    metadata: Mapping[str, Any] = field(default_factory=dict)
