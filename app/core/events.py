"""Standard ATLAS event envelope.

Every event flowing through the system carries this structure:

.. code-block::

    EventID        — globally unique identifier (ULID)
    Version        — event schema version (monotonic integer)
    Timestamp      — when the event was created (UTC, nanosecond precision)
    CorrelationID  — traces a logical operation across subsystems
    Source         — module that published the event
    Target         — optional intended recipient (module, engine, or "*")
    Payload        — the domain-specific data (Pydantic model)
    Metadata       — extensible key-value bag (routing hints, tenant id, etc.)
"""

from __future__ import annotations

import abc
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ulid() -> str:
    """Return a ULID-formatted string (26 chars, sortable, unique).

    Pure-Python implementation avoids a dependency on ``python-ulid``.
    """
    # 48-bit timestamp (ms) + 80 bits of randomness
    import math
    import os

    now_ms = math.floor(datetime.now(timezone.utc).timestamp() * 1000)
    rand_bytes = os.urandom(10)
    rand_int = int.from_bytes(rand_bytes, "big")

    # Encode as Crockford base-32
    value = (now_ms << 80) | rand_int
    alphabet = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
    chars: list[str] = []
    for _ in range(26):
        chars.append(alphabet[value & 0x1F])
        value >>= 5
    chars.reverse()
    return "".join(chars)


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Standard event envelope
# ---------------------------------------------------------------------------


class Event(BaseModel):
    """Base class for **every** event on the ATLAS Event Bus.

    All events are Pydantic models.  The standard envelope fields are
    populated automatically; subclasses add their domain data as
    additional fields.

    Example::

        class TaskCreated(Event):
            task_id: str
            task_type: str
            payload: dict
    """

    model_config = ConfigDict(
        # Events are immutable once created
        frozen=True,
        # Allow extra fields on subclasses
        extra="allow",
        # Use model name as the default event type
        str_strip_whitespace=True,
    )

    # -- Standard envelope --------------------------------------------------

    event_id: str = Field(default_factory=_ulid, description="Globally unique event identifier (ULID)")
    version: int = Field(default=1, description="Event schema version")
    timestamp: datetime = Field(default_factory=_now_utc, description="Creation time (UTC)")
    correlation_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Tracing identifier for logical operations",
    )
    source: str = Field(default="kernel", description="Module or component that published the event")
    target: str = Field(
        default="*",
        description="Intended recipient (module name, engine, or '*' for all)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Extensible routing hints, tenant context, etc.",
    )

    # -- Overridable event type string --------------------------------------

    # Stable string identifier for routing and logging.  Defaults to the
    # class name but can be overridden in subclasses for backward-compat
    # after refactoring.
    _event_type: ClassVar[str] = ""

    @classmethod
    def event_type(cls) -> str:
        """Return the stable event-type string.

        Uses ``_event_type`` if set, otherwise falls back to ``cls.__name__``.
        """
        return cls._event_type or cls.__name__

    # -- Convenience builders -----------------------------------------------

    def with_correlation(self, correlation_id: str) -> "Event":
        """Return a copy with a different ``correlation_id``."""
        return self.model_copy(update={"correlation_id": correlation_id})

    def with_source(self, source: str) -> "Event":
        """Return a copy with a different ``source``."""
        return self.model_copy(update={"source": source})

    def with_metadata(self, **meta: Any) -> "Event":
        """Return a copy with *meta* merged into ``metadata``."""
        merged = {**self.metadata, **meta}
        return self.model_copy(update={"metadata": merged})


# ---------------------------------------------------------------------------
# Kernel lifecycle events (standard payloads)
# ---------------------------------------------------------------------------


class KernelBooting(Event):
    """Published when the kernel begins its boot sequence."""

    _event_type: ClassVar[str] = "kernel.booting"
    source: str = "kernel"
    module_count: int = 0


class KernelBooted(Event):
    """Published when the kernel has successfully booted."""

    _event_type: ClassVar[str] = "kernel.booted"
    source: str = "kernel"
    duration_ms: float = 0.0
    module_count: int = 0


class KernelShuttingDown(Event):
    """Published when the kernel begins its shutdown sequence."""

    _event_type: ClassVar[str] = "kernel.shutting_down"
    source: str = "kernel"


class KernelStopped(Event):
    """Published when the kernel has fully stopped."""

    _event_type: ClassVar[str] = "kernel.stopped"
    source: str = "kernel"


class ModuleRegistered(Event):
    """Published when a module is registered with the kernel."""

    _event_type: ClassVar[str] = "module.registered"
    source: str = "kernel"
    module_name: str = ""
    module_version: str = ""


class ModuleLifecycleEvent(Event):
    """Published at each module lifecycle transition."""

    _event_type: ClassVar[str] = "module.lifecycle"
    source: str = "kernel"
    module_name: str = ""
    lifecycle_hook: str = ""  # e.g. "initialize", "start", "ready", ...
    success: bool = True
    duration_ms: float = 0.0


class ModuleFailed(Event):
    """Published when a module's lifecycle hook fails."""

    _event_type: ClassVar[str] = "module.failed"
    source: str = "kernel"
    module_name: str = ""
    lifecycle_hook: str = ""
    error_message: str = ""
