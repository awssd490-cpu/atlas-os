"""Module manifest and health contracts.

The manifest is the *static* declaration a module exposes to the kernel:
identity, dependencies, capabilities, and configuration schema. Runtime
health is reported separately through :meth:`Module.health`, which returns
a :class:`ModuleHealth`.
"""

from __future__ import annotations

import enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class HealthStatus(str, enum.Enum):
    """Tri-state health classification used across ATLAS."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class ModuleHealth(BaseModel):
    """Runtime health report returned by :meth:`Module.health`."""

    status: HealthStatus
    details: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def ok(cls, **details: Any) -> "ModuleHealth":
        """Convenience constructor for a healthy report."""
        return cls(status=HealthStatus.HEALTHY, details=details)

    @classmethod
    def degraded(cls, **details: Any) -> "ModuleHealth":
        """Convenience constructor for a degraded report."""
        return cls(status=HealthStatus.DEGRADED, details=details)

    @classmethod
    def unhealthy(cls, **details: Any) -> "ModuleHealth":
        """Convenience constructor for an unhealthy report."""
        return cls(status=HealthStatus.UNHEALTHY, details=details)


class CapabilityDeclaration(BaseModel):
    """A capability a module offers to the rest of the system.

    Capability names are namespaced with dots, e.g. ``"storage.sql"``,
    ``"memory.vector_search"``. Consumers discover capabilities by name
    through the :class:`CapabilityRegistry`, never by importing concrete
    module classes.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_.]*$")
    version: str = "1.0"
    description: str = ""


class ModuleManifest(BaseModel):
    """Static, immutable module self-description.

    Every module MUST expose a manifest. The kernel uses it for
    registration, dependency ordering, capability validation, and
    introspection APIs.
    """

    model_config = ConfigDict(frozen=True)

    name: str = Field(pattern=r"^[a-z][a-z0-9_-]*$", min_length=2, max_length=64)
    version: str = Field(min_length=1)
    description: str = ""
    dependencies: list[str] = Field(default_factory=list)
    capabilities: list[CapabilityDeclaration] = Field(default_factory=list)
    config_schema: dict[str, Any] | None = None
