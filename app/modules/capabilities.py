"""Capability registry — service-discovery layer for ATLAS modules.

Modules declare capabilities in their manifests.  Other modules discover
capabilities by name through the registry, never by importing concrete
module classes.

Policy:
    Multiple providers for the same capability are allowed.
    ``get()`` returns the **first-registered** provider.
    ``get_all()`` returns all providers in registration order.
"""

from __future__ import annotations

from typing import Any

from app.core.interfaces import CapabilityEntry, CapabilityName, CapabilityRegistry


class InMemoryCapabilityRegistry(CapabilityRegistry):
    """Capability registry backed by an in-memory dict."""

    def __init__(self) -> None:
        self._entries: dict[CapabilityName, list[_CapabilityRecord]] = {}

    def register(self, capability: Any, provider: str) -> None:
        """Register *provider* (a module name) for a capability.

        Args:
            capability: A :class:`CapabilityDeclaration` object.
            provider: The module name that offers this capability.
        """
        name = capability.name
        if name not in self._entries:
            self._entries[name] = []
        self._entries[name].append(
            _CapabilityRecord(capability=capability, provider=provider)
        )

    def get(self, name: CapabilityName) -> CapabilityEntry | None:
        """Return the first-registered provider for *name*, or ``None``."""
        records = self._entries.get(name)
        if not records:
            return None
        return records[0]  # type: ignore[return-value]

    def get_all(self, name: CapabilityName) -> list[CapabilityEntry]:
        """Return every provider of *name* (empty list if none)."""
        records = self._entries.get(name, [])
        return list(records)  # type: ignore[return-value]

    def find_by_provider(self, module_name: str) -> list[CapabilityEntry]:
        """Return all capabilities registered by a given module."""
        result: list[CapabilityEntry] = []
        for records in self._entries.values():
            for r in records:
                if r.provider == module_name:
                    result.append(r)  # type: ignore[arg-type]
        return result

    def is_registered(self, name: CapabilityName) -> bool:
        """Return ``True`` when at least one provider exists."""
        records = self._entries.get(name)
        return records is not None and len(records) > 0


class _CapabilityRecord:
    """Internal record holding capability metadata."""

    __slots__ = ("capability", "provider")

    def __init__(self, capability: Any, provider: str) -> None:
        self.capability = capability
        self.provider = provider
