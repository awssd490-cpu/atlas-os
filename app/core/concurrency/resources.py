"""ResourceManager — registry and lifecycle for named resources.

Tracks resource state (CREATED, OPEN, CLOSED) and provides
exception-safe lifecycle management.
"""

from __future__ import annotations

from typing import Any

from app.core.concurrency.errors import DuplicateResource, ResourceNotFound
from app.core.concurrency.models import ManagedResource, ResourceState


class ResourceManager:
    """Manages named resources with state tracking.

    Supports registration, lookup, lifecycle transitions, and
    bulk cleanup.

    Usage::

        mgr = ResourceManager()

        # Register a resource (state → CREATED)
        mgr.register("database", db_connection)

        # Open it (state → OPEN)
        mgr.open("database")

        # Retrieve it
        conn = mgr.get("database")

        # Close it (state → CLOSED)
        mgr.close("database")

        # Close all resources
        mgr.close_all()
    """

    def __init__(self) -> None:
        self._resources: dict[str, ManagedResource] = {}
        self._objects: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, name: str, resource: object = None) -> ManagedResource:  # type: ignore[assignment]
        """Register a resource.

        Args:
            name: Unique resource name.
            resource: Optional associated object (connection, client, etc.).

        Returns:
            An immutable ``ManagedResource`` record.

        Raises:
            DuplicateResource: If *name* is already registered.
        """
        if name in self._resources:
            raise DuplicateResource(name)

        record = ManagedResource(
            name=name,
            state=ResourceState.CREATED,
            metadata={"registered": True},
        )
        self._resources[name] = record
        if resource is not None:
            self._objects[name] = resource
        return record

    def unregister(self, name: str) -> None:
        """Unregister a resource.

        Args:
            name: The resource name to unregister.

        Raises:
            ResourceNotFound: If not registered.
        """
        if name not in self._resources:
            raise ResourceNotFound(name)
        del self._resources[name]
        self._objects.pop(name, None)

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> Any:
        """Look up a resource object by name.

        Args:
            name: The resource name.

        Returns:
            The associated object, or ``None`` if no object was
            stored during registration.

        Raises:
            ResourceNotFound: If not registered.
        """
        if name not in self._resources:
            raise ResourceNotFound(name)
        return self._objects.get(name)

    def get_record(self, name: str) -> ManagedResource:
        """Look up a resource record by name.

        Args:
            name: The resource name.

        Returns:
            The immutable ``ManagedResource`` record.

        Raises:
            ResourceNotFound: If not registered.
        """
        if name not in self._resources:
            raise ResourceNotFound(name)
        return self._resources[name]

    def list_resources(self) -> list[ManagedResource]:
        """Return immutable records for all registered resources.

        Results are in registration order.
        """
        return list(self._resources.values())

    def list_names(self) -> list[str]:
        """Return the names of all registered resources.

        Results are in registration order.
        """
        return list(self._resources.keys())

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self, name: str) -> ManagedResource:
        """Transition a resource to OPEN state.

        Args:
            name: The resource name.

        Returns:
            Updated ``ManagedResource`` record.

        Raises:
            ResourceNotFound: If not registered.
        """
        if name not in self._resources:
            raise ResourceNotFound(name)

        record = ManagedResource(
            name=name,
            state=ResourceState.OPEN,
            metadata={"opened": True},
        )
        self._resources[name] = record
        return record

    def close(self, name: str) -> ManagedResource:
        """Transition a resource to CLOSED state.

        Args:
            name: The resource name.

        Returns:
            Updated ``ManagedResource`` record.

        Raises:
            ResourceNotFound: If not registered.
        """
        if name not in self._resources:
            raise ResourceNotFound(name)

        record = ManagedResource(
            name=name,
            state=ResourceState.CLOSED,
            metadata={"closed": True},
        )
        self._resources[name] = record
        return record

    def close_all(self) -> list[ManagedResource]:
        """Close every registered resource.

        Returns:
            List of ``ManagedResource`` records in CLOSED state.
        """
        records: list[ManagedResource] = []
        for name in list(self._resources.keys()):
            record = ManagedResource(
                name=name,
                state=ResourceState.CLOSED,
                metadata={"closed": True},
            )
            self._resources[name] = record
            records.append(record)
        return records
