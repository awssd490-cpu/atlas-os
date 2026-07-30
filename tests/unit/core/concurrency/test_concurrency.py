"""Tests for the Atlas concurrency and resource management subsystem."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.core.concurrency import (
    ConcurrencyError,
    ConcurrencyLimiter,
    DuplicateResource,
    ManagedResource,
    ResourceManager,
    ResourceNotFound,
    ResourceState,
)
from app.core.concurrency.errors import (
    ConcurrencyError as ConcurrencyError_Impl,
    DuplicateResource as DuplicateResource_Impl,
    ResourceNotFound as ResourceNotFound_Impl,
)
from app.core.concurrency.limiter import ConcurrencyLimiter as ConcurrencyLimiter_Impl
from app.core.concurrency.models import ManagedResource as ManagedResource_Impl
from app.core.concurrency.models import ResourceState as ResourceState_Impl
from app.core.concurrency.resources import ResourceManager as ResourceManager_Impl
from app.core.errors import AtlasError


# ======================================================================
# Imports
# ======================================================================


class TestImports:
    def test_concurrency_limiter_imported(self) -> None:
        assert ConcurrencyLimiter is ConcurrencyLimiter_Impl

    def test_resource_state_imported(self) -> None:
        assert ResourceState is ResourceState_Impl

    def test_managed_resource_imported(self) -> None:
        assert ManagedResource is ManagedResource_Impl

    def test_resource_manager_imported(self) -> None:
        assert ResourceManager is ResourceManager_Impl

    def test_concurrency_error_imported(self) -> None:
        assert ConcurrencyError is ConcurrencyError_Impl

    def test_resource_not_found_imported(self) -> None:
        assert ResourceNotFound is ResourceNotFound_Impl

    def test_duplicate_resource_imported(self) -> None:
        assert DuplicateResource is DuplicateResource_Impl

    def test_error_hierarchy(self) -> None:
        assert issubclass(ConcurrencyError, AtlasError)
        assert issubclass(ResourceNotFound, ConcurrencyError)
        assert issubclass(DuplicateResource, ConcurrencyError)


# ======================================================================
# ResourceState
# ======================================================================


class TestResourceState:
    """ResourceState enum values."""

    def test_values(self) -> None:
        assert ResourceState.CREATED.value == 0
        assert ResourceState.OPEN.value == 1
        assert ResourceState.CLOSED.value == 2

    def test_transitions(self) -> None:
        """Verify natural ordering."""
        assert ResourceState.CREATED.value < ResourceState.OPEN.value
        assert ResourceState.OPEN.value < ResourceState.CLOSED.value


# ======================================================================
# ManagedResource
# ======================================================================


class TestManagedResource:
    """ManagedResource frozen dataclass."""

    def test_default_values(self) -> None:
        r = ManagedResource()
        assert r.name == ""
        assert r.state == ResourceState.CREATED
        assert r.metadata == {}

    def test_custom_values(self) -> None:
        r = ManagedResource(
            name="db_pool",
            state=ResourceState.OPEN,
            metadata={"pool_size": 10},
        )
        assert r.name == "db_pool"
        assert r.state == ResourceState.OPEN
        assert r.metadata == {"pool_size": 10}

    def test_immutable(self) -> None:
        r = ManagedResource()
        with pytest.raises(AttributeError):
            r.name = "changed"  # type: ignore[misc]


# ======================================================================
# ConcurrencyLimiter
# ======================================================================


class TestConcurrencyLimiter:
    """ConcurrencyLimiter semaphore-based tests."""

    def test_default_max(self) -> None:
        limiter = ConcurrencyLimiter()
        assert limiter.available_permits == 10

    def test_custom_max(self) -> None:
        limiter = ConcurrencyLimiter(max_concurrent=3)
        assert limiter.available_permits == 3

    def test_invalid_max_zero(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            ConcurrencyLimiter(max_concurrent=0)

    def test_invalid_max_negative(self) -> None:
        with pytest.raises(ValueError, match="at least 1"):
            ConcurrencyLimiter(max_concurrent=-1)

    @pytest.mark.asyncio
    async def test_acquire_release(self) -> None:
        limiter = ConcurrencyLimiter(max_concurrent=2)
        assert limiter.available_permits == 2

        await limiter.acquire()
        assert limiter.available_permits == 1

        await limiter.acquire()
        assert limiter.available_permits == 0

        limiter.release()
        assert limiter.available_permits == 1

        limiter.release()
        assert limiter.available_permits == 2

    @pytest.mark.asyncio
    async def test_context_manager(self) -> None:
        limiter = ConcurrencyLimiter(max_concurrent=1)

        async with limiter:
            assert limiter.available_permits == 0

        assert limiter.available_permits == 1

    @pytest.mark.asyncio
    async def test_concurrency_limited(self) -> None:
        """Only max_concurrent tasks should run simultaneously."""
        limiter = ConcurrencyLimiter(max_concurrent=2)
        events: list[int] = []
        max_concurrent_seen = 0

        async def worker(idx: int) -> None:
            nonlocal max_concurrent_seen
            async with limiter:
                concurrent = 2 - limiter.available_permits
                max_concurrent_seen = max(max_concurrent_seen, concurrent)
                events.append(idx)
                await asyncio.sleep(0.05)

        tasks = [worker(i) for i in range(6)]
        await asyncio.gather(*tasks)

        assert max_concurrent_seen <= 2
        assert len(events) == 6

    @pytest.mark.asyncio
    async def test_release_unlocks(self) -> None:
        """Releasing a permit should allow a new acquire."""
        limiter = ConcurrencyLimiter(max_concurrent=1)
        await limiter.acquire()
        assert limiter.available_permits == 0

        limiter.release()
        assert limiter.available_permits == 1


# ======================================================================
# ResourceManager
# ======================================================================


class TestResourceManager:
    """ResourceManager tests."""

    @pytest.fixture
    def mgr(self) -> ResourceManager:
        return ResourceManager()

    def test_register(self, mgr: ResourceManager) -> None:
        record = mgr.register("db", {"host": "localhost"})
        assert record.name == "db"
        assert record.state == ResourceState.CREATED
        assert record.metadata == {"registered": True}

    def test_register_no_object(self, mgr: ResourceManager) -> None:
        record = mgr.register("lock")
        assert record.name == "lock"
        assert mgr.get("lock") is None

    def test_register_duplicate(self, mgr: ResourceManager) -> None:
        mgr.register("db")
        with pytest.raises(DuplicateResource):
            mgr.register("db")

    def test_get(self, mgr: ResourceManager) -> None:
        obj = {"host": "localhost"}
        mgr.register("db", obj)
        assert mgr.get("db") is obj

    def test_get_unknown(self, mgr: ResourceManager) -> None:
        with pytest.raises(ResourceNotFound):
            mgr.get("nonexistent")

    def test_get_record(self, mgr: ResourceManager) -> None:
        mgr.register("cache")
        record = mgr.get_record("cache")
        assert record.name == "cache"
        assert record.state == ResourceState.CREATED

    def test_get_record_unknown(self, mgr: ResourceManager) -> None:
        with pytest.raises(ResourceNotFound):
            mgr.get_record("nonexistent")

    def test_unregister(self, mgr: ResourceManager) -> None:
        mgr.register("temp")
        mgr.unregister("temp")
        with pytest.raises(ResourceNotFound):
            mgr.get("temp")

    def test_unregister_unknown(self, mgr: ResourceManager) -> None:
        with pytest.raises(ResourceNotFound):
            mgr.unregister("nonexistent")

    def test_list_resources(self, mgr: ResourceManager) -> None:
        mgr.register("a")
        mgr.register("b")
        names = [r.name for r in mgr.list_resources()]
        assert names == ["a", "b"]

    def test_list_names(self, mgr: ResourceManager) -> None:
        mgr.register("x")
        mgr.register("y")
        assert mgr.list_names() == ["x", "y"]

    def test_open(self, mgr: ResourceManager) -> None:
        mgr.register("db")
        record = mgr.open("db")
        assert record.state == ResourceState.OPEN
        assert mgr.get_record("db").state == ResourceState.OPEN

    def test_open_unknown(self, mgr: ResourceManager) -> None:
        with pytest.raises(ResourceNotFound):
            mgr.open("nonexistent")

    def test_close(self, mgr: ResourceManager) -> None:
        mgr.register("db")
        mgr.open("db")
        record = mgr.close("db")
        assert record.state == ResourceState.CLOSED
        assert mgr.get_record("db").state == ResourceState.CLOSED

    def test_close_unknown(self, mgr: ResourceManager) -> None:
        with pytest.raises(ResourceNotFound):
            mgr.close("nonexistent")

    def test_close_all(self, mgr: ResourceManager) -> None:
        mgr.register("a")
        mgr.register("b")
        records = mgr.close_all()
        assert len(records) == 2
        for r in records:
            assert r.state == ResourceState.CLOSED
        assert mgr.get_record("a").state == ResourceState.CLOSED
        assert mgr.get_record("b").state == ResourceState.CLOSED

    def test_close_all_empty(self, mgr: ResourceManager) -> None:
        records = mgr.close_all()
        assert records == []

    def test_deterministic_ordering(self, mgr: ResourceManager) -> None:
        mgr.register("z")
        mgr.register("a")
        mgr.register("m")
        names = [r.name for r in mgr.list_resources()]
        assert names == ["z", "a", "m"]  # registration order


# ======================================================================
# ResourceManager — lifecycle integration
# ======================================================================


class TestResourceManagerLifecycle:
    """ResourceManager end-to-end lifecycle."""

    @pytest.fixture
    def mgr(self) -> ResourceManager:
        return ResourceManager()

    def test_full_lifecycle(self, mgr: ResourceManager) -> None:
        mgr.register("conn", "obj_value")

        # CREATED
        assert mgr.get_record("conn").state == ResourceState.CREATED
        assert mgr.get("conn") == "obj_value"

        # OPEN
        mgr.open("conn")
        assert mgr.get_record("conn").state == ResourceState.OPEN

        # CLOSED
        mgr.close("conn")
        assert mgr.get_record("conn").state == ResourceState.CLOSED

    def test_record_immutable(self, mgr: ResourceManager) -> None:
        mgr.register("x")
        record = mgr.get_record("x")
        with pytest.raises(AttributeError):
            record.name = "changed"  # type: ignore[misc]
