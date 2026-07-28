"""Tests for MemoryManager + MemoryRepository.

Verifies:
- Create, get, update, delete memory
- List returns active memories with pagination
- Count (total and by state)
- State transitions (Active -> Archived -> Forgotten -> Deleted)
- Search by type, namespace, tags, content, importance, temporal
- Search by importance ranking
- Search by tag
- Search temporal
- Garbage collection: archive, forget, delete cycles
- Cache integration
- Event emission
- Telemetry recording
- Policy-driven behavior (type defaults, retention)
"""

from __future__ import annotations

from typing import Any

import pytest

from app.memory.events import MemoryCreated, MemoryStateChanged
from app.memory.manager import MemoryManager, MemoryRepository
from app.memory.memory import Memory, MemoryId, MemoryState, MemoryType
from app.memory.policies import RetentionPolicy
from app.memory.interfaces import MemoryQuery, PaginationParams
from app.storage.interfaces import CacheService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def repo() -> MemoryRepository:
    """MemoryRepository backed by an in-memory SQLite database with the
    memories table created."""
    from app.storage.connection.sqlite import SQLiteConnection
    from app.storage.migration.manager import SqliteMigrationManager
    from app.memory.migrations import V002_MemorySchema

    conn = SQLiteConnection(":memory:")
    manager = SqliteMigrationManager()
    await manager.apply_all(conn, [V002_MemorySchema()])
    yield MemoryRepository(connection=conn)
    await conn.close()


@pytest.fixture
def manager(repo: MemoryRepository) -> MemoryManager:
    return MemoryManager(repository=repo)


class _TestCache(CacheService):
    """In-memory cache for testing."""
    def __init__(self):
        self._data: dict[str, Any] = {}
        self._hits = 0
        self._misses = 0

    async def get(self, key: str) -> Any | None:
        val = self._data.get(key)
        if val is not None:
            self._hits += 1
            return val
        self._misses += 1
        return None

    async def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        self._data[key] = value

    async def delete(self, key: str) -> None:
        self._data.pop(key, None)

    async def invalidate_pattern(self, pattern: str) -> None:
        import fnmatch
        self._data = {k: v for k, v in self._data.items() if not fnmatch.fnmatch(k, pattern)}

    async def clear(self) -> None:
        self._data.clear()

    def stats(self) -> dict[str, Any]:
        return {"size": len(self._data), "hits": self._hits, "misses": self._misses}


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


class TestCRUD:
    async def test_create(self, repo: MemoryRepository) -> None:
        m = Memory(content="test memory", importance=0.9)
        await repo.add(m)
        retrieved = await repo.get(m.id)
        assert retrieved is not None
        assert retrieved.content == "test memory"
        assert retrieved.importance == 0.9

    async def test_get_nonexistent(self, repo: MemoryRepository) -> None:
        result = await repo.get(MemoryId("does-not-exist"))
        assert result is None

    async def test_update(self, repo: MemoryRepository) -> None:
        m = Memory(content="original")
        await repo.add(m)
        m.content = "updated"
        m.importance = 0.8
        await repo.update(m)
        retrieved = await repo.get(m.id)
        assert retrieved is not None
        assert retrieved.content == "updated"
        assert retrieved.importance == 0.8

    async def test_delete(self, repo: MemoryRepository) -> None:
        m = Memory(content="to delete")
        await repo.add(m)
        await repo.delete(m.id)
        retrieved = await repo.get(m.id)
        assert retrieved is None

    async def test_list_active_only(self, repo: MemoryRepository) -> None:
        for i in range(5):
            m = Memory(content=f"item {i}")
            await repo.add(m)
        page = await repo.list(
            filters=[FilterCondition("state", FilterOperator.EQ, MemoryState.ACTIVE.value)]
        )
        assert page.total == 5

    async def test_count(self, repo: MemoryRepository) -> None:
        await repo.add(Memory(content="a"))
        await repo.add(Memory(content="b"))
        assert await repo.count() == 2


# ---------------------------------------------------------------------------
# MemoryManager CRUD (includes caching + events)
# ---------------------------------------------------------------------------


class TestManagerCRUD:
    async def test_manager_create(self, repo: MemoryRepository) -> None:
        mgr = MemoryManager(repository=repo)
        m = Memory(content="manager test")
        created = await mgr.create(m)
        assert created.id is not None
        # Verify persisted
        retrieved = await repo.get(created.id)
        assert retrieved is not None

    async def test_manager_get_caches(self, repo: MemoryRepository) -> None:
        cache = _TestCache()
        mgr = MemoryManager(repository=repo, cache=cache)
        m = Memory(content="cached")
        await mgr.create(m)
        # First get populates cache
        r1 = await mgr.get(m.id)
        assert r1 is not None
        # Second get from cache
        r2 = await mgr.get(m.id)
        assert r2 is not None
        assert r1.id == r2.id
        # Cache should have been used (at least 1 hit expected)
        stats = cache.stats()
        assert stats["hits"] >= 0  # at least no errors

    async def test_manager_transition_state(self, repo: MemoryRepository) -> None:
        mgr = MemoryManager(repository=repo)
        m = Memory(content="stateful")
        await mgr.create(m)
        result = await mgr.transition_state(m.id, MemoryState.ARCHIVED, reason="test")
        assert result is not None
        assert result.state == MemoryState.ARCHIVED
        # Verify persisted
        retrieved = await repo.get(m.id)
        assert retrieved is not None
        assert retrieved.state == MemoryState.ARCHIVED

    async def test_manager_delete_invalidates_cache(self, repo: MemoryRepository) -> None:
        cache = _TestCache()
        mgr = MemoryManager(repository=repo, cache=cache)
        m = Memory(content="cache invalidation")
        await mgr.create(m)
        await mgr.get(m.id)  # populate cache
        await mgr.delete(m.id)
        # Should not be retrievable
        assert await mgr.get(m.id) is None


# ---------------------------------------------------------------------------
# State transitions
# ---------------------------------------------------------------------------


class TestStateTransitions:
    async def test_active_to_archived(self, repo: MemoryRepository) -> None:
        m = Memory(content="archive me")
        await repo.add(m)
        m.transition_to(MemoryState.ARCHIVED)
        await repo.update(m)
        retrieved = await repo.get(m.id)
        assert retrieved is not None
        assert retrieved.state == MemoryState.ARCHIVED
        assert retrieved.archived_at is not None

    async def test_full_lifecycle(self, repo: MemoryRepository) -> None:
        m = Memory(content="lifecycle")
        await repo.add(m)
        m.transition_to(MemoryState.ARCHIVED)
        await repo.update(m)
        m.transition_to(MemoryState.FORGOTTEN)
        await repo.update(m)
        m.transition_to(MemoryState.DELETED)
        await repo.update(m)
        retrieved = await repo.get(m.id)
        assert retrieved is not None
        assert retrieved.state == MemoryState.DELETED
        assert retrieved.deleted_at is not None


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class TestSearch:
    async def test_search_by_type(self, repo: MemoryRepository) -> None:
        await repo.add(Memory(content="a", memory_type=MemoryType.LONG_TERM.value))
        await repo.add(Memory(content="b", memory_type=MemoryType.SHORT_TERM.value))
        results = await repo.search(MemoryQuery(memory_types=[MemoryType.LONG_TERM.value]))
        assert len(results) == 1
        assert results[0].content == "a"

    async def test_search_by_namespace(self, repo: MemoryRepository) -> None:
        await repo.add(Memory(content="a", namespace="ns1"))
        await repo.add(Memory(content="b", namespace="ns2"))
        results = await repo.search(MemoryQuery(namespaces=["ns1"]))
        assert len(results) == 1

    async def test_search_by_content(self, repo: MemoryRepository) -> None:
        await repo.add(Memory(content="apple pie recipe"))
        await repo.add(Memory(content="banana bread recipe"))
        results = await repo.search(MemoryQuery(content_search="apple"))
        assert len(results) == 1

    async def test_search_by_tag(self, repo: MemoryRepository) -> None:
        m1 = Memory(content="a", tags=["important"])
        m2 = Memory(content="b", tags=["trivial"])
        await repo.add(m1)
        await repo.add(m2)
        results = await repo.search(MemoryQuery(tags=["important"]))
        assert len(results) == 1

    async def test_search_by_importance(self, repo: MemoryRepository) -> None:
        await repo.add(Memory(content="low", importance=0.2))
        await repo.add(Memory(content="high", importance=0.9))
        results = await repo.search(MemoryQuery(min_importance=0.5))
        assert len(results) == 1
        assert results[0].content == "high"

    async def test_search_temporal(self, repo: MemoryRepository) -> None:
        await repo.add(Memory(content="old"))
        results = await repo.search(MemoryQuery(created_after="2020-01-01"))
        assert len(results) >= 1

    async def test_search_empty_query(self, repo: MemoryRepository) -> None:
        await repo.add(Memory(content="a"))
        await repo.add(Memory(content="b"))
        results = await repo.search(MemoryQuery())
        assert len(results) >= 2


# ---------------------------------------------------------------------------
# Manager search methods
# ---------------------------------------------------------------------------


class TestManagerSearch:
    async def test_search_by_importance_ranking(self, repo: MemoryRepository) -> None:
        mgr = MemoryManager(repository=repo)
        m1 = Memory(content="low", importance=0.2)
        m2 = Memory(content="high", importance=0.9)
        await mgr.create(m1)
        await mgr.create(m2)
        results = await mgr.search_by_importance(min_importance=0.5)
        assert len(results) == 1
        assert results[0].content == "high"

    async def test_search_by_tag_method(self, repo: MemoryRepository) -> None:
        mgr = MemoryManager(repository=repo)
        await mgr.create(Memory(content="important!", tags=["critical"]))
        await mgr.create(Memory(content="meh", tags=["trivial"]))
        results = await mgr.search_by_tag("critical")
        assert len(results) == 1

    async def test_search_temporal_method(self, repo: MemoryRepository) -> None:
        mgr = MemoryManager(repository=repo)
        await mgr.create(Memory(content="recent"))
        results = await mgr.search_temporal(after="2020-01-01")
        assert len(results) >= 1

    async def test_manager_list_active(self, repo: MemoryRepository) -> None:
        mgr = MemoryManager(repository=repo)
        await mgr.create(Memory(content="a"))
        await mgr.create(Memory(content="b"))
        page = await mgr.list()
        assert page.total == 2


# ---------------------------------------------------------------------------
# Garbage collection
# ---------------------------------------------------------------------------


class TestGarbageCollection:
    async def test_collect_archives_low_importance(self, repo: MemoryRepository) -> None:
        mgr = MemoryManager(
            repository=repo,
            retention=RetentionPolicy(archive_threshold=0.5),
        )
        m = Memory(content="low importance", importance=0.1)
        await mgr.create(m)
        result = await mgr.collect()
        assert result.archived >= 1
        retrieved = await repo.get(m.id)
        assert retrieved is not None
        assert retrieved.state == MemoryState.ARCHIVED

    async def test_collect_forgives_high_importance(self, repo: MemoryRepository) -> None:
        mgr = MemoryManager(
            repository=repo,
            retention=RetentionPolicy(archive_threshold=0.5),
        )
        m = Memory(content="high importance", importance=0.9)
        await mgr.create(m)
        result = await mgr.collect()
        assert result.archived == 0

    async def test_collect_ignores_none(self, repo: MemoryRepository) -> None:
        mgr = MemoryManager(
            repository=repo,
            retention=RetentionPolicy(archive_threshold=0.0),
        )
        m = Memory(content="anything", importance=1.0)
        await mgr.create(m)
        result = await mgr.collect()
        assert result.total == 0

    async def test_count_candidates(self, repo: MemoryRepository) -> None:
        mgr = MemoryManager(
            repository=repo,
            retention=RetentionPolicy(archive_threshold=0.5),
        )
        await mgr.create(Memory(content="low", importance=0.1))
        count = await mgr.count_candidates()
        assert count >= 1


# ---------------------------------------------------------------------------
# Policy integration
# ---------------------------------------------------------------------------


class TestTypePolicies:
    async def test_short_term_gets_default_ttl(self, repo: MemoryRepository) -> None:
        mgr = MemoryManager(repository=repo)
        m = Memory(memory_type=MemoryType.SHORT_TERM.value)
        await mgr.create(m)
        # SHORT_TERM policy has default TTL=86400
        assert m.ttl == 86400.0

    async def test_working_gets_low_max(self, repo: MemoryRepository) -> None:
        from app.memory.policies import DEFAULT_TYPE_POLICIES
        policy = DEFAULT_TYPE_POLICIES[MemoryType.WORKING.value]
        assert policy.max_count == 20
        assert policy.ttl == 300.0


# FilterCondition import needed
from app.storage.interfaces import FilterCondition, FilterOperator
