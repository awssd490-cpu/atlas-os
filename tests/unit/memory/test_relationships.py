"""Tests for MemoryGraphImpl (relationship management).

Verifies:
- Add relationship (basic, with properties, duplicate prevention)
- Self-referencing and non-existent memory rejection
- Cycle prevention for tree types (parent, child, depends_on)
- Permissive cycles for non-tree types (related, references, etc.)
- Get related: outgoing, incoming, both directions
- Get related: depth-limited traversal
- Remove relationship (by type, all between two memories)
- Importance propagation (depth, decay, no-op on unrelated)
- Event emission on add/remove
- MemoryManager integration
"""

from __future__ import annotations

from typing import Any

import pytest

from app.memory.memory import Memory, MemoryId, MemoryState
from app.memory.manager import MemoryManager, MemoryRepository
from app.memory.relationships import MemoryGraphImpl
from app.storage.interfaces import CacheService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def repo() -> MemoryRepository:
    from app.storage.connection.sqlite import SQLiteConnection
    from app.storage.migration.manager import SqliteMigrationManager
    from app.memory.migrations import V002_MemorySchema

    conn = SQLiteConnection(":memory:")
    manager = SqliteMigrationManager()
    await manager.apply_all(conn, [V002_MemorySchema()])
    yield MemoryRepository(connection=conn)
    await conn.close()


@pytest.fixture
async def graph(repo: MemoryRepository) -> MemoryGraphImpl:
    from app.storage.connection.sqlite import SQLiteConnection

    # repo is constructed with the underlying connection — reach it via
    # the private attribute for test purposes.  An alternative would be
    # to expose the connection, but that couples the test to internals.
    conn = repo._conn
    return MemoryGraphImpl(connection=conn)


@pytest.fixture
async def seeded_graph(repo: MemoryRepository, graph: MemoryGraphImpl) -> MemoryGraphImpl:
    """Return a graph with a few memories already created and related.

    Structure:
        mem_a (parent) → mem_b (child) → mem_c (child)
        mem_a (references) → mem_d
    """
    mem_a = Memory(content="parent", memory_id=MemoryId("mem-a"))
    mem_b = Memory(content="child-1", memory_id=MemoryId("mem-b"))
    mem_c = Memory(content="child-2", memory_id=MemoryId("mem-c"))
    mem_d = Memory(content="ref-target", memory_id=MemoryId("mem-d"))

    await repo.add(mem_a)
    await repo.add(mem_b)
    await repo.add(mem_c)
    await repo.add(mem_d)

    await graph.add_relationship("mem-a", "mem-b", "parent", {"weight": 1})
    await graph.add_relationship("mem-b", "mem-c", "parent")
    await graph.add_relationship("mem-a", "mem-d", "references", {"cite": "section 3"})

    return graph


# ---------------------------------------------------------------------------
# Add relationship
# ---------------------------------------------------------------------------


class TestAddRelationship:
    async def test_add_simple(self, repo: MemoryRepository, graph: MemoryGraphImpl) -> None:
        m1 = Memory(content="source")
        m2 = Memory(content="target")
        await repo.add(m1)
        await repo.add(m2)

        await graph.add_relationship(m1.id.value, m2.id.value, "references")
        related = await graph.get_related(m1.id.value, direction="outgoing")
        assert len(related) == 1
        assert related[0].id == m2.id

    async def test_add_with_properties(
        self, repo: MemoryRepository, graph: MemoryGraphImpl
    ) -> None:
        m1 = Memory(content="source", memory_id=MemoryId("m1"))
        m2 = Memory(content="target", memory_id=MemoryId("m2"))
        await repo.add(m1)
        await repo.add(m2)

        props = {"weight": 0.8, "label": "important"}
        await graph.add_relationship("m1", "m2", "references", props)
        # We verify by fetching — internal row reading validates properties
        related = await graph.get_related("m1", direction="outgoing")
        assert len(related) == 1

    async def test_add_rejects_self_reference(
        self, repo: MemoryRepository, graph: MemoryGraphImpl
    ) -> None:
        m1 = Memory(content="alone", memory_id=MemoryId("m1"))
        await repo.add(m1)

        with pytest.raises(ValueError, match="self-referencing"):
            await graph.add_relationship("m1", "m1", "parent")

    async def test_add_rejects_nonexistent_source(
        self, repo: MemoryRepository, graph: MemoryGraphImpl
    ) -> None:
        m1 = Memory(content="target", memory_id=MemoryId("m1"))
        await repo.add(m1)

        with pytest.raises(LookupError, match="not found"):
            await graph.add_relationship("no-such", "m1", "references")

    async def test_add_rejects_nonexistent_target(
        self, repo: MemoryRepository, graph: MemoryGraphImpl
    ) -> None:
        m1 = Memory(content="source", memory_id=MemoryId("m1"))
        await repo.add(m1)

        with pytest.raises(LookupError, match="not found"):
            await graph.add_relationship("m1", "no-such", "references")

    async def test_add_rejects_duplicate(
        self, repo: MemoryRepository, graph: MemoryGraphImpl
    ) ->None:
        m1 = Memory(content="a", memory_id=MemoryId("a"))
        m2 = Memory(content="b", memory_id=MemoryId("b"))
        await repo.add(m1)
        await repo.add(m2)

        await graph.add_relationship("a", "b", "references")
        with pytest.raises(ValueError, match="already exists"):
            await graph.add_relationship("a", "b", "references")

    async def test_add_allows_different_types(
        self, repo: MemoryRepository, graph: MemoryGraphImpl
    ) -> None:
        m1 = Memory(content="a", memory_id=MemoryId("a"))
        m2 = Memory(content="b", memory_id=MemoryId("b"))
        await repo.add(m1)
        await repo.add(m2)

        await graph.add_relationship("a", "b", "references")
        await graph.add_relationship("a", "b", "parent")  # different type, ok
        related = await graph.get_related("a", direction="outgoing")
        assert len(related) == 1  # b appears once, but 2 edges exist


# ---------------------------------------------------------------------------
# Cycle prevention
# ---------------------------------------------------------------------------


class TestCyclePrevention:
    async def test_prevents_parent_cycle(
        self, repo: MemoryRepository, graph: MemoryGraphImpl
    ) -> None:
        m_a = Memory(content="a", memory_id=MemoryId("a"))
        m_b = Memory(content="b", memory_id=MemoryId("b"))
        m_c = Memory(content="c", memory_id=MemoryId("c"))
        await repo.add(m_a)
        await repo.add(m_b)
        await repo.add(m_c)

        await graph.add_relationship("a", "b", "parent")
        await graph.add_relationship("b", "c", "parent")
        # c → a would create a cycle
        with pytest.raises(ValueError, match="cycle"):
            await graph.add_relationship("c", "a", "parent")

    async def test_prevents_child_cycle(
        self, repo: MemoryRepository, graph: MemoryGraphImpl
    ) -> None:
        m_a = Memory(content="a", memory_id=MemoryId("a"))
        m_b = Memory(content="b", memory_id=MemoryId("b"))
        await repo.add(m_a)
        await repo.add(m_b)

        await graph.add_relationship("a", "b", "child")
        with pytest.raises(ValueError, match="cycle"):
            await graph.add_relationship("b", "a", "child")

    async def test_prevents_depends_on_cycle(
        self, repo: MemoryRepository, graph: MemoryGraphImpl
    ) -> None:
        m_a = Memory(content="a", memory_id=MemoryId("a"))
        m_b = Memory(content="b", memory_id=MemoryId("b"))
        await repo.add(m_a)
        await repo.add(m_b)

        await graph.add_relationship("a", "b", "depends_on")
        with pytest.raises(ValueError, match="cycle"):
            await graph.add_relationship("b", "a", "depends_on")

    async def test_allows_cycle_in_references(
        self, repo: MemoryRepository, graph: MemoryGraphImpl
    ) -> None:
        m_a = Memory(content="a", memory_id=MemoryId("a"))
        m_b = Memory(content="b", memory_id=MemoryId("b"))
        await repo.add(m_a)
        await repo.add(m_b)

        # references permits cycles
        await graph.add_relationship("a", "b", "references")
        await graph.add_relationship("b", "a", "references")  # should not raise
        related_from_a = await graph.get_related("a", direction="both")
        assert len(related_from_a) == 1  # b

    async def test_allows_cycle_in_related(
        self, repo: MemoryRepository, graph: MemoryGraphImpl
    ) -> None:
        m_a = Memory(content="a", memory_id=MemoryId("a"))
        m_b = Memory(content="b", memory_id=MemoryId("b"))
        await repo.add(m_a)
        await repo.add(m_b)

        await graph.add_relationship("a", "b", "related")
        await graph.add_relationship("b", "a", "related")  # should not raise
        related_from_a = await graph.get_related("a", direction="both")
        assert len(related_from_a) == 1


# ---------------------------------------------------------------------------
# Get related
# ---------------------------------------------------------------------------


class TestGetRelated:
    async def test_outgoing(self, seeded_graph: MemoryGraphImpl) -> None:
        related = await seeded_graph.get_related("mem-a", direction="outgoing")
        ids = {m.id.value for m in related}
        assert ids == {"mem-b", "mem-d"}

    async def test_incoming(self, seeded_graph: MemoryGraphImpl) -> None:
        related = await seeded_graph.get_related("mem-c", direction="incoming")
        ids = {m.id.value for m in related}
        assert ids == {"mem-b"}

    async def test_both(self, seeded_graph: MemoryGraphImpl) -> None:
        related = await seeded_graph.get_related("mem-b", direction="both")
        ids = {m.id.value for m in related}
        assert ids == {"mem-a", "mem-c"}

    async def test_filter_by_type(self, seeded_graph: MemoryGraphImpl) -> None:
        related = await seeded_graph.get_related("mem-a", rel_type="references")
        ids = {m.id.value for m in related}
        assert ids == {"mem-d"}

    async def test_depth_limited(self, seeded_graph: MemoryGraphImpl) -> None:
        # depth=1 from mem-a should only reach immediate neighbours
        related = await seeded_graph.get_related("mem-a", max_depth=1)
        ids = {m.id.value for m in related}
        assert ids == {"mem-b", "mem-d"}

    async def test_depth_two(self, seeded_graph: MemoryGraphImpl) -> None:
        # depth=2 from mem-a should reach mem-c via mem-b
        related = await seeded_graph.get_related("mem-a", max_depth=2)
        ids = {m.id.value for m in related}
        assert ids == {"mem-b", "mem-c", "mem-d"}

    async def test_depth_zero(self, seeded_graph: MemoryGraphImpl) -> None:
        related = await seeded_graph.get_related("mem-a", max_depth=0)
        assert len(related) == 0

    async def test_empty(self, graph: MemoryGraphImpl) -> None:
        related = await graph.get_related("nonexistent")
        assert len(related) == 0


# ---------------------------------------------------------------------------
# Remove relationship
# ---------------------------------------------------------------------------


class TestRemoveRelationship:
    async def test_remove_by_type(self, seeded_graph: MemoryGraphImpl) -> None:
        await seeded_graph.remove_relationship("mem-a", "mem-b", "parent")
        related = await seeded_graph.get_related("mem-a", direction="outgoing")
        ids = {m.id.value for m in related}
        assert "mem-b" not in ids  # parent edge removed
        assert "mem-d" in ids  # references edge remains

    async def test_remove_all(self, seeded_graph: MemoryGraphImpl) -> None:
        await seeded_graph.remove_relationship("mem-a", "mem-b")  # no type
        related = await seeded_graph.get_related("mem-a", direction="outgoing")
        ids = {m.id.value for m in related}
        assert ids == {"mem-d"}

    async def test_remove_nonexistent(self, graph: MemoryGraphImpl) -> None:
        # Should not raise
        await graph.remove_relationship("no-such-source", "no-such-target", "parent")


# ---------------------------------------------------------------------------
# Importance propagation
# ---------------------------------------------------------------------------


class TestPropagateImportance:
    async def test_propagates_to_children(
        self, repo: MemoryRepository, graph: MemoryGraphImpl
    ) -> None:
        m_a = Memory(content="a", importance=0.9, memory_id=MemoryId("a"))
        m_b = Memory(content="b", importance=0.3, memory_id=MemoryId("b"))
        await repo.add(m_a)
        await repo.add(m_b)
        await graph.add_relationship("a", "b", "parent")

        updated = await graph.propagate_importance("a", decay=0.5, max_depth=1)
        assert updated == 1

        # b's importance should have been boosted
        row = await repo.get(MemoryId("b"))
        assert row is not None
        assert row.importance > 0.3

    async def test_no_propagate_to_irrelevant(
        self, repo: MemoryRepository, graph: MemoryGraphImpl
    ) -> None:
        m_a = Memory(content="a", importance=0.9, memory_id=MemoryId("a"))
        m_b = Memory(content="b", importance=0.9, memory_id=MemoryId("b"))  # already high
        await repo.add(m_a)
        await repo.add(m_b)
        await graph.add_relationship("a", "b", "parent")

        updated = await graph.propagate_importance("a", decay=0.5, max_depth=1)
        assert updated == 0  # already above propagated value

    async def test_propagate_nonexistent(self, graph: MemoryGraphImpl) -> None:
        updated = await graph.propagate_importance("no-such", decay=0.5, max_depth=3)
        assert updated == 0

    async def test_propagate_depth_zero(
        self, repo: MemoryRepository, graph: MemoryGraphImpl
    ) -> None:
        m_a = Memory(content="a", importance=0.9, memory_id=MemoryId("a"))
        await repo.add(m_a)
        updated = await graph.propagate_importance("a", decay=0.5, max_depth=0)
        assert updated == 0


# ---------------------------------------------------------------------------
# Event emission
# ---------------------------------------------------------------------------


class _TestEventBus:
    """Simple event bus that records published events."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    async def publish(self, event: Any) -> None:
        self.events.append(event)


class TestEventEmission:
    async def test_add_emits_event(
        self, repo: MemoryRepository, graph: MemoryGraphImpl
    ) -> None:
        event_bus = _TestEventBus()
        g = type(graph)(connection=repo._conn, event_bus=event_bus)

        m1 = Memory(content="a", memory_id=MemoryId("a"))
        m2 = Memory(content="b", memory_id=MemoryId("b"))
        await repo.add(m1)
        await repo.add(m2)

        await g.add_relationship("a", "b", "references")
        assert len(event_bus.events) == 1
        event = event_bus.events[0]
        assert event.memory_id == "a"
        assert event.added == ["b"]

    async def test_remove_emits_event(
        self, repo: MemoryRepository, graph: MemoryGraphImpl
    ) -> None:
        event_bus = _TestEventBus()
        g = type(graph)(connection=repo._conn, event_bus=event_bus)

        m1 = Memory(content="a", memory_id=MemoryId("a"))
        m2 = Memory(content="b", memory_id=MemoryId("b"))
        await repo.add(m1)
        await repo.add(m2)
        await g.add_relationship("a", "b", "references")

        await g.remove_relationship("a", "b", "references")
        # add + remove = 2 events
        assert len(event_bus.events) == 2
        event = event_bus.events[1]
        assert event.memory_id == "a"
        assert event.removed == ["b"]


# ---------------------------------------------------------------------------
# MemoryManager integration
# ---------------------------------------------------------------------------


class TestManagerIntegration:
    async def test_manager_has_graph_property(self, repo: MemoryRepository) -> None:
        graph = MemoryGraphImpl(connection=repo._conn)
        mgr = MemoryManager(repository=repo, graph=graph)
        assert mgr.graph is graph

    async def test_manager_graph_none_by_default(self, repo: MemoryRepository) -> None:
        mgr = MemoryManager(repository=repo)
        assert mgr.graph is None

    async def test_manager_can_use_graph(self, repo: MemoryRepository) -> None:
        graph = MemoryGraphImpl(connection=repo._conn)
        mgr = MemoryManager(repository=repo, graph=graph)

        m1 = Memory(content="via manager", memory_id=MemoryId("m1"))
        m2 = Memory(content="also via manager", memory_id=MemoryId("m2"))
        await mgr.create(m1)
        await mgr.create(m2)

        assert mgr.graph is not None
        await mgr.graph.add_relationship("m1", "m2", "references")
        related = await mgr.graph.get_related("m1", direction="outgoing")
        assert len(related) == 1
        assert related[0].id == m2.id
