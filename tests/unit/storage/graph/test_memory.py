"""Tests for InMemoryGraphStore.

Verifies:
- Create and get nodes
- Update nodes
- Delete nodes (cascading to relationships)
- Find nodes by label and properties
- Create and query relationships
- BFS traversal
"""

from __future__ import annotations

import pytest

from app.storage.interfaces import GraphNode, GraphRelationship
from app.storage.graph.memory import InMemoryGraphStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store() -> InMemoryGraphStore:
    return InMemoryGraphStore()


# ---------------------------------------------------------------------------
# Node CRUD
# ---------------------------------------------------------------------------


class TestNodeCRUD:
    async def test_create_and_get(self, store: InMemoryGraphStore) -> None:
        node = GraphNode(id="n1", labels=["Person"], properties={"name": "Alice"})
        created = await store.create_node(node)
        assert created.id == "n1"
        retrieved = await store.get_node("n1")
        assert retrieved is not None
        assert retrieved.labels == ["Person"]
        assert retrieved.properties["name"] == "Alice"

    async def test_get_nonexistent(self, store: InMemoryGraphStore) -> None:
        assert await store.get_node("nonexistent") is None

    async def test_create_duplicate_raises(self, store: InMemoryGraphStore) -> None:
        node = GraphNode(id="dup", labels=[])
        await store.create_node(node)
        with pytest.raises(ValueError, match="already exists"):
            await store.create_node(GraphNode(id="dup", labels=[]))

    async def test_update(self, store: InMemoryGraphStore) -> None:
        await store.create_node(GraphNode(id="u1", labels=["A"], properties={"x": 1}))
        await store.update_node(GraphNode(id="u1", labels=["B"], properties={"x": 2}))
        node = await store.get_node("u1")
        assert node is not None
        assert node.labels == ["B"]
        assert node.properties["x"] == 2

    async def test_delete(self, store: InMemoryGraphStore) -> None:
        await store.create_node(GraphNode(id="d1", labels=[]))
        await store.delete_node("d1")
        assert await store.get_node("d1") is None


# ---------------------------------------------------------------------------
# Find nodes
# ---------------------------------------------------------------------------


class TestFindNodes:
    async def test_find_by_label(self, store: InMemoryGraphStore) -> None:
        await store.create_node(GraphNode(id="a", labels=["Person"]))
        await store.create_node(GraphNode(id="b", labels=["Robot"]))
        results = await store.find_nodes(labels=["Person"])
        assert len(results) == 1
        assert results[0].id == "a"

    async def test_find_by_properties(self, store: InMemoryGraphStore) -> None:
        await store.create_node(GraphNode(id="a", properties={"age": 30}))
        await store.create_node(GraphNode(id="b", properties={"age": 25}))
        results = await store.find_nodes(properties={"age": 30})
        assert len(results) == 1
        assert results[0].id == "a"

    async def test_find_empty(self, store: InMemoryGraphStore) -> None:
        results = await store.find_nodes(labels=["Nonexistent"])
        assert results == []


# ---------------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------------


class TestRelationships:
    async def test_create_relationship(self, store: InMemoryGraphStore) -> None:
        await store.create_node(GraphNode(id="a"))
        await store.create_node(GraphNode(id="b"))
        rel = GraphRelationship(id="r1", type="KNOWS", source_id="a", target_id="b")
        created = await store.create_relationship(rel)
        assert created.id == "r1"

    async def test_get_relationships_by_source(self, store: InMemoryGraphStore) -> None:
        await store.create_node(GraphNode(id="a"))
        await store.create_node(GraphNode(id="b"))
        await store.create_relationship(GraphRelationship(id="r1", type="KNOWS", source_id="a", target_id="b"))
        rels = await store.get_relationships(source_id="a")
        assert len(rels) == 1
        assert rels[0].type == "KNOWS"

    async def test_get_relationships_by_type(self, store: InMemoryGraphStore) -> None:
        await store.create_node(GraphNode(id="a"))
        await store.create_node(GraphNode(id="b"))
        await store.create_node(GraphNode(id="c"))
        await store.create_relationship(GraphRelationship(id="r1", type="KNOWS", source_id="a", target_id="b"))
        await store.create_relationship(GraphRelationship(id="r2", type="HATES", source_id="a", target_id="c"))
        rels = await store.get_relationships(type="KNOWS")
        assert len(rels) == 1

    async def test_node_delete_cascades(self, store: InMemoryGraphStore) -> None:
        await store.create_node(GraphNode(id="a"))
        await store.create_node(GraphNode(id="b"))
        await store.create_relationship(GraphRelationship(id="r1", type="KNOWS", source_id="a", target_id="b"))
        await store.delete_node("a")
        rels = await store.get_relationships()
        assert len(rels) == 0


# ---------------------------------------------------------------------------
# Traversal
# ---------------------------------------------------------------------------


class TestTraversal:
    async def test_simple_traversal(self, store: InMemoryGraphStore) -> None:
        # a -> b -> c
        for nid in ("a", "b", "c"):
            await store.create_node(GraphNode(id=nid))
        await store.create_relationship(GraphRelationship(id="r1", type="EDGE", source_id="a", target_id="b"))
        await store.create_relationship(GraphRelationship(id="r2", type="EDGE", source_id="b", target_id="c"))

        nodes = await store.traverse("a", max_depth=3)
        node_ids = {n.id for n in nodes}
        assert "b" in node_ids
        assert "c" in node_ids
        assert "a" not in node_ids  # start node excluded

    async def test_traversal_depth_limit(self, store: InMemoryGraphStore) -> None:
        for nid in ("a", "b", "c"):
            await store.create_node(GraphNode(id=nid))
        await store.create_relationship(GraphRelationship(id="r1", type="EDGE", source_id="a", target_id="b"))
        await store.create_relationship(GraphRelationship(id="r2", type="EDGE", source_id="b", target_id="c"))

        nodes = await store.traverse("a", max_depth=1)
        assert len(nodes) == 1
        assert nodes[0].id == "b"
