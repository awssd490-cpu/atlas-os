"""Tests for InMemoryVectorStore.

Verifies:
- Upsert and search returns nearest neighbors
- Cosine similarity returns correct ordering
- Empty store returns empty results
- Delete removes vectors
- Metadata filtering
- Namespace isolation
- List and count
"""

from __future__ import annotations

import pytest

from app.storage.interfaces import VectorRecord
from app.storage.vector.memory import InMemoryVectorStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


# ---------------------------------------------------------------------------
# Upsert / Search
# ---------------------------------------------------------------------------


class TestUpsertAndSearch:
    async def test_insert_and_search(self, store: InMemoryVectorStore) -> None:
        await store.upsert(VectorRecord(id="a", vector=[1.0, 0.0, 0.0]))
        await store.upsert(VectorRecord(id="b", vector=[0.0, 1.0, 0.0]))
        results = await store.search([1.0, 0.0, 0.0], limit=5)
        assert len(results) == 2
        assert results[0].id == "a"
        assert results[1].id == "b"

    async def test_update_existing(self, store: InMemoryVectorStore) -> None:
        await store.upsert(VectorRecord(id="x", vector=[1.0, 0.0]))
        await store.upsert(VectorRecord(id="x", vector=[0.0, 1.0]))
        results = await store.search([0.0, 1.0], limit=5)
        assert results[0].id == "x"

    async def test_search_empty(self, store: InMemoryVectorStore) -> None:
        results = await store.search([1.0, 0.0])
        assert results == []


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------


class TestSimilarity:
    async def test_identical_vectors(self, store: InMemoryVectorStore) -> None:
        await store.upsert(VectorRecord(id="a", vector=[1.0, 2.0, 3.0]))
        results = await store.search([1.0, 2.0, 3.0])
        assert len(results) == 1
        assert results[0].score > 0.99

    async def test_orthogonal_vectors(self, store: InMemoryVectorStore) -> None:
        await store.upsert(VectorRecord(id="a", vector=[1.0, 0.0]))
        await store.upsert(VectorRecord(id="b", vector=[0.0, 1.0]))
        results = await store.search([1.0, 0.0], limit=5)
        assert results[0].id == "a"
        assert results[1].id == "b"
        assert results[1].score == 0.0


# ---------------------------------------------------------------------------
# Delete
# ---------------------------------------------------------------------------


class TestDelete:
    async def test_delete_removes_vector(self, store: InMemoryVectorStore) -> None:
        await store.upsert(VectorRecord(id="del", vector=[1.0, 0.0]))
        assert await store.count() == 1
        await store.delete("del")
        assert await store.count() == 0

    async def test_delete_nonexistent(self, store: InMemoryVectorStore) -> None:
        await store.delete("nothing")


# ---------------------------------------------------------------------------
# Metadata filtering
# ---------------------------------------------------------------------------


class TestMetadataFilter:
    async def test_filter_matches(self, store: InMemoryVectorStore) -> None:
        await store.upsert(VectorRecord(id="a", vector=[1.0, 0.0], metadata={"type": "x"}))
        await store.upsert(VectorRecord(id="b", vector=[0.0, 1.0], metadata={"type": "y"}))
        results = await store.search([1.0, 0.0], filter={"type": "x"})
        assert len(results) == 1
        assert results[0].id == "a"

    async def test_filter_no_match(self, store: InMemoryVectorStore) -> None:
        await store.upsert(VectorRecord(id="a", vector=[1.0, 0.0], metadata={"type": "x"}))
        results = await store.search([1.0, 0.0], filter={"type": "z"})
        assert results == []


# ---------------------------------------------------------------------------
# Namespace isolation
# ---------------------------------------------------------------------------


class TestNamespace:
    async def test_namespace_isolation(self, store: InMemoryVectorStore) -> None:
        await store.upsert(VectorRecord(id="a", vector=[1.0, 0.0], namespace="ns1"))
        await store.upsert(VectorRecord(id="b", vector=[0.0, 1.0], namespace="ns2"))
        assert await store.count(namespace="ns1") == 1
        assert await store.count(namespace="ns2") == 1

    async def test_search_in_namespace(self, store: InMemoryVectorStore) -> None:
        await store.upsert(VectorRecord(id="a", vector=[1.0, 0.0], namespace="ns1"))
        await store.upsert(VectorRecord(id="b", vector=[1.0, 0.0], namespace="ns2"))
        results = await store.search([1.0, 0.0], namespace="ns1")
        assert len(results) == 1
        assert results[0].id == "a"


# ---------------------------------------------------------------------------
# List and count
# ---------------------------------------------------------------------------


class TestListCount:
    async def test_list_ids(self, store: InMemoryVectorStore) -> None:
        await store.upsert(VectorRecord(id="a", vector=[1.0, 0.0]))
        await store.upsert(VectorRecord(id="b", vector=[0.0, 1.0]))
        ids = await store.list_ids()
        assert sorted(ids) == ["a", "b"]

    async def test_count(self, store: InMemoryVectorStore) -> None:
        await store.upsert(VectorRecord(id="a", vector=[1.0, 0.0]))
        assert await store.count() == 1
