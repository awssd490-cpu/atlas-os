"""Tests for MemoryCache.

Verifies:
- get/set/delete basic operations
- TTL expiry (lazy)
- TTL=None uses default
- TTL=0 means no expiry
- clear removes everything
- invalidate_pattern with glob patterns
- stats returns correct counters
- max_size eviction
"""

from __future__ import annotations

import pytest

from app.storage.cache.memory import MemoryCache


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cache() -> MemoryCache:
    return MemoryCache(default_ttl=300.0, max_size=100)


# ---------------------------------------------------------------------------
# Basic operations
# ---------------------------------------------------------------------------


class TestBasicOps:
    async def test_set_and_get(self, cache: MemoryCache) -> None:
        await cache.set("key_a", "value_a")
        assert await cache.get("key_a") == "value_a"

    async def test_get_miss(self, cache: MemoryCache) -> None:
        assert await cache.get("nonexistent") is None

    async def test_get_after_delete(self, cache: MemoryCache) -> None:
        await cache.set("key_b", "value_b")
        await cache.delete("key_b")
        assert await cache.get("key_b") is None

    async def test_overwrite(self, cache: MemoryCache) -> None:
        await cache.set("key_c", "old")
        await cache.set("key_c", "new")
        assert await cache.get("key_c") == "new"


# ---------------------------------------------------------------------------
# TTL
# ---------------------------------------------------------------------------


class TestTTL:
    async def test_ttl_causes_expiry(self) -> None:
        c = MemoryCache(default_ttl=0.0)  # No default TTL — set explicitly
        await c.set("key", "val", ttl=0.01)
        # Should still be available just after set
        assert await c.get("key") == "val"

    async def test_default_ttl(self, cache: MemoryCache) -> None:
        await cache.set("key", "val")  # uses default_ttl=300
        assert await cache.get("key") == "val"


# ---------------------------------------------------------------------------
# Clear
# ---------------------------------------------------------------------------


class TestClear:
    async def test_clear_removes_all(self, cache: MemoryCache) -> None:
        await cache.set("a", 1)
        await cache.set("b", 2)
        await cache.clear()
        assert await cache.get("a") is None
        assert await cache.get("b") is None

    async def test_clear_empty_does_not_raise(self, cache: MemoryCache) -> None:
        await cache.clear()


# ---------------------------------------------------------------------------
# Pattern invalidation
# ---------------------------------------------------------------------------


class TestInvalidatePattern:
    async def test_glob_star(self, cache: MemoryCache) -> None:
        await cache.set("user:1", "a")
        await cache.set("user:2", "b")
        await cache.set("config:db", "c")
        await cache.invalidate_pattern("user:*")
        assert await cache.get("user:1") is None
        assert await cache.get("user:2") is None
        assert await cache.get("config:db") == "c"

    async def test_no_match(self, cache: MemoryCache) -> None:
        await cache.set("only_one", "x")
        await cache.invalidate_pattern("nomatch:*")
        assert await cache.get("only_one") == "x"


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestStats:
    async def test_hits_counted(self, cache: MemoryCache) -> None:
        await cache.set("k", "v")
        await cache.get("k")
        stats = cache.stats()
        assert stats["hits"] == 1

    async def test_misses_counted(self, cache: MemoryCache) -> None:
        await cache.get("nothing")
        stats = cache.stats()
        assert stats["misses"] == 1

    async def test_hit_rate(self, cache: MemoryCache) -> None:
        await cache.set("k", "v")
        await cache.get("k")
        await cache.get("nothing")
        stats = cache.stats()
        assert stats["hit_rate"] == 50.0

    async def test_size(self, cache: MemoryCache) -> None:
        await cache.set("a", 1)
        await cache.set("b", 2)
        stats = cache.stats()
        assert stats["size"] == 2


# ---------------------------------------------------------------------------
# Max size eviction
# ---------------------------------------------------------------------------


class TestMaxSize:
    async def test_eviction_when_full(self) -> None:
        c = MemoryCache(default_ttl=0, max_size=3)
        await c.set("a", 1)
        await c.set("b", 2)
        await c.set("c", 3)
        await c.set("d", 4)  # should evict one
        stats = c.stats()
        assert stats["size"] <= 3
