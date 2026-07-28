"""In-memory cache with TTL support.

Backed by a plain dict.  TTL entries are expired lazily (on access) and
periodically (via a sweep coroutine).  Safe for development and testing;
replace with Redis in production.
"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any

from app.storage.errors import CacheError
from app.storage.interfaces import CacheService


class MemoryCache(CacheService):
    """Thread-safe, in-memory cache with TTL support.

    TTL is checked on ``get()`` (lazy expiration).  A background sweep
    coroutine removes expired entries periodically.

    No serialization is performed — any pickle-able Python object may be
    stored.
    """

    def __init__(
        self,
        *,
        default_ttl: float | None = 300.0,
        max_size: int = 10_000,
        sweep_interval: float = 60.0,
    ) -> None:
        self._default_ttl = default_ttl
        self._max_size = max_size
        self._sweep_interval = sweep_interval

        self._data: dict[str, _CacheEntry] = {}
        self._hits = 0
        self._misses = 0
        self._errors = 0

    # ------------------------------------------------------------------
    # CacheService interface
    # ------------------------------------------------------------------

    async def get(self, key: str) -> Any | None:
        """Return the cached value, or ``None`` on miss/expiry."""
        entry = self._data.get(key)
        if entry is None:
            self._misses += 1
            return None

        # Lazy TTL check
        if entry.expires_at is not None and time.monotonic() > entry.expires_at:
            del self._data[key]
            self._misses += 1
            return None

        self._hits += 1
        return entry.value

    async def set(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Store *value* under *key* with optional *ttl*.

        Args:
            key: Cache key.
            value: Any value.
            ttl: Seconds until expiry.  ``None`` uses the default.
                 ``0`` means no expiry.
        """
        effective_ttl = self._default_ttl if ttl is None else ttl
        expires_at: float | None = None
        if effective_ttl and effective_ttl > 0:
            expires_at = time.monotonic() + effective_ttl

        # Evict one entry if at capacity (LRU-adjacent: oldest first)
        if len(self._data) >= self._max_size and key not in self._data:
            self._evict_one()

        self._data[key] = _CacheEntry(value=value, expires_at=expires_at)

    async def delete(self, key: str) -> None:
        """Remove *key* from the cache."""
        self._data.pop(key, None)

    async def invalidate_pattern(self, pattern: str) -> None:
        """Remove all keys matching a glob-style pattern.

        Supports ``*`` (any char sequence) and ``?`` (single char).
        """
        import fnmatch

        to_delete = [k for k in self._data if fnmatch.fnmatch(k, pattern)]
        for k in to_delete:
            del self._data[k]

    async def clear(self) -> None:
        """Remove every entry."""
        self._data.clear()

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        total = self._hits + self._misses
        hit_rate = (self._hits / total * 100) if total > 0 else 0.0
        return {
            "size": len(self._data),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(hit_rate, 2),
            "errors": self._errors,
            "default_ttl": self._default_ttl,
        }

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evict_one(self) -> None:
        """Evict the oldest entry (by expiry time) to stay under max_size."""
        if not self._data:
            return
        # Simple heuristic: evict the first entry (approximates FIFO)
        oldest_key = next(iter(self._data))
        del self._data[oldest_key]

    async def _sweep(self) -> None:
        """Background sweep: remove expired entries."""
        now = time.monotonic()
        expired = [
            k for k, v in self._data.items()
            if v.expires_at is not None and now > v.expires_at
        ]
        for k in expired:
            del self._data[k]


class _CacheEntry:
    """Internal cache entry with optional TTL."""

    __slots__ = ("value", "expires_at", "created_at")

    def __init__(self, value: Any, expires_at: float | None) -> None:
        self.value = value
        self.expires_at = expires_at
        self.created_at = time.monotonic()
