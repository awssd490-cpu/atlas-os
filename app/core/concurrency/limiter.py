"""ConcurrencyLimiter — lightweight async concurrency limiter.

Wraps ``asyncio.Semaphore`` with a clean API for controlling
concurrent access to a shared resource.
"""

from __future__ import annotations

import asyncio


class ConcurrencyLimiter:
    """Async concurrency limiter backed by ``asyncio.Semaphore``.

    Limits the number of concurrent operations accessing a shared
    resource.

    Usage::

        limiter = ConcurrencyLimiter(5)  # max 5 concurrent

        async def worker() -> None:
            async with limiter:
                # … access shared resource …

        # Manual acquire / release
        await limiter.acquire()
        try:
            # … access shared resource …
        finally:
            limiter.release()
    """

    def __init__(self, max_concurrent: int = 10) -> None:
        if max_concurrent < 1:
            raise ValueError(
                f"max_concurrent must be at least 1, got {max_concurrent}"
            )
        self._semaphore = asyncio.Semaphore(max_concurrent)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> ConcurrencyLimiter:
        await self._semaphore.acquire()
        return self

    async def __aexit__(
        self,
        exc_type: object,
        exc_val: object,
        exc_tb: object,
    ) -> None:
        self._semaphore.release()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def acquire(self) -> None:
        """Acquire a permit, blocking until one is available."""
        await self._semaphore.acquire()

    def release(self) -> None:
        """Release a permit.

        Raises:
            ValueError: If the semaphore is already fully released.
        """
        self._semaphore.release()

    @property
    def available_permits(self) -> int:
        """Return the number of available permits."""
        return self._semaphore._value  # type: ignore[attr-defined]
