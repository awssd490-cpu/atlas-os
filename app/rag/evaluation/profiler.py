"""PerformanceProfiler — lightweight execution time and memory profiler.

Uses ``time.perf_counter()`` for wall-clock timing and ``tracemalloc``
for memory measurement.  No external dependencies.
"""

from __future__ import annotations

import time
import tracemalloc
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Mapping

from app.rag.evaluation.errors import EvaluationError


# ---------------------------------------------------------------------------
# PerformanceProfile
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PerformanceProfile:
    """Immutable result of a single profiling run.

    Attributes:
        execution_time_ms: Wall-clock execution time in milliseconds.
        peak_memory_bytes: Peak memory usage in bytes during execution.
        current_memory_bytes: Current memory usage in bytes at the
            time of measurement.
        metadata: Additional metadata about the profiling run (e.g.
            component name, argument summary).
    """

    execution_time_ms: float = 0.0
    peak_memory_bytes: int = 0
    current_memory_bytes: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# PerformanceProfiler
# ---------------------------------------------------------------------------


class PerformanceProfiler:
    """Lightweight profiler for measuring execution time and memory.

    Wraps a callable (sync or async), measures its execution with
    ``time.perf_counter()`` and its memory footprint with
    ``tracemalloc``, and returns an immutable ``PerformanceProfile``.

    ``tracemalloc`` is started on first use if not already active,
    and left running for subsequent calls to allow cumulative tracing.

    Usage::

        profiler = PerformanceProfiler()

        # Profile a sync function
        profile = await profiler.profile(my_sync_fn, arg1, arg2)

        # Profile an async function
        profile = await profiler.profile(my_async_fn, arg1, arg2)

        print(f"Executed in {profile.execution_time_ms:.1f} ms")
        print(f"Peak memory: {profile.peak_memory_bytes} bytes")
    """

    def __init__(self) -> None:
        self._component_name: str = ""

    # ------------------------------------------------------------------
    # Profile
    # ------------------------------------------------------------------

    async def profile(
        self,
        component: Callable[..., Any] | Callable[..., Awaitable[Any]],
        *args: Any,
        **kwargs: Any,
    ) -> PerformanceProfile:
        """Profile a callable.

        Supports both synchronous and asynchronous callables.  Memory
        is measured with ``tracemalloc`` before and after execution.

        Args:
            component: The callable to profile (sync or async).
            *args: Positional arguments forwarded to *component*.
            **kwargs: Keyword arguments forwarded to *component*.

        Returns:
            A ``PerformanceProfile`` with timing and memory data.

        Raises:
            EvaluationError: If profiling fails.
        """
        if not callable(component):
            raise EvaluationError(
                "Profiler requires a callable component",
                details={"received_type": type(component).__name__},
            )

        self._component_name = getattr(component, "__name__", str(component))

        # Ensure tracemalloc is active
        if not tracemalloc.is_tracing():
            tracemalloc.start()

        # Pre-execution memory snapshot
        before_current, before_peak = tracemalloc.get_traced_memory()
        time_start = time.perf_counter()

        exception: BaseException | None = None

        try:
            val = component(*args, **kwargs)
            if isinstance(val, Awaitable):
                await val
            _ = val  # keep reference to prevent premature GC
        except BaseException as exc:
            exception = exc

        # Post-execution measurement — always runs even on exception
        time_end = time.perf_counter()
        after_current, after_peak = tracemalloc.get_traced_memory()

        metadata: dict[str, Any] = {
            "component": self._component_name,
            "args": str(args) if args else "",
            "kwargs": str(kwargs) if kwargs else "",
        }

        # If the component raised, propagate the exception
        if exception is not None:
            raise EvaluationError(
                f"Profiled component raised: {exception}",
                details={
                    "component": self._component_name,
                    "original_error": str(exception),
                },
            ) from exception

        execution_time_ms = (time_end - time_start) * 1000.0
        peak_memory = after_peak - before_peak
        current_memory = after_current - before_current

        return PerformanceProfile(
            execution_time_ms=execution_time_ms,
            peak_memory_bytes=max(0, peak_memory),
            current_memory_bytes=max(0, current_memory),
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Tracing control
    # ------------------------------------------------------------------

    @staticmethod
    def start_tracing() -> None:
        """Start ``tracemalloc`` tracing if not already active."""
        if not tracemalloc.is_tracing():
            tracemalloc.start()

    @staticmethod
    def stop_tracing() -> None:
        """Stop ``tracemalloc`` tracing and clear traces."""
        if tracemalloc.is_tracing():
            tracemalloc.stop()

    @staticmethod
    def get_traced_memory() -> tuple[int, int]:
        """Return ``(current, peak)`` memory in bytes.

        Returns ``(0, 0)`` if tracing is not active.
        """
        if tracemalloc.is_tracing():
            return tracemalloc.get_traced_memory()
        return (0, 0)
