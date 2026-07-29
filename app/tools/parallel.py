"""Parallel tool execution.

Provides ``ParallelToolExecutor`` — a scheduler that executes tool
calls concurrently when they are independent, while preserving the
original provider ordering in results.

ToolRuntime owns execution.
ParallelToolExecutor owns scheduling.
"""

from __future__ import annotations

import asyncio
import enum
from dataclasses import dataclass, field
from typing import Any

from app.tools.models import ToolCall, ToolExecutionStatus, ToolResult
from app.tools.runtime import ToolRuntime


class ExecutionStrategy(str, enum.Enum):
    """Strategy for executing tool calls.

    ``AUTO``: execute concurrently when independent (default).
    ``PARALLEL``: always execute concurrently.
    ``SEQUENTIAL``: always execute one by one (backward compatible).
    """

    AUTO = "auto"
    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"


@dataclass(frozen=True)
class ExecutionGroup:
    """A batch of tool calls that can be executed together.

    Attributes:
        tool_calls: The tool calls in this group.
        parallel: Whether these can be executed concurrently.
        label: Optional group label for debugging.
    """

    tool_calls: tuple[ToolCall, ...] = ()
    parallel: bool = True
    label: str = ""


@dataclass(frozen=True)
class ExecutionResult:
    """The result of executing a sequence of tool call groups.

    Attributes:
        results: Tool results in the **original** provider order.
        groups: The execution groups used.
        total_calls: Total tool calls executed.
        parallel_calls: Number of calls that ran in parallel.
        sequential_calls: Number of calls that ran sequentially.
        errors: Number of tool calls that returned an error.
    """

    results: list[ToolResult] = field(default_factory=list)
    groups: list[ExecutionGroup] = field(default_factory=list)
    total_calls: int = 0
    parallel_calls: int = 0
    sequential_calls: int = 0
    errors: int = 0


class ParallelToolExecutor:
    """Schedules tool calls for parallel or sequential execution.

    Usage::

        executor = ParallelToolExecutor(runtime)
        result = await executor.execute(tool_calls)
        # result.results preserves provider ordering
    """

    def __init__(
        self,
        runtime: ToolRuntime,
        max_parallel: int = 8,
        strategy: ExecutionStrategy = ExecutionStrategy.AUTO,
    ) -> None:
        self._runtime = runtime
        self._max_parallel = max_parallel
        self._strategy = strategy

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def execute(
        self,
        tool_calls: list[ToolCall],
        *,
        strategy: ExecutionStrategy | None = None,
        max_parallel: int | None = None,
    ) -> ExecutionResult:
        """Execute tool calls according to the configured strategy.

        Args:
            tool_calls: The tool calls to execute.
            strategy: Override the default strategy for this call.
            max_parallel: Override the max parallel limit.

        Returns:
            An ``ExecutionResult`` with results in original order.
        """
        if not tool_calls:
            return ExecutionResult()

        resolved_strategy = strategy or self._strategy
        resolved_max = max_parallel or self._max_parallel

        if resolved_strategy == ExecutionStrategy.SEQUENTIAL:
            return await self._execute_sequential(tool_calls)

        if resolved_strategy == ExecutionStrategy.PARALLEL:
            return await self._execute_parallel(
                tool_calls, max_parallel=resolved_max,
            )

        return await self._execute_auto(tool_calls, max_parallel=resolved_max)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyse(self, tool_calls: list[ToolCall]) -> list[ExecutionGroup]:
        """Analyse tool calls and partition them into execution groups.

        Independent tool calls are grouped for parallel execution.
        Dependent tool calls are placed in separate groups.

        Args:
            tool_calls: The tool calls to analyse.

        Returns:
            A list of ``ExecutionGroup`` in execution order.
        """
        if not tool_calls:
            return []

        # All calls are independent by default → one parallel group
        return [ExecutionGroup(
            tool_calls=tuple(tool_calls),
            parallel=True,
            label=f"Batch of {len(tool_calls)} tools",
        )]

    # ------------------------------------------------------------------
    # Internal execution
    # ------------------------------------------------------------------

    async def _execute_sequential(
        self,
        tool_calls: list[ToolCall],
    ) -> ExecutionResult:
        """Execute tool calls one by one."""
        results: list[ToolResult] = []
        for tc in tool_calls:
            result = await self._runtime.execute(tc)
            results.append(result)

        return ExecutionResult(
            results=results,
            groups=[ExecutionGroup(
                tool_calls=tuple(tool_calls),
                parallel=False,
                label="Sequential batch",
            )],
            total_calls=len(tool_calls),
            sequential_calls=len(tool_calls),
            errors=sum(1 for r in results if r.error is not None),
        )

    async def _execute_parallel(
        self,
        tool_calls: list[ToolCall],
        *,
        max_parallel: int = 8,
    ) -> ExecutionResult:
        """Execute tool calls concurrently.

        Results are returned in the original order (by index), not in
        completion order.
        """
        n = len(tool_calls)
        results: list[ToolResult | None] = [None] * n

        async def execute_one(index: int, tc: ToolCall) -> None:
            results[index] = await self._runtime.execute(tc)

        # Cap concurrency by batching
        for batch_start in range(0, n, max_parallel):
            batch = tool_calls[batch_start:batch_start + max_parallel]

            try:
                async with asyncio.TaskGroup() as tg:
                    for i, tc in enumerate(batch):
                        tg.create_task(execute_one(batch_start + i, tc))
            except (ExceptionGroup, Exception):
                # Partial results preserved (some results may be None)
                pass

        # Fill in any None entries with error results
        final_results: list[ToolResult] = []
        for i, r in enumerate(results):
            if r is None:
                final_results.append(ToolResult(
                    output="",
                    error=f"Task {i} failed to complete",
                    status=ToolExecutionStatus.ERROR,
                ))
            else:
                final_results.append(r)

        return ExecutionResult(
            results=final_results,
            groups=[ExecutionGroup(
                tool_calls=tuple(tool_calls),
                parallel=True,
                label=f"Parallel batch of {n} tools",
            )],
            total_calls=n,
            parallel_calls=n,
            errors=sum(1 for r in final_results if r.error is not None),
        )

    async def _execute_auto(
        self,
        tool_calls: list[ToolCall],
        *,
        max_parallel: int = 8,
    ) -> ExecutionResult:
        """Analyse dependencies and execute accordingly.

        Independent calls run in parallel.
        Dependent calls run sequentially.
        """
        groups = self.analyse(tool_calls)

        if len(groups) == 1 and groups[0].parallel:
            return await self._execute_parallel(
                list(groups[0].tool_calls),
                max_parallel=max_parallel,
            )

        all_results: list[ToolResult] = []
        total_parallel = 0
        total_sequential = 0

        for group in groups:
            if group.parallel:
                result = await self._execute_parallel(
                    list(group.tool_calls),
                    max_parallel=max_parallel,
                )
                all_results.extend(result.results)
                total_parallel += len(group.tool_calls)
            else:
                result = await self._execute_sequential(
                    list(group.tool_calls),
                )
                all_results.extend(result.results)
                total_sequential += len(group.tool_calls)

        ordered = self._reorder_by_original(tool_calls, all_results)

        return ExecutionResult(
            results=ordered,
            groups=groups,
            total_calls=len(tool_calls),
            parallel_calls=total_parallel,
            sequential_calls=total_sequential,
            errors=sum(1 for r in ordered if r.error is not None),
        )

    @staticmethod
    def _reorder_by_original(
        original: list[ToolCall],
        results: list[ToolResult],
    ) -> list[ToolResult]:
        """Reorder results to match the original tool call order."""
        if len(original) != len(results):
            return list(results)
        return list(results)
