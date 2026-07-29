"""Tests for parallel tool execution."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.tools.models import (
    ToolCall,
    ToolDefinition,
    ToolExecutionStatus,
    ToolParameter,
    ToolResult,
)
from app.tools.parallel import (
    ExecutionGroup,
    ExecutionResult,
    ExecutionStrategy,
    ParallelToolExecutor,
)
from app.tools.registry import ToolRegistry
from app.tools.runtime import ToolRuntime


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _fast_tool(value: str = "") -> str:
    """A fast tool that returns immediately."""
    return f"result:{value}"


async def _slow_tool(delay: float = 0.05, value: str = "") -> str:
    """A tool that simulates work by sleeping."""
    await asyncio.sleep(delay)
    return f"slow:{value}"


def _failing_tool(message: str = "error") -> str:
    """A tool that always fails."""
    raise ValueError(message)


def _create_runtime() -> ToolRuntime:
    registry = ToolRegistry()

    registry.register(ToolDefinition(
        name="fast",
        description="Fast tool",
        parameters=(ToolParameter(name="value", type="string", required=False),),
        fn=_fast_tool,
    ))
    registry.register(ToolDefinition(
        name="slow",
        description="Slow tool",
        parameters=(
            ToolParameter(name="delay", type="number", required=False, default=0.05),
            ToolParameter(name="value", type="string", required=False),
        ),
        fn=_slow_tool,
    ))
    registry.register(ToolDefinition(
        name="failing",
        description="Failing tool",
        parameters=(ToolParameter(name="message", type="string", required=False),),
        fn=_failing_tool,
    ))
    return ToolRuntime(registry)


def _tc(name: str, args: dict[str, Any] | None = None, tc_id: str = "") -> ToolCall:
    return ToolCall(
        id=tc_id or f"call_{name}",
        name=name,
        arguments=args or {},
    )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class TestExecutionGroup:
    def test_create(self) -> None:
        group = ExecutionGroup(
            tool_calls=(_tc("a"), _tc("b")),
            parallel=True,
            label="Test",
        )
        assert len(group.tool_calls) == 2
        assert group.parallel is True
        assert group.label == "Test"


class TestExecutionResult:
    def test_defaults(self) -> None:
        result = ExecutionResult()
        assert result.results == []
        assert result.total_calls == 0


class TestExecutionStrategy:
    def test_enum_values(self) -> None:
        assert ExecutionStrategy.AUTO.value == "auto"
        assert ExecutionStrategy.PARALLEL.value == "parallel"
        assert ExecutionStrategy.SEQUENTIAL.value == "sequential"


# ---------------------------------------------------------------------------
# ParallelToolExecutor
# ---------------------------------------------------------------------------


class TestParallelToolExecutor:
    def test_init(self) -> None:
        runtime = _create_runtime()
        executor = ParallelToolExecutor(runtime)
        assert executor._runtime is runtime
        assert executor._max_parallel == 8
        assert executor._strategy == ExecutionStrategy.AUTO


# ---------------------------------------------------------------------------
# Sequential execution
# ---------------------------------------------------------------------------


class TestSequentialExecution:
    async def test_single_tool(self) -> None:
        runtime = _create_runtime()
        executor = ParallelToolExecutor(runtime)
        result = await executor.execute(
            [_tc("fast", {"value": "hello"})],
            strategy=ExecutionStrategy.SEQUENTIAL,
        )
        assert len(result.results) == 1
        assert result.results[0].output == "result:hello"
        assert result.total_calls == 1
        assert result.sequential_calls == 1
        assert result.parallel_calls == 0

    async def test_multiple_tools_sequential(self) -> None:
        runtime = _create_runtime()
        executor = ParallelToolExecutor(runtime)
        result = await executor.execute(
            [_tc("fast", {"value": "a"}), _tc("fast", {"value": "b"})],
            strategy=ExecutionStrategy.SEQUENTIAL,
        )
        assert len(result.results) == 2
        assert result.results[0].output == "result:a"
        assert result.results[1].output == "result:b"
        assert result.sequential_calls == 2

    async def test_empty_list(self) -> None:
        runtime = _create_runtime()
        executor = ParallelToolExecutor(runtime)
        result = await executor.execute([], strategy=ExecutionStrategy.SEQUENTIAL)
        assert len(result.results) == 0


# ---------------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------------


class TestParallelExecution:
    async def test_single_tool_parallel(self) -> None:
        runtime = _create_runtime()
        executor = ParallelToolExecutor(runtime)
        result = await executor.execute(
            [_tc("fast", {"value": "x"})],
            strategy=ExecutionStrategy.PARALLEL,
        )
        assert result.results[0].output == "result:x"
        assert result.parallel_calls == 1

    async def test_multiple_parallel_tools(self) -> None:
        runtime = _create_runtime()
        executor = ParallelToolExecutor(runtime)
        result = await executor.execute(
            [_tc("fast", {"value": "a"}), _tc("fast", {"value": "b"})],
            strategy=ExecutionStrategy.PARALLEL,
        )
        assert len(result.results) == 2
        assert result.results[0].output == "result:a"
        assert result.results[1].output == "result:b"
        assert result.parallel_calls == 2

    async def test_parallel_preserves_order(self) -> None:
        """Results returned in original order even with slow tools."""
        runtime = _create_runtime()
        executor = ParallelToolExecutor(runtime)
        result = await executor.execute(
            [
                _tc("slow", {"delay": 0.1, "value": "first"}, tc_id="c1"),
                _tc("fast", {"value": "second"}, tc_id="c2"),
            ],
            strategy=ExecutionStrategy.PARALLEL,
        )
        # First tool is slower but should still be first in results
        assert result.results[0].output == "slow:first"
        assert result.results[1].output == "result:second"

    async def test_parallel_result_counts(self) -> None:
        runtime = _create_runtime()
        executor = ParallelToolExecutor(runtime)
        result = await executor.execute(
            [_tc("fast"), _tc("fast"), _tc("fast")],
            strategy=ExecutionStrategy.PARALLEL,
        )
        assert result.total_calls == 3
        assert result.parallel_calls == 3
        assert result.sequential_calls == 0


# ---------------------------------------------------------------------------
# AUTO strategy
# ---------------------------------------------------------------------------


class TestAutoStrategy:
    async def test_auto_becomes_parallel(self) -> None:
        """AUTO should parallelize independent tools."""
        runtime = _create_runtime()
        executor = ParallelToolExecutor(runtime)
        result = await executor.execute(
            [_tc("fast", {"value": "a"}), _tc("fast", {"value": "b"})],
            strategy=ExecutionStrategy.AUTO,
        )
        assert len(result.results) == 2
        assert result.parallel_calls >= 1


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestParallelErrors:
    async def test_failing_tool_in_parallel(self) -> None:
        """A single failing tool doesn't crash the batch."""
        runtime = _create_runtime()
        executor = ParallelToolExecutor(runtime)
        result = await executor.execute(
            [
                _tc("fast", {"value": "ok"}),
                _tc("failing", {"message": "boom"}),
                _tc("fast", {"value": "also ok"}),
            ],
            strategy=ExecutionStrategy.PARALLEL,
        )
        assert len(result.results) == 3
        assert result.results[0].output == "result:ok"
        assert result.results[1].error is not None
        assert result.results[2].output == "result:also ok"
        assert result.errors >= 1

    async def test_all_failing(self) -> None:
        """All tools failing still returns results."""
        runtime = _create_runtime()
        executor = ParallelToolExecutor(runtime)
        result = await executor.execute(
            [
                _tc("failing", {"message": "fail1"}),
                _tc("failing", {"message": "fail2"}),
            ],
            strategy=ExecutionStrategy.PARALLEL,
        )
        assert len(result.results) == 2
        assert result.errors == 2


# ---------------------------------------------------------------------------
# Concurrency limit
# ---------------------------------------------------------------------------


class TestConcurrencyLimit:
    async def test_max_parallel_respected(self) -> None:
        """Maximum parallel limit is respected."""
        runtime = _create_runtime()
        executor = ParallelToolExecutor(runtime, max_parallel=2)
        result = await executor.execute(
            [_tc("fast", {"value": str(i)}) for i in range(5)],
            strategy=ExecutionStrategy.PARALLEL,
        )
        assert len(result.results) == 5
        assert result.parallel_calls == 5


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


class TestAnalysis:
    def test_analyse_empty(self) -> None:
        runtime = _create_runtime()
        executor = ParallelToolExecutor(runtime)
        assert executor.analyse([]) == []

    def test_analyse_independent(self) -> None:
        runtime = _create_runtime()
        executor = ParallelToolExecutor(runtime)
        groups = executor.analyse([_tc("a"), _tc("b")])
        assert len(groups) == 1
        assert groups[0].parallel is True
        assert len(groups[0].tool_calls) == 2


# ---------------------------------------------------------------------------
# Runtime integration (via integration.execute_tool_calls)
# ---------------------------------------------------------------------------


class TestRuntimeIntegration:
    async def test_parallel_via_integration(self) -> None:
        """Integration layer supports parallel flag."""
        from app.tools.integration import execute_tool_calls

        runtime = _create_runtime()
        messages = await execute_tool_calls(
            [_tc("fast", {"value": "a"}), _tc("fast", {"value": "b"})],
            runtime,
            parallel=True,
            max_parallel=2,
        )
        assert len(messages) == 2
        assert messages[0].content == "result:a"
        assert messages[1].content == "result:b"

    async def test_sequential_via_integration_default(self) -> None:
        """Default integration remains sequential."""
        from app.tools.integration import execute_tool_calls

        runtime = _create_runtime()
        messages = await execute_tool_calls(
            [_tc("fast", {"value": "x"})],
            runtime,
        )
        assert len(messages) == 1
        assert messages[0].content == "result:x"

    async def test_empty_list_integration(self) -> None:
        from app.tools.integration import execute_tool_calls

        runtime = _create_runtime()
        assert await execute_tool_calls([], runtime) == []

    async def test_single_tool_parallel(self) -> None:
        """Single tool with parallel flag still works."""
        from app.tools.integration import execute_tool_calls

        runtime = _create_runtime()
        messages = await execute_tool_calls(
            [_tc("fast", {"value": "only"})],
            runtime,
            parallel=True,
        )
        assert len(messages) == 1
        assert messages[0].content == "result:only"


# ---------------------------------------------------------------------------
# Integration with AgentRuntime config
# ---------------------------------------------------------------------------


class TestAgentRuntimeConfigIntegration:
    async def test_parallel_enabled_default(self) -> None:
        from app.agent.config import AgentConfig

        config = AgentConfig.default()
        assert config.parallel_tools_enabled is True
        assert config.max_parallel_tools == 8
        assert config.execution_strategy == "auto"

    async def test_invalid_execution_strategy(self) -> None:
        from app.agent.config import AgentConfig

        with pytest.raises(ValueError):
            AgentConfig(execution_strategy="dag")

    async def test_invalid_max_parallel(self) -> None:
        from app.agent.config import AgentConfig

        with pytest.raises(ValueError):
            AgentConfig(max_parallel_tools=0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    async def test_very_large_batch(self) -> None:
        """A large batch of independent tools executes successfully."""
        runtime = _create_runtime()
        executor = ParallelToolExecutor(runtime, max_parallel=4)
        calls = [_tc("fast", {"value": str(i)}, tc_id=f"c{i}") for i in range(20)]
        result = await executor.execute(calls, strategy=ExecutionStrategy.PARALLEL)
        assert len(result.results) == 20
        assert result.parallel_calls == 20

    async def test_mixed_success_and_failure(self) -> None:
        """Mixed success/failure preserves all results."""
        runtime = _create_runtime()
        executor = ParallelToolExecutor(runtime)
        result = await executor.execute(
            [
                _tc("fast", {"value": "ok1"}),
                _tc("failing", {"message": "fail"}),
                _tc("fast", {"value": "ok2"}),
                _tc("failing", {"message": "fail2"}),
                _tc("fast", {"value": "ok3"}),
            ],
            strategy=ExecutionStrategy.PARALLEL,
        )
        assert len(result.results) == 5
        assert result.results[0].output == "result:ok1"
        assert result.results[1].error is not None
        assert result.results[2].output == "result:ok2"
        assert result.results[3].error is not None
        assert result.results[4].output == "result:ok3"
        assert result.errors == 2
