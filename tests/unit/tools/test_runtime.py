"""Tests for ToolRuntime."""

from __future__ import annotations

import pytest

from app.tools.errors import ToolNotFoundError
from app.tools.models import (
    ToolCall,
    ToolDefinition,
    ToolExecutionStatus,
    ToolParameter,
    ToolResult,
)
from app.tools.registry import ToolRegistry
from app.tools.runtime import ToolRuntime


# ---------------------------------------------------------------------------
# Helper tools
# ---------------------------------------------------------------------------


async def async_echo(text: str) -> str:
    """Async tool that echoes input."""
    return f"echo: {text}"


def sync_echo(text: str) -> str:
    """Sync tool that echoes input."""
    return f"sync: {text}"


def failing_tool(message: str = "error") -> str:
    """Tool that always raises."""
    raise ValueError(message)


def calculator(expression: str = "") -> str:
    """Evaluate a simple expression."""
    parts = expression.split("+")
    total = sum(int(p.strip()) for p in parts)
    return str(total)


def multi_param_tool(a: str, b: int, c: float = 1.0) -> str:
    """Tool with multiple parameters."""
    return f"{a}:{b}:{c}"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(ToolDefinition(
        name="async_echo",
        description="Async echo tool",
        parameters=(ToolParameter(name="text", type="string"),),
        fn=async_echo,
    ))
    reg.register(ToolDefinition(
        name="sync_echo",
        description="Sync echo tool",
        parameters=(ToolParameter(name="text", type="string"),),
        fn=sync_echo,
    ))
    reg.register(ToolDefinition(
        name="failing",
        description="Failing tool",
        parameters=(ToolParameter(name="message", type="string", required=False),),
        fn=failing_tool,
    ))
    reg.register(ToolDefinition(
        name="calculator",
        description="Calculator tool",
        parameters=(ToolParameter(name="expression", type="string"),),
        fn=calculator,
    ))
    reg.register(ToolDefinition(
        name="multi",
        description="Multi param tool",
        parameters=(
            ToolParameter(name="a", type="string"),
            ToolParameter(name="b", type="integer"),
            ToolParameter(name="c", type="number", required=False, default=1.0),
        ),
        fn=multi_param_tool,
    ))
    return reg


@pytest.fixture
def runtime(registry: ToolRegistry) -> ToolRuntime:
    return ToolRuntime(registry)


# ---------------------------------------------------------------------------
# Execution tests
# ---------------------------------------------------------------------------


class TestToolRuntime:
    async def test_execute_async_tool(self, runtime: ToolRuntime) -> None:
        result = await runtime.execute(
            ToolCall(name="async_echo", arguments={"text": "hello"})
        )
        assert result.status == ToolExecutionStatus.SUCCESS
        assert result.output == "echo: hello"
        assert result.error is None
        # Duration may be 0 for very fast tools, but should not be negative
        assert result.duration_ms >= 0

    async def test_execute_sync_tool(self, runtime: ToolRuntime) -> None:
        result = await runtime.execute(
            ToolCall(name="sync_echo", arguments={"text": "world"})
        )
        assert result.status == ToolExecutionStatus.SUCCESS
        assert result.output == "sync: world"

    async def test_execute_with_arguments(self, runtime: ToolRuntime) -> None:
        result = await runtime.execute(
            ToolCall(name="calculator", arguments={"expression": "1+2+3"})
        )
        assert result.status == ToolExecutionStatus.SUCCESS
        assert result.output == "6"

    async def test_execute_tool_not_found(self, runtime: ToolRuntime) -> None:
        result = await runtime.execute(
            ToolCall(name="nonexistent", arguments={})
        )
        assert result.status == ToolExecutionStatus.ERROR
        assert result.error is not None
        assert "nonexistent" in result.error
        assert result.output == ""

    async def test_execute_tool_not_found_class(self, runtime: ToolRuntime) -> None:
        result = await runtime.execute(
            ToolCall(name="does_not_exist", arguments={})
        )
        assert result.status == ToolExecutionStatus.ERROR

    async def test_execute_with_missing_required_args(
        self, runtime: ToolRuntime
    ) -> None:
        result = await runtime.execute(
            ToolCall(name="calculator", arguments={})
        )
        assert result.status == ToolExecutionStatus.ERROR
        assert "Missing required" in (result.error or "")

    async def test_execute_with_unknown_args(self, runtime: ToolRuntime) -> None:
        result = await runtime.execute(
            ToolCall(name="sync_echo", arguments={"text": "hi", "extra": "bad"})
        )
        assert result.status == ToolExecutionStatus.ERROR
        assert "Unknown parameter" in (result.error or "")

    async def test_execute_tool_raises_exception(
        self, runtime: ToolRuntime
    ) -> None:
        result = await runtime.execute(
            ToolCall(name="failing", arguments={"message": "boom"})
        )
        assert result.status == ToolExecutionStatus.ERROR
        assert result.error is not None
        # Error message indicates execution failed
        assert "execution failed" in result.error

    async def test_execute_with_type_validation(
        self, runtime: ToolRuntime
    ) -> None:
        """Type validation catches wrong types."""
        result = await runtime.execute(
            ToolCall(name="multi", arguments={"a": "text", "b": "not_an_int"})
        )
        assert result.status == ToolExecutionStatus.ERROR
        assert "expected integer" in (result.error or "")

    async def test_execute_with_optional_params(
        self, runtime: ToolRuntime
    ) -> None:
        """Optional parameters can be omitted."""
        result = await runtime.execute(
            ToolCall(name="multi", arguments={"a": "hello", "b": 42})
        )
        assert result.status == ToolExecutionStatus.SUCCESS
        assert result.output == "hello:42:1.0"

    async def test_execute_with_all_params(self, runtime: ToolRuntime) -> None:
        """All parameters provided including optional."""
        result = await runtime.execute(
            ToolCall(name="multi", arguments={"a": "hello", "b": 42, "c": 3.14})
        )
        assert result.status == ToolExecutionStatus.SUCCESS
        assert result.output == "hello:42:3.14"


# ---------------------------------------------------------------------------
# Execution with record tests
# ---------------------------------------------------------------------------


class TestToolRuntimeWithRecord:
    async def test_execute_with_record_success(
        self, runtime: ToolRuntime
    ) -> None:
        record = await runtime.execute_with_record(
            ToolCall(name="sync_echo", arguments={"text": "test"})
        )
        assert record.result.status == ToolExecutionStatus.SUCCESS
        assert record.result.output == "sync: test"
        assert record.definition is not None
        assert record.definition.name == "sync_echo"
        assert record.start_time > 0
        assert record.end_time >= record.start_time

    async def test_execute_with_record_not_found(
        self, runtime: ToolRuntime
    ) -> None:
        record = await runtime.execute_with_record(
            ToolCall(name="missing", arguments={})
        )
        assert record.result.status == ToolExecutionStatus.ERROR
        assert record.definition is None

    async def test_execute_with_record_call_preserved(
        self, runtime: ToolRuntime
    ) -> None:
        call = ToolCall(id="test_id", name="calculator", arguments={"expression": "2+2"})
        record = await runtime.execute_with_record(call)
        assert record.tool_call.id == "test_id"
        assert record.tool_call.name == "calculator"
        assert record.tool_call.arguments == {"expression": "2+2"}


# ---------------------------------------------------------------------------
# Runtime creation
# ---------------------------------------------------------------------------


class TestCreateRuntime:
    async def test_create_runtime_with_fresh_registry(self) -> None:
        from app.tools.runtime import create_runtime

        runtime = create_runtime()
        assert runtime.registry is not None
        assert runtime.registry.count() == 0

    async def test_create_runtime_with_existing_registry(self) -> None:
        from app.tools.runtime import create_runtime

        reg = ToolRegistry()
        reg.register(ToolDefinition(
            name="test",
            parameters=(ToolParameter(name="x", type="string"),),
            fn=lambda x: x,
        ))
        runtime = create_runtime(reg)
        assert runtime.registry.count() == 1
        assert runtime.registry.exists("test")


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestToolRuntimeEdgeCases:
    async def test_execute_with_no_fn(self) -> None:
        """ToolDefinition with no callable returns error."""
        registry = ToolRegistry()
        # Register via internal dict to bypass fn validation for testing
        no_fn_def = ToolDefinition(
            name="empty",
            description="No callable",
            parameters=(ToolParameter(name="x", type="string"),),
            fn=None,
        )
        registry._tools["empty"] = no_fn_def
        runtime = ToolRuntime(registry)
        result = await runtime.execute(
            ToolCall(name="empty", arguments={"x": "hello"})
        )
        assert result.status == ToolExecutionStatus.ERROR
        assert "no callable" in (result.error or "")

    async def test_execute_returns_none(self) -> None:
        """Tool that returns None produces empty string output."""
        registry = ToolRegistry()
        async def returns_none() -> None:
            return None

        registry.register(ToolDefinition(
            name="none_returns",
            fn=returns_none,
        ))
        runtime = ToolRuntime(registry)
        result = await runtime.execute(
            ToolCall(name="none_returns", arguments={})
        )
        assert result.status == ToolExecutionStatus.SUCCESS
        assert result.output == ""

    async def test_execute_with_empty_args(self) -> None:
        """Tool with no required params can be called with empty args."""
        registry = ToolRegistry()

        def no_params() -> str:
            return "done"

        registry.register(ToolDefinition(
            name="no_params",
            fn=no_params,
        ))
        runtime = ToolRuntime(registry)
        result = await runtime.execute(ToolCall(name="no_params", arguments={}))
        assert result.status == ToolExecutionStatus.SUCCESS
        assert result.output == "done"

    async def test_runtime_never_crashes_on_exception(self) -> None:
        """Runtime catches all exceptions from tool functions."""

        def crash() -> str:
            raise RuntimeError("catastrophic failure")

        registry = ToolRegistry()
        registry.register(ToolDefinition(name="crash", fn=crash))
        runtime = ToolRuntime(registry)
        result = await runtime.execute(ToolCall(name="crash", arguments={}))
        assert result.status == ToolExecutionStatus.ERROR
        assert result.error is not None

    async def test_type_validation_string(self) -> None:
        registry = ToolRegistry()
        registry.register(ToolDefinition(
            name="str_tool",
            parameters=(ToolParameter(name="val", type="string"),),
            fn=lambda val: val,
        ))
        runtime = ToolRuntime(registry)

        # Wrong type
        result = await runtime.execute(
            ToolCall(name="str_tool", arguments={"val": 42})
        )
        assert result.status == ToolExecutionStatus.ERROR

        # Correct type
        result = await runtime.execute(
            ToolCall(name="str_tool", arguments={"val": "hello"})
        )
        assert result.status == ToolExecutionStatus.SUCCESS

    async def test_type_validation_integer(self) -> None:
        registry = ToolRegistry()
        registry.register(ToolDefinition(
            name="int_tool",
            parameters=(ToolParameter(name="val", type="integer"),),
            fn=lambda val: str(val),
        ))
        runtime = ToolRuntime(registry)

        result = await runtime.execute(
            ToolCall(name="int_tool", arguments={"val": "not_int"})
        )
        assert result.status == ToolExecutionStatus.ERROR

        result = await runtime.execute(
            ToolCall(name="int_tool", arguments={"val": 42})
        )
        assert result.status == ToolExecutionStatus.SUCCESS

    async def test_type_validation_number(self) -> None:
        registry = ToolRegistry()
        registry.register(ToolDefinition(
            name="num_tool",
            parameters=(ToolParameter(name="val", type="number"),),
            fn=lambda val: str(val),
        ))
        runtime = ToolRuntime(registry)

        result = await runtime.execute(
            ToolCall(name="num_tool", arguments={"val": "x"})
        )
        assert result.status == ToolExecutionStatus.ERROR

        result = await runtime.execute(
            ToolCall(name="num_tool", arguments={"val": 3.14})
        )
        assert result.status == ToolExecutionStatus.SUCCESS

        result = await runtime.execute(
            ToolCall(name="num_tool", arguments={"val": 42})
        )
        assert result.status == ToolExecutionStatus.SUCCESS

    async def test_type_validation_boolean(self) -> None:
        registry = ToolRegistry()
        registry.register(ToolDefinition(
            name="bool_tool",
            parameters=(ToolParameter(name="flag", type="boolean"),),
            fn=lambda flag: str(flag),
        ))
        runtime = ToolRuntime(registry)

        result = await runtime.execute(
            ToolCall(name="bool_tool", arguments={"flag": "yes"})
        )
        assert result.status == ToolExecutionStatus.ERROR

        result = await runtime.execute(
            ToolCall(name="bool_tool", arguments={"flag": True})
        )
        assert result.status == ToolExecutionStatus.SUCCESS

    async def test_type_validation_array(self) -> None:
        registry = ToolRegistry()
        registry.register(ToolDefinition(
            name="arr_tool",
            parameters=(ToolParameter(name="items", type="array"),),
            fn=lambda items: str(items),
        ))
        runtime = ToolRuntime(registry)

        result = await runtime.execute(
            ToolCall(name="arr_tool", arguments={"items": "not_list"})
        )
        assert result.status == ToolExecutionStatus.ERROR

        result = await runtime.execute(
            ToolCall(name="arr_tool", arguments={"items": [1, 2, 3]})
        )
        assert result.status == ToolExecutionStatus.SUCCESS

    async def test_type_validation_object(self) -> None:
        registry = ToolRegistry()
        registry.register(ToolDefinition(
            name="obj_tool",
            parameters=(ToolParameter(name="data", type="object"),),
            fn=lambda data: str(data),
        ))
        runtime = ToolRuntime(registry)

        result = await runtime.execute(
            ToolCall(name="obj_tool", arguments={"data": "not_dict"})
        )
        assert result.status == ToolExecutionStatus.ERROR

        result = await runtime.execute(
            ToolCall(name="obj_tool", arguments={"data": {"key": "val"}})
        )
        assert result.status == ToolExecutionStatus.SUCCESS


# ---------------------------------------------------------------------------
# Error message clarity
# ---------------------------------------------------------------------------


class TestToolRuntimeErrors:
    async def test_error_message_includes_tool_name(
        self, runtime: ToolRuntime
    ) -> None:
        result = await runtime.execute(
            ToolCall(name="sync_echo", arguments={"extra_param": "bad"})
        )
        assert result.status == ToolExecutionStatus.ERROR
        assert result.error is not None
        assert "sync_echo" in result.error

    async def test_various_finish_states(self) -> None:
        """Verify status enum covers expected states."""
        assert ToolExecutionStatus.SUCCESS.value == "success"
        assert ToolExecutionStatus.ERROR.value == "error"
        assert ToolExecutionStatus.TIMEOUT.value == "timeout"
        assert ToolExecutionStatus.CANCELLED.value == "cancelled"
