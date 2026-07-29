"""Tests for ToolRegistry."""

from __future__ import annotations

import pytest

from app.tools.errors import DuplicateToolError, ToolNotFoundError
from app.tools.models import ToolDefinition, ToolParameter
from app.tools.registry import ToolRegistry


def _make_dummy_tool(name: str = "test") -> ToolDefinition:
    """Create a minimal ToolDefinition for testing."""
    return ToolDefinition(
        name=name,
        description=f"A {name} tool",
        parameters=(ToolParameter(name="input", type="string"),),
        fn=lambda input="": f"executed {name}: {input}",
    )


class TestToolRegistry:
    def setup_method(self) -> None:
        self.registry = ToolRegistry()

    def test_register(self) -> None:
        tool = _make_dummy_tool("calculator")
        result = self.registry.register(tool)
        assert result is tool
        assert self.registry.count() == 1
        assert self.registry.exists("calculator")

    def test_register_and_get(self) -> None:
        tool = _make_dummy_tool("weather")
        self.registry.register(tool)
        retrieved = self.registry.get("weather")
        assert retrieved is tool
        assert retrieved.name == "weather"

    def test_register_duplicate_raises(self) -> None:
        self.registry.register(_make_dummy_tool("calc"))
        with pytest.raises(DuplicateToolError, match="calc"):
            self.registry.register(_make_dummy_tool("calc"))

    def test_register_duplicate_includes_name(self) -> None:
        self.registry.register(_make_dummy_tool("dupe"))
        with pytest.raises(DuplicateToolError) as exc_info:
            self.registry.register(_make_dummy_tool("dupe"))
        assert "dupe" in str(exc_info.value)

    def test_unregister(self) -> None:
        self.registry.register(_make_dummy_tool("temp"))
        self.registry.unregister("temp")
        assert self.registry.exists("temp") is False
        assert self.registry.count() == 0

    def test_unregister_nonexistent_raises(self) -> None:
        with pytest.raises(ToolNotFoundError, match="nonexistent"):
            self.registry.unregister("nonexistent")

    def test_get_nonexistent_raises(self) -> None:
        with pytest.raises(ToolNotFoundError, match="ghost"):
            self.registry.get("ghost")

    def test_get_nonexistent_has_available_list(self) -> None:
        self.registry.register(_make_dummy_tool("a"))
        with pytest.raises(ToolNotFoundError) as exc_info:
            self.registry.get("b")
        assert "available" in exc_info.value.details

    def test_exists(self) -> None:
        self.registry.register(_make_dummy_tool("exists"))
        assert self.registry.exists("exists") is True
        assert self.registry.exists("missing") is False

    def test_list_tools(self) -> None:
        self.registry.register(_make_dummy_tool("alpha"))
        self.registry.register(_make_dummy_tool("beta"))
        names = self.registry.list_tools()
        assert "alpha" in names
        assert "beta" in names
        assert len(names) == 2

    def test_list_definitions(self) -> None:
        a = _make_dummy_tool("a")
        b = _make_dummy_tool("b")
        self.registry.register(a)
        self.registry.register(b)
        defs = self.registry.list_definitions()
        assert a in defs
        assert b in defs
        assert len(defs) == 2

    def test_count(self) -> None:
        assert self.registry.count() == 0
        self.registry.register(_make_dummy_tool("one"))
        assert self.registry.count() == 1
        self.registry.register(_make_dummy_tool("two"))
        assert self.registry.count() == 2

    def test_clear(self) -> None:
        self.registry.register(_make_dummy_tool("a"))
        self.registry.register(_make_dummy_tool("b"))
        assert self.registry.count() == 2
        self.registry.clear()
        assert self.registry.count() == 0

    def test_empty_registry(self) -> None:
        assert self.registry.count() == 0
        assert self.registry.list_tools() == []
        assert self.registry.list_definitions() == []
        assert self.registry.exists("anything") is False


class TestToolRegistryWithCallable:
    def test_register_with_fn_attribute(self) -> None:
        """Register a ToolDefinition that has a callable."""

        def my_fn(x: str = "") -> str:
            return f"result: {x}"

        tool = ToolDefinition(name="echo", fn=my_fn)
        registry = ToolRegistry()
        registry.register(tool)
        assert registry.exists("echo") is True

    def test_register_definition_without_fn_raises(self) -> None:
        """A ToolDefinition with no fn raises ValueError."""
        tool = ToolDefinition(name="empty", description="No callable")
        registry = ToolRegistry()
        with pytest.raises(ValueError, match="no callable"):
            registry.register(tool)
