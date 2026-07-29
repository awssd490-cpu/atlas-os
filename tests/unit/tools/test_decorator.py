"""Tests for the @tool decorator."""

from __future__ import annotations

from app.tools.decorators import tool
from app.tools.models import ToolDefinition, ToolMetadata


class TestToolDecorator:
    def test_decorate_sync_function(self) -> None:
        @tool(name="greet", description="Greet someone")
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        definition = greet.to_definition()
        assert definition.name == "greet"
        assert definition.description == "Greet someone"
        assert definition.fn is not None
        assert len(definition.parameters) == 1
        assert definition.parameters[0].name == "name"
        assert definition.parameters[0].type == "string"

    def test_decorate_async_function(self) -> None:
        @tool(name="async_greet")
        async def async_greet(name: str) -> str:
            return f"Hello, {name}!"

        definition = async_greet.to_definition()
        assert definition.name == "async_greet"
        assert definition.fn is not None

    def test_default_name_from_function(self) -> None:
        @tool()
        def my_custom_function(x: int) -> int:
            return x * 2

        assert my_custom_function.to_definition().name == "my_custom_function"

    def test_default_description_from_docstring(self) -> None:
        @tool()
        def documented_tool(query: str) -> str:
            """Search for things."""
            return f"searching {query}"

        assert documented_tool.to_definition().description == "Search for things."

    def test_no_docstring_empty_description(self) -> None:
        @tool()
        def no_docs(a: str) -> str:
            return a

        assert no_docs.to_definition().description == ""

    def test_parameter_types_inferred(self) -> None:
        @tool()
        def typed_func(
            text: str,
            count: int,
            price: float,
            active: bool,
        ) -> str:
            return f"{text} {count} {price} {active}"

        params = typed_func.to_definition().parameters
        param_map = {p.name: p for p in params}

        assert param_map["text"].type == "string"
        assert param_map["count"].type == "integer"
        assert param_map["price"].type == "number"
        assert param_map["active"].type == "boolean"

    def test_optional_parameter(self) -> None:
        @tool()
        def with_default(name: str, greeting: str = "Hello") -> str:
            return f"{greeting}, {name}!"

        params = with_default.to_definition().parameters
        param_map = {p.name: p for p in params}

        assert param_map["name"].required is True
        assert param_map["greeting"].required is False
        assert param_map["greeting"].default == "Hello"

    def test_required_parameters_list(self) -> None:
        @tool()
        def required_test(a: str, b: int = 0) -> str:
            return f"{a} {b}"

        definition = required_test.to_definition()
        assert "a" in definition.required_parameters
        assert "b" not in definition.required_parameters

    def test_execute_sync_decorated(self) -> None:
        @tool(name="double")
        def double(x: int) -> int:
            return x * 2

        # Can call the wrapped function directly
        import inspect
        assert inspect.iscoroutinefunction(double) is False

        # Execute via definition.fn
        result = double.to_definition().fn(x=5)  # type: ignore[misc]
        assert result == 10

    async def test_execute_async_decorated(self) -> None:
        @tool(name="async_double")
        async def async_double(x: int) -> int:
            return x * 2

        result = await async_double.to_definition().fn(x=5)  # type: ignore[misc]
        assert result == 10

    def test_metadata(self) -> None:
        meta = ToolMetadata(
            author="test-user",
            version="2.0.0",
            tags=("utility",),
            category="math",
        )

        @tool(name="with_meta", metadata=meta)
        def with_meta(x: int) -> int:
            return x

        definition = with_meta.to_definition()
        assert definition.metadata.author == "test-user"
        assert definition.metadata.version == "2.0.0"
        assert definition.metadata.tags == ("utility",)

    def test_wrapper_callable(self) -> None:
        @tool(name="callable_test")
        def callable_test(x: int) -> int:
            return x * 3

        # The wrapper itself is not a function with positional args
        # but it has a to_definition() method
        assert hasattr(callable_test, "to_definition")
        assert hasattr(callable_test, "definition")
        assert callable_test.__name__ == "callable_test"

    def test_skip_self_and_cls(self) -> None:
        """The decorator should skip self/cls/args/kwargs parameters."""

        @tool(name="method_style")
        def method_style(self: str, x: int) -> str:  # type: ignore[valid-type]
            return f"{self} {x}"

        params = method_style.to_definition().parameters
        param_names = [p.name for p in params]
        assert "self" not in param_names
        assert "x" in param_names
