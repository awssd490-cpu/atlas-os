"""Tests for tool domain models."""

from __future__ import annotations

from app.tools.models import (
    ToolCall,
    ToolDefinition,
    ToolExecution,
    ToolExecutionStatus,
    ToolMetadata,
    ToolParameter,
    ToolResult,
)


class TestToolParameter:
    def test_create(self) -> None:
        param = ToolParameter(
            name="expression",
            type="string",
            description="Math expression",
            required=True,
        )
        assert param.name == "expression"
        assert param.type == "string"
        assert param.required is True

    def test_create_optional(self) -> None:
        param = ToolParameter(
            name="units",
            type="string",
            description="Unit system",
            required=False,
            default="metric",
        )
        assert param.required is False
        assert param.default == "metric"

    def test_create_with_enum(self) -> None:
        param = ToolParameter(
            name="unit",
            type="string",
            enum_values=("celsius", "fahrenheit"),
        )
        assert param.enum_values == ("celsius", "fahrenheit")

    def test_to_json_schema(self) -> None:
        param = ToolParameter(
            name="expression",
            type="string",
            description="Math expression",
            required=True,
        )
        schema = param.to_json_schema()
        assert schema == {"type": "string", "description": "Math expression"}

    def test_to_json_schema_with_default(self) -> None:
        param = ToolParameter(
            name="units",
            type="string",
            description="Units",
            default="metric",
        )
        schema = param.to_json_schema()
        assert schema["default"] == "metric"

    def test_to_json_schema_with_enum(self) -> None:
        param = ToolParameter(
            name="unit",
            type="string",
            enum_values=("celsius", "fahrenheit"),
        )
        schema = param.to_json_schema()
        assert schema["enum"] == ["celsius", "fahrenheit"]


class TestToolDefinition:
    def test_create(self) -> None:
        def dummy() -> None:
            pass

        definition = ToolDefinition(
            name="calculator",
            description="Evaluate math expressions",
            parameters=(
                ToolParameter(name="expression", type="string"),
            ),
            fn=dummy,
        )
        assert definition.name == "calculator"
        assert definition.description == "Evaluate math expressions"
        assert len(definition.parameters) == 1
        assert definition.fn is dummy

    def test_parameter_names(self) -> None:
        definition = ToolDefinition(
            name="weather",
            parameters=(
                ToolParameter(name="city", type="string"),
                ToolParameter(name="units", type="string", required=False, default="celsius"),
            ),
        )
        assert definition.parameter_names == ["city", "units"]

    def test_required_parameters_from_flag(self) -> None:
        definition = ToolDefinition(
            name="test",
            parameters=(
                ToolParameter(name="a", type="string", required=True),
                ToolParameter(name="b", type="string", required=False),
            ),
            required=("a",),
        )
        assert definition.required_parameters == ["a"]

    def test_required_parameters_default(self) -> None:
        """When 'required' is empty, use required flag from parameters."""
        definition = ToolDefinition(
            name="test",
            parameters=(
                ToolParameter(name="a", type="string", required=True),
                ToolParameter(name="b", type="string", required=False),
            ),
        )
        assert definition.required_parameters == ["a"]

    def test_to_json_schema(self) -> None:
        definition = ToolDefinition(
            name="calculator",
            description="Evaluate math expressions",
            parameters=(
                ToolParameter(name="expression", type="string", description="The expression"),
            ),
        )
        schema = definition.to_json_schema()
        assert schema["name"] == "calculator"
        assert schema["description"] == "Evaluate math expressions"
        assert schema["parameters"]["type"] == "object"
        assert "expression" in schema["parameters"]["properties"]
        assert schema["parameters"]["required"] == ["expression"]

    def test_to_openai_tool(self) -> None:
        definition = ToolDefinition(
            name="calculator",
            description="Evaluate math expressions",
            parameters=(
                ToolParameter(name="expression", type="string"),
            ),
        )
        tool = definition.to_openai_tool()
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "calculator"
        assert tool["function"]["parameters"]["type"] == "object"

    def test_to_anthropic_tool(self) -> None:
        definition = ToolDefinition(
            name="calculator",
            description="Evaluate math expressions",
            parameters=(
                ToolParameter(name="expression", type="string"),
            ),
        )
        tool = definition.to_anthropic_tool()
        assert tool["name"] == "calculator"
        assert tool["description"] == "Evaluate math expressions"
        assert tool["input_schema"]["type"] == "object"

    def test_no_fn(self) -> None:
        """ToolDefinition can exist without a callable."""
        definition = ToolDefinition(
            name="abstract",
            description="No callable attached",
        )
        assert definition.fn is None
        assert definition.name == "abstract"


class TestToolCall:
    def test_create(self) -> None:
        call = ToolCall(
            id="call_1",
            name="calculator",
            arguments={"expression": "2+2"},
        )
        assert call.id == "call_1"
        assert call.name == "calculator"
        assert call.arguments == {"expression": "2+2"}

    def test_defaults(self) -> None:
        call = ToolCall()
        assert call.id == ""
        assert call.name == ""
        assert call.arguments == {}


class TestToolResult:
    def test_success(self) -> None:
        result = ToolResult(
            output="4",
            duration_ms=5.0,
            status=ToolExecutionStatus.SUCCESS,
        )
        assert result.output == "4"
        assert result.error is None
        assert result.duration_ms == 5.0
        assert result.status == ToolExecutionStatus.SUCCESS

    def test_error(self) -> None:
        result = ToolResult(
            output="",
            error="Something went wrong",
            status=ToolExecutionStatus.ERROR,
        )
        assert result.error == "Something went wrong"
        assert result.status == ToolExecutionStatus.ERROR

    def test_defaults(self) -> None:
        result = ToolResult()
        assert result.output == ""
        assert result.error is None
        assert result.duration_ms == 0.0
        assert result.status == ToolExecutionStatus.SUCCESS


class TestToolExecution:
    def test_create(self) -> None:
        execution = ToolExecution(
            tool_call=ToolCall(id="c1", name="calc", arguments={"x": "1"}),
            result=ToolResult(output="1"),
        )
        assert execution.tool_call.name == "calc"
        assert execution.result.output == "1"

    def test_defaults(self) -> None:
        execution = ToolExecution()
        assert execution.tool_call.name == ""
        assert execution.result.status == ToolExecutionStatus.SUCCESS
        assert execution.definition is None


class TestToolMetadata:
    def test_create(self) -> None:
        meta = ToolMetadata(
            author="test",
            version="2.0.0",
            tags=("math", "utility"),
            category="calculator",
        )
        assert meta.author == "test"
        assert meta.version == "2.0.0"
        assert meta.tags == ("math", "utility")

    def test_defaults(self) -> None:
        meta = ToolMetadata()
        assert meta.author == ""
        assert meta.version == "1.0.0"
        assert meta.tags == ()
