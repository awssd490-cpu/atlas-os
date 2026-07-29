"""Tests for Provider ↔ Tool integration layer."""

from __future__ import annotations

import pytest

from app.provider.models import (
    ContentType,
    ProviderMessage,
    ProviderResponse,
    ProviderUsage,
    Role,
    StopReason,
    ToolCallRequest,
)
from app.tools.integration import (
    execute_tool_calls,
    extract_tool_calls,
    format_tool_result,
    request_to_tool_call,
    tool_call_to_request,
)
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
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def calculator_definition() -> ToolCall:
    return ToolCall(
        id="call_calc",
        name="calculator",
        arguments={"expression": "2+2"},
    )


@pytest.fixture
def weather_definition() -> ToolCall:
    return ToolCall(
        id="call_weather",
        name="get_weather",
        arguments={"city": "London"},
    )


# ---------------------------------------------------------------------------
# Extraction tests
# ---------------------------------------------------------------------------


class TestExtractToolCalls:
    def test_extract_from_claude_response(self) -> None:
        """Extract ToolCalls from a Claude-style ProviderResponse."""
        response = ProviderResponse(
            content="Let me check that.",
            message=ProviderMessage(
                role=Role.ASSISTANT,
                content="Let me check that.",
                tool_calls=[
                    ToolCallRequest(
                        id="tu_1",
                        name="get_weather",
                        arguments={"city": "San Francisco"},
                    ),
                ],
            ),
            stop_reason=StopReason.TOOL_CALL,
            tool_calls=[
                ToolCallRequest(
                    id="tu_1",
                    name="get_weather",
                    arguments={"city": "San Francisco"},
                ),
            ],
        )
        calls = extract_tool_calls(response)
        assert len(calls) == 1
        assert calls[0].id == "tu_1"
        assert calls[0].name == "get_weather"
        assert calls[0].arguments == {"city": "San Francisco"}

    def test_extract_from_openai_response(self) -> None:
        """Extract ToolCalls from an OpenAI-style ProviderResponse."""
        response = ProviderResponse(
            content="Let me look that up.",
            message=ProviderMessage(
                role=Role.ASSISTANT,
                content="Let me look that up.",
                tool_calls=[
                    ToolCallRequest(
                        id="call_abc",
                        name="get_weather",
                        arguments={"city": "Paris"},
                    ),
                ],
            ),
            stop_reason=StopReason.TOOL_CALL,
            tool_calls=[
                ToolCallRequest(
                    id="call_abc",
                    name="get_weather",
                    arguments={"city": "Paris"},
                ),
            ],
        )
        calls = extract_tool_calls(response)
        assert len(calls) == 1
        assert calls[0].id == "call_abc"
        assert calls[0].arguments == {"city": "Paris"}

    def test_extract_multiple_tool_calls(self) -> None:
        """Multiple tool calls in one response."""
        response = ProviderResponse(
            content="",
            message=ProviderMessage(
                role=Role.ASSISTANT,
                content="",
                tool_calls=[
                    ToolCallRequest(id="c1", name="tool_a", arguments={"x": "1"}),
                    ToolCallRequest(id="c2", name="tool_b", arguments={"y": "2"}),
                ],
            ),
            stop_reason=StopReason.TOOL_CALL,
            tool_calls=[
                ToolCallRequest(id="c1", name="tool_a", arguments={"x": "1"}),
                ToolCallRequest(id="c2", name="tool_b", arguments={"y": "2"}),
            ],
        )
        calls = extract_tool_calls(response)
        assert len(calls) == 2
        assert calls[0].name == "tool_a"
        assert calls[1].name == "tool_b"

    def test_extract_empty_tool_calls(self) -> None:
        """No tool calls in response returns empty list."""
        response = ProviderResponse(
            content="Hello",
            stop_reason=StopReason.STOP,
        )
        calls = extract_tool_calls(response)
        assert calls == []

    def test_extract_preserves_arguments(self) -> None:
        """Tool call arguments are preserved as a new dict."""
        response = ProviderResponse(
            content="",
            tool_calls=[
                ToolCallRequest(
                    id="c1",
                    name="search",
                    arguments={"query": "test", "limit": 10},
                ),
            ],
        )
        calls = extract_tool_calls(response)
        assert calls[0].arguments["query"] == "test"
        assert calls[0].arguments["limit"] == 10


# ---------------------------------------------------------------------------
# ToolCall ↔ ToolCallRequest conversion tests
# ---------------------------------------------------------------------------


class TestConversion:
    def test_tool_call_to_request(self) -> None:
        result = tool_call_to_request(
            ToolCall(id="c1", name="test", arguments={"x": 1})
        )
        assert result.id == "c1"
        assert result.name == "test"
        assert result.arguments == {"x": 1}

    def test_request_to_tool_call(self) -> None:
        result = request_to_tool_call(
            ToolCallRequest(id="c2", name="echo", arguments={"msg": "hi"})
        )
        assert result.id == "c2"
        assert result.name == "echo"
        assert result.arguments == {"msg": "hi"}

    def test_round_trip(self) -> None:
        original = ToolCallRequest(id="rt", name="round", arguments={"a": 1, "b": 2})
        call = request_to_tool_call(original)
        back = tool_call_to_request(call)
        assert back.id == original.id
        assert back.name == original.name
        assert back.arguments == original.arguments


# ---------------------------------------------------------------------------
# Formatting tests — OpenAI
# ---------------------------------------------------------------------------


class TestFormatOpenAIToolResult:
    def test_format_success(self) -> None:
        """Successful tool result formatted as OpenAI tool message."""
        result = ToolResult(
            output="4",
            status=ToolExecutionStatus.SUCCESS,
        )
        msg = format_tool_result(
            ToolCall(id="call_1", name="calculator", arguments={"expr": "2+2"}),
            result,
            provider_type="openai",
        )
        assert msg.role == Role.TOOL
        assert msg.content == "4"
        assert msg.tool_call_id == "call_1"

    def test_format_error(self) -> None:
        """Failed tool result formatted with error content."""
        result = ToolResult(
            output="",
            error="Division by zero",
            status=ToolExecutionStatus.ERROR,
        )
        msg = format_tool_result(
            ToolCall(id="call_2", name="divide", arguments={}),
            result,
            provider_type="openai",
        )
        assert msg.role == Role.TOOL
        assert msg.content == "Division by zero"
        assert msg.tool_call_id == "call_2"

    def test_format_empty_output(self) -> None:
        """Tool with no output produces empty content."""
        result = ToolResult(output="", status=ToolExecutionStatus.SUCCESS)
        msg = format_tool_result(
            ToolCall(id="call_3", name="void", arguments={}),
            result,
            provider_type="openai",
        )
        assert msg.content == ""

    def test_openai_default_provider(self) -> None:
        """Default provider_type is 'openai'."""
        result = ToolResult(output="hello", status=ToolExecutionStatus.SUCCESS)
        msg = format_tool_result(
            ToolCall(id="c", name="echo", arguments={}),
            result,
        )
        assert msg.role == Role.TOOL

    def test_multiple_tool_results(self) -> None:
        """Multiple tool results each produce a tool message."""
        results = [
            format_tool_result(
                ToolCall(id="c1", name="a"),
                ToolResult(output="1"),
                provider_type="openai",
            ),
            format_tool_result(
                ToolCall(id="c2", name="b"),
                ToolResult(output="2"),
                provider_type="openai",
            ),
        ]
        assert len(results) == 2
        assert results[0].tool_call_id == "c1"
        assert results[1].tool_call_id == "c2"


# ---------------------------------------------------------------------------
# Formatting tests — Claude
# ---------------------------------------------------------------------------


class TestFormatClaudeToolResult:
    def test_format_success(self) -> None:
        """Successful tool result as Claude tool_result."""
        result = ToolResult(
            output="72°F",
            status=ToolExecutionStatus.SUCCESS,
        )
        msg = format_tool_result(
            ToolCall(id="tu_1", name="get_weather", arguments={"city": "NYC"}),
            result,
            provider_type="claude",
        )
        assert msg.role == Role.USER
        assert msg.content == "72°F"
        assert msg.content_type == ContentType.TOOL_RESULT
        assert msg.tool_call_id == "tu_1"
        assert msg.metadata["type"] == "tool_result"
        assert msg.metadata["tool_use_id"] == "tu_1"
        assert msg.metadata["is_error"] is False

    def test_format_error(self) -> None:
        """Failed tool result as Claude tool_result with error flag."""
        result = ToolResult(
            output="",
            error="City not found",
            status=ToolExecutionStatus.ERROR,
        )
        msg = format_tool_result(
            ToolCall(id="tu_2", name="get_weather"),
            result,
            provider_type="claude",
        )
        assert msg.role == Role.USER
        assert msg.content == "City not found"
        assert msg.metadata["is_error"] is True

    def test_format_fallback_error_message(self) -> None:
        """Error with no message gets a generic fallback."""
        result = ToolResult(
            output="",
            error=None,
            status=ToolExecutionStatus.ERROR,
        )
        msg = format_tool_result(
            ToolCall(id="tu_3", name="failing_tool"),
            result,
            provider_type="claude",
        )
        assert "failed" in msg.content


# ---------------------------------------------------------------------------
# Round-trip execution tests
# ---------------------------------------------------------------------------


class TestExecuteToolCalls:
    @pytest.fixture
    def runtime(self) -> ToolRuntime:
        """Create a runtime with a calculator tool."""
        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="calculator",
                description="Evaluate math",
                parameters=(
                    ToolParameter(name="expression", type="string"),
                ),
                fn=lambda expression: str(eval(expression)),
            )
        )
        return ToolRuntime(registry)

    async def test_execute_single_tool_call(self, runtime: ToolRuntime) -> None:
        """Single tool call executed and formatted for OpenAI."""
        calls = [ToolCall(id="c1", name="calculator", arguments={"expression": "1+1"})]
        messages = await execute_tool_calls(calls, runtime, provider_type="openai")
        assert len(messages) == 1
        assert messages[0].role == Role.TOOL
        assert messages[0].content == "2"

    async def test_execute_multiple_tool_calls(self, runtime: ToolRuntime) -> None:
        """Multiple tool calls each produce a result message."""
        calls = [
            ToolCall(id="c1", name="calculator", arguments={"expression": "1+1"}),
            ToolCall(id="c2", name="calculator", arguments={"expression": "2*3"}),
        ]
        messages = await execute_tool_calls(calls, runtime, provider_type="openai")
        assert len(messages) == 2
        assert messages[0].content == "2"
        assert messages[1].content == "6"

    async def test_execute_empty_list(self, runtime: ToolRuntime) -> None:
        """Empty list of tool calls returns empty list."""
        messages = await execute_tool_calls([], runtime)
        assert messages == []

    async def test_execute_claude_format(self, runtime: ToolRuntime) -> None:
        """Tool results formatted as Claude tool_result messages."""
        calls = [ToolCall(id="tu_1", name="calculator", arguments={"expression": "4+4"})]
        messages = await execute_tool_calls(calls, runtime, provider_type="claude")
        assert len(messages) == 1
        assert messages[0].role == Role.USER
        assert messages[0].content == "8"
        assert messages[0].metadata["type"] == "tool_result"

    async def test_execute_not_found_returns_error(
        self, runtime: ToolRuntime
    ) -> None:
        """Unknown tool execution returns error message."""
        calls = [ToolCall(id="c1", name="nonexistent", arguments={})]
        messages = await execute_tool_calls(calls, runtime, provider_type="openai")
        assert len(messages) == 1
        assert messages[0].role == Role.TOOL
        assert "not found" in messages[0].content.lower()


# ---------------------------------------------------------------------------
# Provider-specific tool mapper tests
# ---------------------------------------------------------------------------


class TestClaudeToolMapper:
    def test_extract_tool_calls_from_raw_response(self) -> None:
        """Extract ToolCall from raw Claude API response."""
        from app.provider.claude.tool_mapper import extract_tool_calls_from_response

        data = {
            "id": "msg_123",
            "content": [
                {"type": "text", "text": "Let me check..."},
                {
                    "type": "tool_use",
                    "id": "tu_1",
                    "name": "get_weather",
                    "input": {"city": "Tokyo"},
                },
            ],
        }
        calls = extract_tool_calls_from_response(data)
        assert len(calls) == 1
        assert calls[0].id == "tu_1"
        assert calls[0].name == "get_weather"
        assert calls[0].arguments == {"city": "Tokyo"}

    def test_extract_multiple_from_raw_response(self) -> None:
        """Multiple tool_use blocks in a Claude response."""
        from app.provider.claude.tool_mapper import extract_tool_calls_from_response

        data = {
            "content": [
                {"type": "tool_use", "id": "tu_1", "name": "a", "input": {"x": 1}},
                {"type": "tool_use", "id": "tu_2", "name": "b", "input": {"y": 2}},
            ],
        }
        calls = extract_tool_calls_from_response(data)
        assert len(calls) == 2

    def test_extract_no_tool_use(self) -> None:
        """No tool_use blocks returns empty list."""
        from app.provider.claude.tool_mapper import extract_tool_calls_from_response

        data = {"content": [{"type": "text", "text": "Hello"}]}
        calls = extract_tool_calls_from_response(data)
        assert calls == []

    def test_extract_empty_content(self) -> None:
        """Empty content returns empty list."""
        from app.provider.claude.tool_mapper import extract_tool_calls_from_response

        calls = extract_tool_calls_from_response({"content": []})
        assert calls == []

    def test_tool_call_to_provider(self) -> None:
        """Convert ToolCall → ToolCallRequest."""
        from app.provider.claude.tool_mapper import tool_call_to_provider

        request = tool_call_to_provider(
            ToolCall(id="tu_1", name="test", arguments={"a": 1})
        )
        assert request.id == "tu_1"
        assert request.name == "test"
        assert request.arguments == {"a": 1}

    def test_format_tool_result(self) -> None:
        """Format ToolResult as Claude user message."""
        from app.provider.claude.tool_mapper import format_tool_result as fmt

        msg = fmt(
            ToolCall(id="tu_1", name="test"),
            ToolResult(output="done"),
        )
        assert msg.role == Role.USER
        assert msg.content == "done"
        assert msg.metadata["is_error"] is False


class TestOpenAIToolMapper:
    def test_extract_tool_calls_from_raw_response(self) -> None:
        """Extract ToolCall from raw OpenAI API response."""
        from app.provider.openai.tool_mapper import extract_tool_calls_from_response

        data = {
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call_abc",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"city": "London"}',
                            },
                        },
                    ],
                },
            }],
        }
        calls = extract_tool_calls_from_response(data)
        assert len(calls) == 1
        assert calls[0].id == "call_abc"
        assert calls[0].name == "get_weather"
        assert calls[0].arguments == {"city": "London"}

    def test_extract_multiple_from_raw_response(self) -> None:
        """Multiple tool_calls in an OpenAI response."""
        from app.provider.openai.tool_mapper import extract_tool_calls_from_response

        data = {
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {"name": "a", "arguments": '{"x": 1}'},
                        },
                        {
                            "id": "c2",
                            "function": {"name": "b", "arguments": '{"y": 2}'},
                        },
                    ],
                },
            }],
        }
        calls = extract_tool_calls_from_response(data)
        assert len(calls) == 2

    def test_extract_no_choices(self) -> None:
        """No choices returns empty list."""
        from app.provider.openai.tool_mapper import extract_tool_calls_from_response

        calls = extract_tool_calls_from_response({"choices": []})
        assert calls == []

    def test_extract_no_tool_calls(self) -> None:
        """No tool_calls in message returns empty list."""
        from app.provider.openai.tool_mapper import extract_tool_calls_from_response

        data = {
            "choices": [{
                "message": {"role": "assistant", "content": "Hello"},
            }],
        }
        calls = extract_tool_calls_from_response(data)
        assert calls == []

    def test_extract_invalid_json_arguments(self) -> None:
        """Invalid JSON arguments string handled gracefully."""
        from app.provider.openai.tool_mapper import extract_tool_calls_from_response

        data = {
            "choices": [{
                "message": {
                    "tool_calls": [
                        {
                            "id": "c1",
                            "function": {
                                "name": "broken",
                                "arguments": "not-json",
                            },
                        },
                    ],
                },
            }],
        }
        calls = extract_tool_calls_from_response(data)
        assert len(calls) == 1
        assert calls[0].arguments == {"raw": "not-json"}

    def test_tool_call_to_provider(self) -> None:
        """Convert ToolCall → ToolCallRequest."""
        from app.provider.openai.tool_mapper import tool_call_to_provider

        request = tool_call_to_provider(
            ToolCall(id="c1", name="test", arguments={"b": True})
        )
        assert request.id == "c1"
        assert request.name == "test"
        assert request.arguments == {"b": True}

    def test_format_tool_result_success(self) -> None:
        """Format successful ToolResult as OpenAI tool message."""
        from app.provider.openai.tool_mapper import format_tool_result as fmt

        msg = fmt(
            ToolCall(id="c1", name="test"),
            ToolResult(output="42"),
        )
        assert msg.role == Role.TOOL
        assert msg.content == "42"
        assert msg.tool_call_id == "c1"

    def test_format_tool_result_error(self) -> None:
        """Format failed ToolResult as OpenAI tool message."""
        from app.provider.openai.tool_mapper import format_tool_result as fmt

        msg = fmt(
            ToolCall(id="c2", name="test"),
            ToolResult(error="fail", status=ToolExecutionStatus.ERROR),
        )
        assert msg.role == Role.TOOL
        assert msg.content == "fail"
