"""Tests for AgentRuntime."""

from __future__ import annotations

from typing import Any

import pytest

from app.agent.config import AgentConfig
from app.agent.errors import (
    IterationLimitExceeded,
    ProviderExecutionError,
    ToolCallLimitExceeded,
)
from app.agent.runtime import AgentRuntime
from app.provider.models import (
    ProviderMessage,
    ProviderRequest,
    ProviderResponse,
    ProviderUsage,
    Role,
    StopReason,
    ToolCallRequest,
)
from app.provider.provider import Provider
from app.tools.models import (
    ToolDefinition,
    ToolExecutionStatus,
    ToolParameter,
)
from app.tools.registry import ToolRegistry
from app.tools.runtime import ToolRuntime


# ---------------------------------------------------------------------------
# Mock provider
# ---------------------------------------------------------------------------


class _MockProvider(Provider):
    """A mock provider that returns canned responses.

    Each call to ``generate`` returns the next response from *responses*.
    This allows deterministic testing of the agent loop.
    """

    def __init__(
        self,
        responses: list[ProviderResponse],
        *,
        name: str = "mock",
    ) -> None:
        self._responses = list(responses)
        self._call_count = 0
        self._received_requests: list[ProviderRequest] = []
        self._should_fail = False
        self._fail_message = ""
        self._name = name

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def received_requests(self) -> list[ProviderRequest]:
        return list(self._received_requests)

    def fail_on_next(self, message: str = "API Error") -> None:
        self._should_fail = True
        self._fail_message = message

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def generate(self, request: ProviderRequest) -> ProviderResponse:
        self._call_count += 1
        self._received_requests.append(request)

        if self._should_fail:
            raise RuntimeError(self._fail_message)

        if not self._responses:
            return ProviderResponse(
                content="No more responses",
                stop_reason=StopReason.STOP,
            )

        return self._responses.pop(0)

    def stream(self, request: ProviderRequest):  # type: ignore[override]
        raise NotImplementedError("Streaming not supported in mock")

    async def count_tokens(self, request: ProviderRequest) -> int:
        return sum(len(m.content) for m in request.messages)

    @property
    def provider_info(self) -> Any:
        from app.provider.models import (
            ProviderCapability,
            ProviderInfo,
            ProviderMetadata,
        )
        return ProviderInfo(
            metadata=ProviderMetadata(name=self._name),
            capabilities=[ProviderCapability(name="tool_calling")],
        )


# ---------------------------------------------------------------------------
# Mock tool: calculator
# ---------------------------------------------------------------------------


async def _mock_calculator(expression: str) -> str:
    """Simple expression evaluator for testing."""
    result = eval(expression)
    return str(result)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provider_response(
    content: str,
    tool_calls: list[ToolCallRequest] | None = None,
    stop_reason: StopReason = StopReason.STOP,
) -> ProviderResponse:
    """Create a provider response with optional tool calls."""
    calls = tool_calls or []
    return ProviderResponse(
        content=content,
        message=ProviderMessage(
            role=Role.ASSISTANT,
            content=content,
            tool_calls=calls,
        ),
        stop_reason=stop_reason if not calls else StopReason.TOOL_CALL,
        tool_calls=calls,
        usage=ProviderUsage(prompt_tokens=10, completion_tokens=5),
    )


def _make_tool_call(
    name: str,
    arguments: dict[str, Any] | None = None,
    tool_call_id: str = "call_1",
) -> ToolCallRequest:
    return ToolCallRequest(
        id=tool_call_id,
        name=name,
        arguments=arguments or {},
    )


def _make_calculator_request(expression: str) -> ToolCallRequest:
    return _make_tool_call("calculator", {"expression": expression}, "calc_1")


def _create_default_tool_runtime() -> ToolRuntime:
    """Create a ToolRuntime with a calculator tool."""
    registry = ToolRegistry()
    registry.register(ToolDefinition(
        name="calculator",
        description="Evaluate math expressions",
        parameters=(
            ToolParameter(name="expression", type="string", description="Expression"),
        ),
        fn=_mock_calculator,
    ))
    return ToolRuntime(registry)


def _create_agent(
    responses: list[ProviderResponse],
    tool_runtime: ToolRuntime | None = None,
    config: AgentConfig | None = None,
    provider_name: str = "mock",
) -> AgentRuntime:
    """Create an AgentRuntime with a mock provider."""
    provider = _MockProvider(responses, name=provider_name)
    tr = tool_runtime or _create_default_tool_runtime()
    return AgentRuntime(provider, tr, config=config)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class TestAgentRuntimeConfig:
    async def test_default_config(self) -> None:
        """AgentRuntime accepts default config."""
        provider = _MockProvider([
            _make_provider_response("final answer"),
        ])
        tr = _create_default_tool_runtime()
        runtime = AgentRuntime(provider, tr)
        assert runtime.config.max_iterations == 10
        assert runtime.config.max_tool_calls == 100

    async def test_custom_config(self) -> None:
        """AgentRuntime accepts custom config."""
        config = AgentConfig(max_iterations=3, max_tool_calls=5)
        runtime = AgentRuntime(
            _MockProvider([_make_provider_response("ok")]),
            _create_default_tool_runtime(),
            config=config,
        )
        assert runtime.config.max_iterations == 3
        assert runtime.config.max_tool_calls == 5

    async def test_per_run_config_override(self) -> None:
        """Per-run config overrides instance config."""
        runtime = _create_agent(
            [_make_provider_response("done")],
            config=AgentConfig(max_iterations=10),
        )
        # Run with a different config
        response = await runtime.run(
            [ProviderMessage(role=Role.USER, content="hi")],
            config=AgentConfig(max_iterations=1),
        )
        assert response.content == "done"
        assert runtime.iteration_count == 1


# ---------------------------------------------------------------------------
# Successful runs
# ---------------------------------------------------------------------------


class TestAgentRuntimeSuccess:
    async def test_provider_returns_final_answer_immediately(self) -> None:
        """No tool calls — agent returns the provider response."""
        runtime = _create_agent([
            _make_provider_response("Hello, world!"),
        ])
        response = await runtime.run(
            [ProviderMessage(role=Role.USER, content="Say hello")],
        )
        assert response.content == "Hello, world!"
        assert response.stop_reason == StopReason.STOP
        assert runtime.iteration_count == 1
        assert runtime.provider_requests == 1
        assert runtime.tool_calls_executed == 0

    async def test_single_tool_call(self) -> None:
        """One tool call followed by final response."""
        runtime = _create_agent([
            _make_provider_response(
                "Let me calculate.",
                tool_calls=[_make_calculator_request("2+2")],
            ),
            _make_provider_response("The answer is 4"),
        ])
        response = await runtime.run(
            [ProviderMessage(role=Role.USER, content="Calculate 2+2")],
        )
        assert response.content == "The answer is 4"
        assert runtime.iteration_count == 2
        assert runtime.provider_requests == 2
        assert runtime.tool_calls_executed == 1

    async def test_multiple_sequential_tool_calls(self) -> None:
        """Multiple tool calls across iterations."""
        runtime = _create_agent([
            _make_provider_response(
                "First calculation.",
                tool_calls=[_make_calculator_request("1+1")],
            ),
            _make_provider_response(
                "Second calculation.",
                tool_calls=[_make_calculator_request("2+2")],
            ),
            _make_provider_response("Final answer."),
        ])
        response = await runtime.run(
            [ProviderMessage(role=Role.USER, content="Calculate")],
        )
        assert response.content == "Final answer."
        assert runtime.iteration_count == 3
        assert runtime.tool_calls_executed == 2

    async def test_multiple_tool_calls_in_one_iteration(self) -> None:
        """Multiple tool calls from a single provider response."""
        runtime = _create_agent([
            _make_provider_response(
                "Calculating...",
                tool_calls=[
                    _make_calculator_request("1+1"),
                    _make_calculator_request("2+2"),
                ],
            ),
            _make_provider_response("Results computed."),
        ])
        response = await runtime.run(
            [ProviderMessage(role=Role.USER, content="Compute")],
        )
        assert runtime.iteration_count == 2
        assert runtime.tool_calls_executed == 2

    async def test_conversation_grows(self) -> None:
        """Conversation history grows with each iteration."""
        responses = [
            _make_provider_response(
                "Thinking...",
                tool_calls=[_make_calculator_request("1+1")],
            ),
            _make_provider_response("Got 2, continuing.",
                tool_calls=[_make_calculator_request("2+2")],
            ),
            _make_provider_response("Done."),
        ]
        provider = _MockProvider(responses)
        runtime = AgentRuntime(provider, _create_default_tool_runtime())
        await runtime.run(
            [ProviderMessage(role=Role.USER, content="Start")],
        )
        # The provider was called 3 times
        assert provider.call_count == 3

    async def test_system_prompt_preserved(self) -> None:
        """System prompt is passed through to the provider."""
        runtime = _create_agent([
            _make_provider_response("Ok"),
        ])
        await runtime.run(
            [ProviderMessage(role=Role.USER, content="Hi")],
            system="Be helpful.",
        )
        assert runtime.provider_requests == 1

    async def test_tool_call_with_openai_format(self) -> None:
        """Tool results formatted as OpenAI tool messages by default."""
        runtime = _create_agent([
            _make_provider_response(
                "Using calculator",
                tool_calls=[_make_calculator_request("3*3")],
            ),
            _make_provider_response("Result: 9"),
        ])
        response = await runtime.run(
            [ProviderMessage(role=Role.USER, content="Calculate 3*3")],
        )
        assert response.content == "Result: 9"

    async def test_tool_call_with_claude_format(self) -> None:
        """Tool results formatted as Claude tool_result blocks."""
        # Use a provider whose metadata.name contains "claude"
        runtime = _create_agent(
            [
                _make_provider_response(
                    "Using calculator",
                    tool_calls=[_make_calculator_request("4+4")],
                ),
                _make_provider_response("Result: 8"),
            ],
            provider_name="claude",
        )
        response = await runtime.run(
            [ProviderMessage(role=Role.USER, content="Calculate 4+4")],
        )
        assert response.content == "Result: 8"

    async def test_provider_request_kwargs_passed(self) -> None:
        """Additional kwargs passed to ProviderRequest."""
        runtime = _create_agent([
            _make_provider_response("done"),
        ])
        await runtime.run(
            [ProviderMessage(role=Role.USER, content="Hi")],
            max_tokens=500,
            temperature=0.3,
        )
        assert runtime.provider_requests == 1

    async def test_empty_tool_calls_list(self) -> None:
        """Provider response with empty tool_calls list."""
        runtime = _create_agent([
            _make_provider_response("Final", tool_calls=[]),
        ])
        response = await runtime.run(
            [ProviderMessage(role=Role.USER, content="Hello")],
        )
        assert response.content == "Final"
        assert runtime.iteration_count == 1


# ---------------------------------------------------------------------------
# Loop behavior
# ---------------------------------------------------------------------------


class TestAgentRuntimeLoopBehavior:
    async def test_stops_on_final_response(self) -> None:
        """Loop stops when provider returns non-tool-call response."""
        runtime = _create_agent([
            _make_provider_response("a", tool_calls=[_make_calculator_request("1+1")]),
            _make_provider_response("b", tool_calls=[_make_calculator_request("2+2")]),
            _make_provider_response("c"),  # final
        ])
        response = await runtime.run(
            [ProviderMessage(role=Role.USER, content="Go")],
        )
        assert response.content == "c"
        assert runtime.iteration_count == 3

    async def test_iteration_counting(self) -> None:
        """Iteration count matches actual iterations."""
        runtime = _create_agent([
            _make_provider_response("1", tool_calls=[_make_calculator_request("1+1")]),
            _make_provider_response("2", tool_calls=[_make_calculator_request("2+2")]),
            _make_provider_response("3"),
        ])
        await runtime.run(
            [ProviderMessage(role=Role.USER, content="Count")],
        )
        assert runtime.iteration_count == 3

    async def test_tool_calls_exceeded_counting(self) -> None:
        """Tool call count tracks cumulative tool calls."""
        runtime = _create_agent([
            _make_provider_response(
                "Calc",
                tool_calls=[
                    _make_calculator_request("1"),
                    _make_calculator_request("2"),
                ],
            ),
            _make_provider_response("Done"),
        ])
        await runtime.run(
            [ProviderMessage(role=Role.USER, content="Start")],
        )
        assert runtime.tool_calls_executed == 2

    async def test_elapsed_time_recorded(self) -> None:
        """Elapsed time is recorded after run."""
        runtime = _create_agent([
            _make_provider_response("done"),
        ])
        await runtime.run(
            [ProviderMessage(role=Role.USER, content="Hi")],
        )
        assert runtime.elapsed_time >= 0

    async def test_state_reset_between_runs(self) -> None:
        """State resets between consecutive runs."""
        runtime = _create_agent([
            _make_provider_response(
                "Calc",
                tool_calls=[_make_calculator_request("1+1")],
            ),
            _make_provider_response("Final"),
        ])
        await runtime.run(
            [ProviderMessage(role=Role.USER, content="Run 1")],
        )
        assert runtime.tool_calls_executed == 1
        assert runtime.iteration_count == 2

        # Second run with new responses
        provider = _MockProvider([_make_provider_response("Done2")])
        runtime2 = AgentRuntime(provider, _create_default_tool_runtime())
        await runtime2.run(
            [ProviderMessage(role=Role.USER, content="Run 2")],
        )
        assert runtime2.iteration_count == 1
        assert runtime2.tool_calls_executed == 0


# ---------------------------------------------------------------------------
# Limits
# ---------------------------------------------------------------------------


class TestAgentRuntimeLimits:
    async def test_iteration_limit_raises(self) -> None:
        """Iteration limit exceeded raises IterationLimitExceeded."""
        runtime = _create_agent(
            # Fill with tool-call responses
            [_make_provider_response(
                "more",
                tool_calls=[_make_calculator_request(str(i))],
            ) for i in range(20)],  # more than max_iterations
            config=AgentConfig(max_iterations=3),
        )
        with pytest.raises(IterationLimitExceeded) as exc_info:
            await runtime.run(
                [ProviderMessage(role=Role.USER, content="Loop")],
            )
        assert exc_info.value.details["max_iterations"] == 3

    async def test_iteration_limit_counts_correctly(self) -> None:
        """Iteration limit error includes the count."""
        runtime = _create_agent(
            [_make_provider_response(
                "more",
                tool_calls=[_make_calculator_request("1")],
            ) for _ in range(10)],
            config=AgentConfig(max_iterations=5),
        )
        with pytest.raises(IterationLimitExceeded) as exc_info:
            await runtime.run(
                [ProviderMessage(role=Role.USER, content="Loop")],
            )
        assert exc_info.value.details["iteration_count"] == 5

    async def test_iteration_limit_with_partial_response(self) -> None:
        """When raise_on_iteration_limit is False, return partial response."""
        runtime = _create_agent(
            [_make_provider_response(
                "more",
                tool_calls=[_make_calculator_request("1")],
            ) for _ in range(10)],
            config=AgentConfig(
                max_iterations=3,
                raise_on_iteration_limit=False,
                return_partial_response=True,
            ),
        )
        response = await runtime.run(
            [ProviderMessage(role=Role.USER, content="Loop")],
        )
        assert response.metadata.get("partial") is True
        assert response.metadata.get("iteration_count") == 3

    async def test_partial_response_preserves_real_output(self) -> None:
        """Partial response preserves the actual provider content and tool_calls."""
        runtime = _create_agent(
            [_make_provider_response(
                "actual content from provider",
                tool_calls=[_make_calculator_request("1")],
            ) for _ in range(10)],
            config=AgentConfig(
                max_iterations=3,
                raise_on_iteration_limit=False,
                return_partial_response=True,
            ),
        )
        response = await runtime.run(
            [ProviderMessage(role=Role.USER, content="Loop")],
        )
        # The real provider output should be preserved, not an empty synthetic response
        assert response.content == "actual content from provider"
        assert response.metadata.get("partial") is True
        assert response.metadata.get("iteration_count") == 3

    async def test_iteration_limit_without_partial_still_raises(self) -> None:
        """When return_partial_response is False, still raises."""
        runtime = _create_agent(
            [_make_provider_response(
                "more",
                tool_calls=[_make_calculator_request("1")],
            ) for _ in range(10)],
            config=AgentConfig(
                max_iterations=3,
                raise_on_iteration_limit=False,
                return_partial_response=False,
            ),
        )
        with pytest.raises(IterationLimitExceeded):
            await runtime.run(
                [ProviderMessage(role=Role.USER, content="Loop")],
            )

    async def test_tool_call_limit_raises(self) -> None:
        """Tool call limit exceeded raises ToolCallLimitExceeded."""
        runtime = _create_agent(
            [_make_provider_response(
                "calc",
                tool_calls=[_make_calculator_request("1")],
            ) for _ in range(20)],
            config=AgentConfig(max_tool_calls=3),
        )
        with pytest.raises(ToolCallLimitExceeded) as exc_info:
            await runtime.run(
                [ProviderMessage(role=Role.USER, content="Calc many")],
            )
        assert exc_info.value.details["max_tool_calls"] == 3

    async def test_tool_call_limit_counts_correctly(self) -> None:
        """Tool call limit error includes cumulative count."""
        runtime = _create_agent(
            [_make_provider_response(
                "calc",
                tool_calls=[_make_calculator_request("1")],
            ) for _ in range(20)],
            config=AgentConfig(max_tool_calls=3),
        )
        with pytest.raises(ToolCallLimitExceeded) as exc_info:
            await runtime.run(
                [ProviderMessage(role=Role.USER, content="Calc")],
            )
        assert exc_info.value.details["tool_calls_executed"] >= 2

    async def test_tool_call_limit_considers_batch_size(self) -> None:
        """Tool call limit checks against current + pending batch."""
        runtime = _create_agent(
            [
                _make_provider_response(
                    "batch",
                    tool_calls=[
                        _make_calculator_request("1"),
                        _make_calculator_request("2"),
                        _make_calculator_request("3"),
                    ],
                ),
                _make_provider_response("done"),
            ],
            config=AgentConfig(max_tool_calls=2),
        )
        with pytest.raises(ToolCallLimitExceeded):
            await runtime.run(
                [ProviderMessage(role=Role.USER, content="Batch")],
            )


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class TestAgentRuntimeErrors:
    async def test_provider_exception(self) -> None:
        """Provider exception wrapped in ProviderExecutionError."""
        provider = _MockProvider([])
        provider.fail_on_next("Connection refused")
        runtime = AgentRuntime(provider, _create_default_tool_runtime())

        with pytest.raises(ProviderExecutionError) as exc_info:
            await runtime.run(
                [ProviderMessage(role=Role.USER, content="Hi")],
            )
        assert "Connection refused" in str(exc_info.value)

    async def test_tool_execution_error_does_not_crash(self) -> None:
        """Tool execution error is captured in the conversation."""
        registry = ToolRegistry()
        registry.register(ToolDefinition(
            name="failing_tool",
            description="Always fails",
            parameters=(ToolParameter(name="msg", type="string", required=False),),
            fn=lambda msg="": (_ for _ in ()).throw(ValueError("tool failed")),
        ))
        tr = ToolRuntime(registry)

        runtime = _create_agent(
            [
                _make_provider_response(
                    "Trying tool",
                    tool_calls=[_make_tool_call("failing_tool", {"msg": "boom"})],
                ),
                _make_provider_response("Recovered from error."),
            ],
            tool_runtime=tr,
        )
        # Should not crash — tool error is handled gracefully
        response = await runtime.run(
            [ProviderMessage(role=Role.USER, content="Run failing tool")],
        )
        assert response.content == "Recovered from error."

    async def test_unknown_tool_does_not_crash(self) -> None:
        """Unknown tool name returns error result, does not crash."""
        runtime = _create_agent(
            [
                _make_provider_response(
                    "Unknown tool",
                    tool_calls=[_make_tool_call("nonexistent_tool", {})],
                ),
                _make_provider_response("Continued."),
            ],
        )
        response = await runtime.run(
            [ProviderMessage(role=Role.USER, content="Use unknown tool")],
        )
        assert response.content == "Continued."

    async def test_malformed_tool_calls(self) -> None:
        """Tool calls with missing parameters produce error results."""
        runtime = _create_agent(
            [
                _make_provider_response(
                    "Malformed",
                    tool_calls=[_make_tool_call("calculator", {"wrong_param": "x"})],
                ),
                _make_provider_response("Fixed."),
            ],
        )
        response = await runtime.run(
            [ProviderMessage(role=Role.USER, content="Malformed tool call")],
        )
        assert response.content == "Fixed."


# ---------------------------------------------------------------------------
# State properties
# ---------------------------------------------------------------------------


class TestAgentRuntimeState:
    async def test_properties_accessible(self) -> None:
        """All runtime properties are accessible."""
        runtime = _create_agent([
            _make_provider_response("done"),
        ])
        assert isinstance(runtime.provider, Provider)
        assert isinstance(runtime.tool_runtime, ToolRuntime)
        assert isinstance(runtime.config, AgentConfig)
        assert runtime.iteration_count == 0  # before any run
        assert runtime.tool_calls_executed == 0
        assert runtime.provider_requests == 0
        assert runtime.elapsed_time == 0.0

    async def test_properties_after_run(self) -> None:
        """Properties reflect state after a run."""
        runtime = _create_agent([
            _make_provider_response(
                "tool call",
                tool_calls=[_make_calculator_request("1+1")],
            ),
            _make_provider_response("done"),
        ])
        await runtime.run(
            [ProviderMessage(role=Role.USER, content="Start")],
        )
        assert runtime.iteration_count == 2
        assert runtime.tool_calls_executed == 1
        assert runtime.provider_requests == 2
        assert runtime.elapsed_time >= 0

    async def test_provider_request_counting(self) -> None:
        """Provider request counter reflects actual calls."""
        runtime = _create_agent([
            _make_provider_response("1", tool_calls=[_make_calculator_request("1")]),
            _make_provider_response("2", tool_calls=[_make_calculator_request("2")]),
            _make_provider_response("3", tool_calls=[_make_calculator_request("3")]),
            _make_provider_response("4"),
        ])
        await runtime.run(
            [ProviderMessage(role=Role.USER, content="Count")],
        )
        assert runtime.provider_requests == 4


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestAgentRuntimeEdgeCases:
    async def test_empty_messages(self) -> None:
        """Empty initial messages list."""
        runtime = _create_agent([
            _make_provider_response("Hello"),
        ])
        response = await runtime.run([])
        assert response.content == "Hello"

    async def test_single_message(self) -> None:
        """Single user message."""
        runtime = _create_agent([
            _make_provider_response("Hi back"),
        ])
        response = await runtime.run(
            [ProviderMessage(role=Role.USER, content="Hi")],
        )
        assert response.content == "Hi back"

    async def test_multiple_initial_messages(self) -> None:
        """Multiple initial messages preserved."""
        runtime = _create_agent([
            _make_provider_response("Final"),
        ])
        messages = [
            ProviderMessage(role=Role.USER, content="First"),
            ProviderMessage(role=Role.ASSISTANT, content="Second"),
            ProviderMessage(role=Role.USER, content="Third"),
        ]
        response = await runtime.run(messages)
        assert response.content == "Final"

    async def test_stop_reason_tool_call_mapped(self) -> None:
        """Response with tool_call stop reason is handled."""
        runtime = _create_agent([
            ProviderResponse(
                content="Using tool",
                message=ProviderMessage(
                    role=Role.ASSISTANT,
                    content="Using tool",
                    tool_calls=[_make_calculator_request("5+5")],
                ),
                stop_reason=StopReason.TOOL_CALL,
                tool_calls=[_make_calculator_request("5+5")],
            ),
            _make_provider_response("Result: 10"),
        ])
        response = await runtime.run(
            [ProviderMessage(role=Role.USER, content="Add")],
        )
        assert response.content == "Result: 10"
        assert runtime.tool_calls_executed == 1

    async def test_zero_iterations_not_allowed(self) -> None:
        """max_iterations must be >= 1."""
        with pytest.raises(ValueError):
            AgentConfig(max_iterations=0)

    async def test_zero_tool_calls_not_allowed(self) -> None:
        """max_tool_calls must be >= 1."""
        with pytest.raises(ValueError):
            AgentConfig(max_tool_calls=0)


# ---------------------------------------------------------------------------
# Provider type derivation
# ---------------------------------------------------------------------------


class TestAgentRuntimeProviderType:
    async def test_provider_type_openai_by_default(self) -> None:
        """Default mock provider name 'mock' maps to 'openai'."""
        runtime = _create_agent([_make_provider_response("ok")])
        # The _provider_type property derives from provider_info.metadata.name
        assert runtime._provider_type == "openai"

    async def test_provider_type_claude_by_name(self) -> None:
        """Provider name containing 'claude' maps to 'claude'."""
        runtime = _create_agent(
            [_make_provider_response("ok")],
            provider_name="claude-3-sonnet",
        )
        assert runtime._provider_type == "claude"

    async def test_provider_type_with_claude_provider(self) -> None:
        """Claude format tool results when using claude-named provider."""
        runtime = _create_agent(
            [
                _make_provider_response(
                    "Calc",
                    tool_calls=[_make_calculator_request("1+1")],
                ),
                _make_provider_response("Done"),
            ],
            provider_name="claude",
        )
        await runtime.run(
            [ProviderMessage(role=Role.USER, content="Start")],
        )
        assert runtime.tool_calls_executed == 1


# ---------------------------------------------------------------------------
# request_kwargs immutability
# ---------------------------------------------------------------------------


class TestAgentRuntimeKwargsImmutability:
    async def test_kwargs_not_mutated(self) -> None:
        """Caller-owned request_kwargs dict is never mutated."""
        runtime = _create_agent([_make_provider_response("ok")])
        original = {"max_tokens": 100, "temperature": 0.5, "tools": ["tool1"]}
        kwargs_copy = dict(original)

        await runtime.run(
            [ProviderMessage(role=Role.USER, content="Hi")],
            **kwargs_copy,
        )

        # Original dict should be unchanged (runtime should not mutate it)
        assert kwargs_copy == original

    async def test_kwargs_immutable_when_no_tools(self) -> None:
        """request_kwargs without tools key is also not mutated."""
        runtime = _create_agent([_make_provider_response("ok")])
        original = {"max_tokens": 200}
        kwargs_copy = dict(original)

        await runtime.run(
            [ProviderMessage(role=Role.USER, content="Hi")],
            **kwargs_copy,
        )

        assert kwargs_copy == original


# ---------------------------------------------------------------------------
# Claude tool result format (derived from provider name)
# ---------------------------------------------------------------------------


class TestAgentRuntimeClaudeFormat:
    async def test_claude_format_by_provider_name(self) -> None:
        """Claude format is selected based on provider metadata name."""
        runtime = _create_agent(
            [
                _make_provider_response(
                    "Thinking",
                    tool_calls=[_make_calculator_request("2+2")],
                ),
                _make_provider_response("Answer: 4"),
            ],
            provider_name="claude",
        )
        response = await runtime.run(
            [ProviderMessage(role=Role.USER, content="Calculate 2+2")],
        )
        assert response.content == "Answer: 4"
        assert runtime.iteration_count == 2
