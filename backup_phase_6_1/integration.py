"""Provider ↔ Tool integration layer.

Bridges the provider layer and the tool runtime.

``ProviderResponse``
    ↓
``extract_tool_calls()``
    ↓
``ToolCall``
    ↓
``ToolRuntime.execute()``
    ↓
``ToolResult``
    ↓
``format_tool_result()``  (dispatches to provider-specific formatter)
    ↓
``ProviderMessage``

Providers remain responsible only for formatting tool results.
Tool execution remains entirely provider-independent.
"""

from __future__ import annotations

from typing import Any

from app.provider.models import ProviderMessage, ProviderResponse, Role, ToolCallRequest
from app.tools.models import ToolCall, ToolResult


# ---------------------------------------------------------------------------
# Extraction: ProviderResponse → ToolCall list
# ---------------------------------------------------------------------------


def extract_tool_calls(response: ProviderResponse) -> list[ToolCall]:
    """Extract ``ToolCall`` objects from a ``ProviderResponse``.

    Works with any provider — the response already contains normalized
    ``ToolCallRequest`` objects from the provider's response mapper.

    Args:
        response: A ``ProviderResponse`` from any provider.

    Returns:
        A list of ``ToolCall`` objects ready for the ``ToolRuntime``.
    """
    if not response.tool_calls:
        return []

    return [
        ToolCall(
            id=tc.id,
            name=tc.name,
            arguments=dict(tc.arguments),
        )
        for tc in response.tool_calls
    ]


# ---------------------------------------------------------------------------
# Formatting: ToolResult → ProviderMessage (dispatcher)
# ---------------------------------------------------------------------------


def format_tool_result(
    tool_call: ToolCall,
    result: ToolResult,
    *,
    provider_type: str = "openai",
) -> ProviderMessage:
    """Format a ``ToolResult`` into a ``ProviderMessage``.

    Dispatches to the correct provider-specific formatter based on
    *provider_type*.

    Args:
        tool_call: The original ``ToolCall`` that was executed.
        result: The ``ToolResult`` from execution.
        provider_type: ``"openai"`` (default) or ``"claude"``.

    Returns:
        A ``ProviderMessage`` that can be appended to the conversation.
    """
    if provider_type == "claude":
        from app.provider.claude.tool_mapper import format_tool_result as fmt

        return fmt(tool_call, result)

    # Default: OpenAI format
    from app.provider.openai.tool_mapper import format_tool_result as fmt

    return fmt(tool_call, result)


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def tool_call_to_request(tool_call: ToolCall) -> ToolCallRequest:
    """Convert a ``ToolCall`` back to a provider ``ToolCallRequest``.

    Args:
        tool_call: The ``ToolCall`` from the tool runtime.

    Returns:
        A ``ToolCallRequest`` for the provider layer.
    """
    return ToolCallRequest(
        id=tool_call.id,
        name=tool_call.name,
        arguments=dict(tool_call.arguments),
    )


def request_to_tool_call(request: ToolCallRequest) -> ToolCall:
    """Convert a provider ``ToolCallRequest`` to a ``ToolCall``.

    Args:
        request: A ``ToolCallRequest`` from the provider layer.

    Returns:
        A ``ToolCall`` for the tool runtime.
    """
    return ToolCall(
        id=request.id,
        name=request.name,
        arguments=dict(request.arguments),
    )


# ---------------------------------------------------------------------------
# Round-trip helper
# ---------------------------------------------------------------------------


async def execute_tool_calls(
    tool_calls: list[ToolCall],
    runtime: Any,
    *,
    provider_type: str = "openai",
) -> list[ProviderMessage]:
    """Execute a list of tool calls and return formatted ``ProviderMessage`` objects.

    One tool request.  One execution.  One tool response.
    No automatic looping.

    Args:
        tool_calls: The tool calls to execute.
        runtime: A ``ToolRuntime`` instance.
        provider_type: ``"openai"`` or ``"claude"`` for formatting.

    Returns:
        A list of ``ProviderMessage`` objects containing tool results,
        one per tool call.
    """
    messages: list[ProviderMessage] = []
    for tc in tool_calls:
        result = await runtime.execute(tc)
        msg = format_tool_result(tc, result, provider_type=provider_type)
        messages.append(msg)
    return messages
