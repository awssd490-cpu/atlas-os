"""OpenAI-compatible tool mapping.

Formats ``ToolResult`` objects into OpenAI ``tool``-role messages
and extracts ``ToolCall`` objects from OpenAI responses.

Formatting belongs inside providers.
Execution belongs inside ToolRuntime.
"""

from __future__ import annotations

from typing import Any

from app.provider.models import ProviderMessage, Role, ToolCallRequest
from app.tools.models import ToolCall, ToolResult


def format_tool_result(
    tool_call: ToolCall,
    result: ToolResult,
) -> ProviderMessage:
    """Format a ``ToolResult`` as an OpenAI ``tool``-role message.

    OpenAI format::

        {"role": "tool", "tool_call_id": "call_abc", "content": "output"}

    Args:
        tool_call: The original ``ToolCall`` that was executed.
        result: The ``ToolResult`` from execution.

    Returns:
        A ``ProviderMessage`` with role ``TOOL``.
    """
    content = result.output if result.status == "success" and result.error is None else (result.error or "")
    return ProviderMessage(
        role=Role.TOOL,
        content=content,
        tool_call_id=tool_call.id,
    )


def extract_tool_calls_from_response(data: dict[str, Any]) -> list[ToolCall]:
    """Extract ``ToolCall`` objects from a raw OpenAI Chat Completions response.

    Args:
        data: The raw parsed JSON from an OpenAI-compatible API response.

    Returns:
        A list of ``ToolCall`` objects.
    """
    calls: list[ToolCall] = []
    choices = data.get("choices", [])
    if not choices:
        return calls

    message = choices[0].get("message", {})
    raw_calls = message.get("tool_calls", [])
    if not raw_calls:
        return calls

    for tc in raw_calls:
        function = tc.get("function", {})
        arguments = function.get("arguments", "{}")
        if isinstance(arguments, str):
            import json

            try:
                parsed_args: dict[str, Any] = json.loads(arguments)
            except json.JSONDecodeError:
                parsed_args = {"raw": arguments}
        else:
            parsed_args = dict(arguments)

        calls.append(ToolCall(
            id=tc.get("id", ""),
            name=function.get("name", ""),
            arguments=parsed_args,
        ))
    return calls


def tool_call_to_provider(
    tool_call: ToolCall,
) -> ToolCallRequest:
    """Convert a ``ToolCall`` to a provider ``ToolCallRequest``.

    Useful when bridging between the tool runtime and provider layers.

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
