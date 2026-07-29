"""Claude-specific tool mapping.

Formats ``ToolResult`` objects into Claude ``tool_result`` content blocks
and extracts ``ToolCall`` objects from Claude responses.

Formatting belongs inside providers.
Execution belongs inside ToolRuntime.
"""

from __future__ import annotations

from typing import Any

from app.provider.models import ContentType, ProviderMessage, Role, ToolCallRequest
from app.tools.models import ToolCall, ToolResult


def format_tool_result(
    tool_call: ToolCall,
    result: ToolResult,
) -> ProviderMessage:
    """Format a ``ToolResult`` as a Claude ``tool_result`` content block.

    Claude uses a ``user``-role message with a ``tool_result`` content block::

        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "call_1", "content": "output"}
        ]}

    Args:
        tool_call: The original ``ToolCall`` that was executed.
        result: The ``ToolResult`` from execution.

    Returns:
        A ``ProviderMessage`` with role ``USER``.
    """
    is_error = result.status != "success" or result.error is not None
    content = result.output if not is_error else (result.error or "Tool execution failed")

    return ProviderMessage(
        role=Role.USER,
        content=content,
        content_type=ContentType.TOOL_RESULT,
        tool_call_id=tool_call.id,
        metadata={
            "type": "tool_result",
            "tool_use_id": tool_call.id,
            "is_error": is_error,
        },
    )


def extract_tool_calls_from_response(data: dict[str, Any]) -> list[ToolCall]:
    """Extract ``ToolCall`` objects from a raw Claude API response.

    Args:
        data: The raw parsed JSON from a Claude response.

    Returns:
        A list of ``ToolCall`` objects.
    """
    calls: list[ToolCall] = []
    for block in data.get("content", []):
        if block.get("type") == "tool_use":
            calls.append(ToolCall(
                id=block.get("id", ""),
                name=block.get("name", ""),
                arguments=dict(block.get("input", {})),
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
