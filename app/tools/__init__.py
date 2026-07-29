"""Universal Tool Calling Runtime.

The runtime executes tool calls in a completely provider-independent manner.
Providers may produce tool calls differently — Atlas exposes ONE unified
execution model.

Architecture::

    Application
         │
         ▼
    ToolRuntime
         │
         ▼
    ToolRegistry
         │
         ▼
    Registered Tool  (Python function decorated with @tool)
         │
         ▼
    ToolResult

Usage::

    @tool(name="calculator", description="Evaluate a math expression")
    async def calculator(expression: str) -> str:
        ...

    registry = ToolRegistry()
    registry.register(calculator)

    runtime = ToolRuntime(registry)
    result = await runtime.execute(ToolCall(name="calculator", arguments={"expression": "2+2"}))
    print(result.output)  # "4"
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.tools.models import (
        ToolDefinition,
        ToolParameter,
        ToolCall,
        ToolResult,
        ToolExecution,
        ToolMetadata,
    )
    from app.tools.registry import ToolRegistry
    from app.tools.runtime import ToolRuntime
    from app.tools.errors import (
        ToolError,
        ToolNotFoundError,
        DuplicateToolError,
        ToolValidationError,
        ToolExecutionError,
        ToolInvalidArgumentsError,
    )

__all__ = [
    "ToolDefinition",
    "ToolParameter",
    "ToolCall",
    "ToolResult",
    "ToolExecution",
    "ToolMetadata",
    "ToolRegistry",
    "ToolRuntime",
    "ToolError",
    "ToolNotFoundError",
    "DuplicateToolError",
    "ToolValidationError",
    "ToolExecutionError",
    "ToolInvalidArgumentsError",
]
