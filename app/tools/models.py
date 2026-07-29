"""Tool domain models.

Every model in this module is immutable and independent of any specific
provider implementation.  They represent the canonical data types that
flow through the Universal Tool Calling Runtime.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Callable


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ToolExecutionStatus(str, enum.Enum):
    """The status of a tool execution."""

    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


# ---------------------------------------------------------------------------
# Tool metadata
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolMetadata:
    """Optional metadata attached to a tool definition."""

    author: str = ""
    version: str = "1.0.0"
    tags: tuple[str, ...] = ()
    category: str = ""
    notes: str = ""


# ---------------------------------------------------------------------------
# Tool parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolParameter:
    """A single parameter of a tool.

    Describes the expected type and constraints for one argument.
    """

    name: str = ""
    type: str = "string"  # JSON Schema type: string, number, integer, boolean, array, object
    description: str = ""
    required: bool = True
    default: Any = None
    enum_values: tuple[str, ...] = ()

    def to_json_schema(self) -> dict[str, Any]:
        """Convert to a JSON Schema property entry."""
        entry: dict[str, Any] = {
            "type": self.type,
            "description": self.description,
        }
        if self.default is not None:
            entry["default"] = self.default
        if self.enum_values:
            entry["enum"] = list(self.enum_values)
        return entry


# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolDefinition:
    """The canonical definition of a tool.

    This is the universal format that all providers translate to/from.
    Providers convert this to their native tool format (e.g. Anthropic
    tool schema, OpenAI function calling schema).
    """

    name: str = ""
    description: str = ""
    parameters: tuple[ToolParameter, ...] = ()
    required: tuple[str, ...] = ()
    metadata: ToolMetadata = field(default_factory=ToolMetadata)
    fn: Callable[..., Any] | None = None

    @property
    def parameter_names(self) -> list[str]:
        """Return the list of parameter names."""
        return [p.name for p in self.parameters]

    @property
    def required_parameters(self) -> list[str]:
        """Return the list of required parameter names."""
        if self.required:
            return list(self.required)
        return [p.name for p in self.parameters if p.required]

    def to_json_schema(self) -> dict[str, Any]:
        """Convert to an OpenAI-compatible JSON Schema for function calling.

        Returns a dict with ``name``, ``description``, and ``parameters``
        keys suitable for provider tool definitions.
        """
        properties: dict[str, Any] = {}
        for param in self.parameters:
            properties[param.name] = param.to_json_schema()

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(self.required_parameters),
            },
        }

    def to_openai_tool(self) -> dict[str, Any]:
        """Convert to an OpenAI tool definition format.

        Returns::

            {"type": "function", "function": {"name": ..., "description": ..., "parameters": ...}}
        """
        schema = self.to_json_schema()
        return {
            "type": "function",
            "function": {
                "name": schema["name"],
                "description": schema["description"],
                "parameters": schema["parameters"],
            },
        }

    def to_anthropic_tool(self) -> dict[str, Any]:
        """Convert to an Anthropic tool definition format.

        Returns::

            {"name": ..., "description": ..., "input_schema": {"type": "object", ...}}
        """
        schema = self.to_json_schema()
        return {
            "name": schema["name"],
            "description": schema["description"],
            "input_schema": schema["parameters"],
        }


# ---------------------------------------------------------------------------
# Tool call (inbound)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCall:
    """A request to execute a tool.

    Produced by provider response mappers and consumed by the
    ``ToolRuntime``.
    """

    id: str = ""
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Tool result (outbound)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolResult:
    """The result of executing a tool.

    Always returned — never raised.  Tools that raise are caught and
    returned as errors inside a ``ToolResult``.
    """

    output: str = ""
    error: str | None = None
    duration_ms: float = 0.0
    status: ToolExecutionStatus = ToolExecutionStatus.SUCCESS


# ---------------------------------------------------------------------------
# Execution record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolExecution:
    """A complete record of a single tool execution.

    Captures everything needed for logging, telemetry, and debugging.
    """

    tool_call: ToolCall = field(default_factory=ToolCall)
    result: ToolResult = field(default_factory=ToolResult)
    definition: ToolDefinition | None = None
    start_time: float = 0.0
    end_time: float = 0.0
