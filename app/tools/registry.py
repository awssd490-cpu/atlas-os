"""Tool Registry — central directory of all available tools.

Tools register themselves here by name.  The registry supports
discovery, lookup, and existence checks.

The registry is dependency injectable.
"""

from __future__ import annotations

from typing import Any

from app.tools.errors import DuplicateToolError, ToolNotFoundError
from app.tools.models import ToolDefinition


class ToolRegistry:
    """Central registry of all available tools.

    Thread-safe for concurrent reads.  Registration is typically done
    once at startup.
    """

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, tool: ToolDefinition | Any) -> ToolDefinition:
        """Register a tool.

        Accepts a ``ToolDefinition`` directly, or any object that has
        a ``to_definition()`` method (such as a ``@tool``-decorated
        function).

        Args:
            tool: A ``ToolDefinition`` or ``@tool``-decorated callable.

        Returns:
            The registered ``ToolDefinition``.

        Raises:
            DuplicateToolError: If a tool with the same name is already registered.
            ValueError: If *tool* is neither a ``ToolDefinition`` nor has ``to_definition()``.
        """
        definition = self._resolve_definition(tool)

        if definition.name in self._tools:
            raise DuplicateToolError(
                name=definition.name,
                details={"existing_name": definition.name},
            )

        self._tools[definition.name] = definition
        return definition

    def unregister(self, name: str) -> None:
        """Remove a tool from the registry.

        Args:
            name: The tool name to remove.

        Raises:
            ToolNotFoundError: If *name* is not registered.
        """
        if name not in self._tools:
            raise ToolNotFoundError(name=name)
        del self._tools[name]

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> ToolDefinition:
        """Look up a tool by name.

        Args:
            name: The tool name.

        Returns:
            The ``ToolDefinition``.

        Raises:
            ToolNotFoundError: If *name* is not registered.
        """
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFoundError(
                name=name,
                details={"available": list(self._tools.keys())},
            )
        return tool

    def exists(self, name: str) -> bool:
        """Check if a tool is registered.

        Args:
            name: The tool name.

        Returns:
            ``True`` if registered, ``False`` otherwise.
        """
        return name in self._tools

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def list_tools(self) -> list[str]:
        """Return all registered tool names."""
        return list(self._tools.keys())

    def list_definitions(self) -> list[ToolDefinition]:
        """Return all registered tool definitions."""
        return list(self._tools.values())

    def count(self) -> int:
        """Return the number of registered tools."""
        return len(self._tools)

    def clear(self) -> None:
        """Remove all registered tools."""
        self._tools.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_definition(tool: Any) -> ToolDefinition:
        """Resolve a ``ToolDefinition`` from *tool*.

        Accepts:
        - ``ToolDefinition`` instances (returned as-is).
        - Objects with a ``to_definition()`` method (e.g. ``@tool``-decorated functions).
        """
        if isinstance(tool, ToolDefinition):
            if tool.fn is None:
                raise ValueError(
                    f"ToolDefinition {tool.name!r} has no callable (fn is None) — "
                    "use a @tool-decorated function or set fn"
                )
            return tool

        if hasattr(tool, "to_definition") and callable(tool.to_definition):
            definition = tool.to_definition()
            if isinstance(definition, ToolDefinition):
                if definition.fn is None:
                    raise ValueError(
                        f"Tool definition from {type(tool).__name__} has no callable"
                    )
                return definition
            raise ValueError(
                f"{type(tool).__name__}.to_definition() did not return a ToolDefinition"
            )

        raise ValueError(
            f"Cannot register {type(tool).__name__}: must be a ToolDefinition "
            "or a @tool-decorated callable with a to_definition() method"
        )
