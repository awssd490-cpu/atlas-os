"""Tool Runtime — the universal tool execution engine.

The runtime accepts ``ToolCall`` objects, locates the corresponding
``ToolDefinition`` in the registry, validates arguments, executes
the tool function, and returns a ``ToolResult``.

Providers remain completely outside this layer.
"""

from __future__ import annotations

import inspect
import time
from typing import Any

from app.tools.errors import (
    ToolExecutionError,
    ToolInvalidArgumentsError,
    ToolNotFoundError,
    ToolValidationError,
)
from app.tools.models import (
    ToolCall,
    ToolDefinition,
    ToolExecution,
    ToolExecutionStatus,
    ToolParameter,
    ToolResult,
)
from app.tools.registry import ToolRegistry


class ToolRuntime:
    """Universal tool execution engine.

    Usage::

        runtime = ToolRuntime(registry)
        result = await runtime.execute(
            ToolCall(id="call_1", name="calculator", arguments={"expression": "2+2"})
        )
        print(result.output)  # "4"
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    @property
    def registry(self) -> ToolRegistry:
        return self._registry

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    async def execute(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call and return the result.

        This method never raises.  All errors are captured inside the
        returned ``ToolResult``.

        Args:
            tool_call: The tool call to execute.

        Returns:
            A ``ToolResult`` with the output or error information.
        """
        try:
            # 1. Locate the tool
            definition = self._registry.get(tool_call.name)
        except ToolNotFoundError as exc:
            return ToolResult(
                output="",
                error=str(exc),
                status=ToolExecutionStatus.ERROR,
            )

        try:
            # 2. Validate arguments
            validated = self._validate_arguments(definition, tool_call.arguments)

            # 3. Resolve the callable
            fn = definition.fn
            if fn is None:
                return ToolResult(
                    output="",
                    error=f"Tool {tool_call.name!r} has no callable function",
                    status=ToolExecutionStatus.ERROR,
                )

            # 4. Execute with timing
            start_time = time.monotonic()
            output = await self._execute_fn(fn, validated)
            duration_ms = (time.monotonic() - start_time) * 1000

            # 5. Format output
            output_str = str(output) if output is not None else ""

            return ToolResult(
                output=output_str,
                error=None,
                duration_ms=duration_ms,
                status=ToolExecutionStatus.SUCCESS,
            )

        except (ToolValidationError, ToolInvalidArgumentsError) as exc:
            return ToolResult(
                output="",
                error=str(exc),
                duration_ms=0.0,
                status=ToolExecutionStatus.ERROR,
            )

        except ToolExecutionError as exc:
            return ToolResult(
                output="",
                error=str(exc),
                duration_ms=0.0,
                status=ToolExecutionStatus.ERROR,
            )

        except Exception as exc:
            # Safety net: never crash Atlas
            error = ToolExecutionError(
                name=tool_call.name,
                original_exception=exc,
            )
            return ToolResult(
                output="",
                error=str(error),
                duration_ms=0.0,
                status=ToolExecutionStatus.ERROR,
            )

    # ------------------------------------------------------------------
    # Execution with full record
    # ------------------------------------------------------------------

    async def execute_with_record(self, tool_call: ToolCall) -> ToolExecution:
        """Execute a tool call and return a full ``ToolExecution`` record.

        Includes the definition, timing, and both the call and result.
        """
        start_time = time.time()
        result = await self.execute(tool_call)
        end_time = time.time()

        try:
            definition = self._registry.get(tool_call.name)
        except ToolNotFoundError:
            definition = None

        return ToolExecution(
            tool_call=tool_call,
            result=result,
            definition=definition,
            start_time=start_time,
            end_time=end_time,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_arguments(
        definition: ToolDefinition,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """Validate tool arguments against the tool definition.

        Checks:
        - No unknown parameters
        - All required parameters present
        - Basic type validation (string, integer, number, boolean)

        Args:
            definition: The tool definition to validate against.
            arguments: The raw arguments from the tool call.

        Returns:
            The validated arguments dict.

        Raises:
            ToolInvalidArgumentsError: If validation fails.
        """
        errors: list[str] = []
        param_map = {p.name: p for p in definition.parameters}

        # Check for unknown parameters
        for key in arguments:
            if key not in param_map:
                errors.append(f"Unknown parameter {key!r}")

        if errors:
            raise ToolInvalidArgumentsError(
                name=definition.name,
                message="; ".join(errors),
                details={
                    "unknown_params": [k for k in arguments if k not in param_map],
                },
            )

        # Check for missing required parameters
        for param_name in definition.required_parameters:
            if param_name not in arguments:
                errors.append(f"Missing required parameter {param_name!r}")

        if errors:
            raise ToolInvalidArgumentsError(
                name=definition.name,
                message="; ".join(errors),
                details={
                    "missing_params": [
                        p for p in definition.required_parameters
                        if p not in arguments
                    ],
                },
            )

        # Basic type validation
        for param_name, value in arguments.items():
            param = param_map.get(param_name)
            if param is not None:
                _validate_value_type(param, value, errors)

        if errors:
            raise ToolValidationError(
                name=definition.name,
                message="; ".join(errors),
                details={"type_errors": errors},
            )

        return dict(arguments)

    # ------------------------------------------------------------------
    # Execution helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _execute_fn(
        fn: Any,
        arguments: dict[str, Any],
    ) -> Any:
        """Execute a callable with arguments.

        Supports both async and sync functions.
        """
        try:
            if inspect.iscoroutinefunction(fn):
                return await fn(**arguments)
            return fn(**arguments)
        except Exception as exc:
            raise ToolExecutionError(
                name=getattr(fn, "__name__", "unknown"),
                original_exception=exc,
            ) from exc


# ---------------------------------------------------------------------------
# Module-level type validation helper
# ---------------------------------------------------------------------------


def _validate_value_type(
    param: ToolParameter,
    value: Any,
    errors: list[str],
) -> None:
    """Validate a single value against its parameter type.

    Performs basic JSON Schema type matching.
    """
    json_type = param.type

    if json_type == "string":
        if not isinstance(value, str):
            errors.append(
                f"Parameter {param.name!r} expected string, got {type(value).__name__}"
            )

    elif json_type == "integer":
        if not isinstance(value, int):
            errors.append(
                f"Parameter {param.name!r} expected integer, got {type(value).__name__}"
            )

    elif json_type == "number":
        if not isinstance(value, (int, float)):
            errors.append(
                f"Parameter {param.name!r} expected number, got {type(value).__name__}"
            )

    elif json_type == "boolean":
        if not isinstance(value, bool):
            errors.append(
                f"Parameter {param.name!r} expected boolean, got {type(value).__name__}"
            )

    elif json_type == "array":
        if not isinstance(value, (list, tuple)):
            errors.append(
                f"Parameter {param.name!r} expected array, got {type(value).__name__}"
            )

    elif json_type == "object":
        if not isinstance(value, dict):
            errors.append(
                f"Parameter {param.name!r} expected object, got {type(value).__name__}"
            )


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------


def create_runtime(
    registry: ToolRegistry | None = None,
) -> ToolRuntime:
    """Create a ``ToolRuntime`` with an optional or fresh registry.

    Args:
        registry: An existing ``ToolRegistry``, or ``None`` to create a new one.

    Returns:
        A new ``ToolRuntime``.
    """
    return ToolRuntime(registry or ToolRegistry())
