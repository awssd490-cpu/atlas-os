"""Tool decorators.

Provides the ``@tool`` decorator that automatically creates
``ToolDefinition`` metadata from a function's signature.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, get_type_hints

from app.tools.models import ToolDefinition, ToolMetadata, ToolParameter


class _ToolWrapper:
    """Wraps a function decorated with ``@tool``.

    The wrapper stores the ``ToolDefinition`` and provides a
    ``to_definition()`` method so it can be registered directly.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        definition: ToolDefinition,
    ) -> None:
        self._fn = fn
        self._definition = definition

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    def to_definition(self) -> ToolDefinition:
        return self._definition

    @property
    def __name__(self) -> str:
        return self._definition.name

    @property
    def __doc__(self) -> str | None:
        return self._fn.__doc__

    async def __call__(self, **kwargs: Any) -> Any:
        """Execute the wrapped function.

        Supports both async and sync functions transparently.
        """
        if inspect.iscoroutinefunction(self._fn):
            return await self._fn(**kwargs)
        return self._fn(**kwargs)


def tool(
    *,
    name: str | None = None,
    description: str | None = None,
    metadata: ToolMetadata | None = None,
) -> Callable[[Callable[..., Any]], _ToolWrapper]:
    """Decorate a function as an Atlas tool.

    Automatically generates a ``ToolDefinition`` from the function's
    signature and docstring.

    Args:
        name: Optional override for the tool name (defaults to function name).
        description: Optional description (defaults to function docstring).
        metadata: Optional ``ToolMetadata``.

    Returns:
        A decorator that wraps the function as a ``_ToolWrapper``.

    Usage::

        @tool(name="calculator", description="Evaluate a math expression")
        async def calculator(expression: str) -> str:
            \"\"\"Calculate the result of *expression*.\"\"\"
            ...

    The decorated function is callable directly (``calculator(expression="2+2")``)
    and can be registered with a ``ToolRegistry``.
    """

    def decorator(fn: Callable[..., Any]) -> _ToolWrapper:
        tool_name = name or fn.__name__
        tool_description = (
            description
            or (fn.__doc__ and fn.__doc__.strip().split("\n")[0])
            or ""
        )

        parameters, required = _extract_parameters(fn)
        resolved_metadata = metadata or ToolMetadata()

        definition = ToolDefinition(
            name=tool_name,
            description=tool_description,
            parameters=tuple(parameters),
            required=tuple(required),
            metadata=resolved_metadata,
            fn=fn,
        )

        return _ToolWrapper(fn=fn, definition=definition)

    return decorator


def _extract_parameters(
    fn: Callable[..., Any],
) -> tuple[list[ToolParameter], list[str]]:
    """Extract ``ToolParameter`` list from a function's signature.

    Reads type hints and default values to build JSON Schema-compatible
    parameter definitions.

    Returns:
        ``(parameters, required_names)`` tuple.
    """
    sig = inspect.signature(fn)
    type_hints: dict[str, Any] = {}
    try:
        type_hints = get_type_hints(fn)
    except (NameError, AttributeError, TypeError):
        pass

    parameters: list[ToolParameter] = []
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls", "return", "args", "kwargs"):
            continue

        # Determine type
        py_type = type_hints.get(param_name, str)
        json_type = _py_type_to_json_type(py_type)

        # Determine if required
        is_required = param.default is inspect.Parameter.empty
        default_value = (
            None
            if param.default is inspect.Parameter.empty
            else param.default
        )

        # Description from annotation (e.g. str: "The expression to evaluate")
        description = ""
        if hasattr(py_type, "__metadata__"):
            try:
                args = getattr(py_type, "__args__", ())
                if args:
                    description = str(args[0]) if args else ""
            except Exception:
                pass

        tool_param = ToolParameter(
            name=param_name,
            type=json_type,
            description=description,
            required=is_required,
            default=default_value,
        )
        parameters.append(tool_param)
        if is_required:
            required.append(param_name)

    return parameters, required


def _py_type_to_json_type(py_type: Any) -> str:
    """Map a Python type to a JSON Schema type string."""
    # Handle common types
    origin = getattr(py_type, "__origin__", None)
    if origin is not None:
        py_type = origin

    if py_type is str or py_type is type(None) or py_type is Any:
        return "string"
    if py_type is int:
        return "integer"
    if py_type is float:
        return "number"
    if py_type is bool:
        return "boolean"
    if py_type is list or py_type is tuple:
        return "array"
    if py_type is dict:
        return "object"

    # Handle Optional / Union types — already caught by str fallback above
    if origin is not None:
        pass

    return "string"
