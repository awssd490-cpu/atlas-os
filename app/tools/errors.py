"""Tool error hierarchy.

Every tool-related error derives from ``ToolError`` so callers
can catch all tool issues distinctly from provider or platform errors.
"""

from __future__ import annotations

from typing import Any

from app.core.errors import AtlasError


class ToolError(AtlasError):
    """Base class for all tool errors."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "TOOL_ERROR",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message, code=code, details=details)


class ToolNotFoundError(ToolError):
    """Raised when a requested tool is not registered."""

    def __init__(
        self,
        name: str = "",
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        msg = f"Tool {name!r} not found" if name else "Tool not found"
        super().__init__(msg, code="TOOL_NOT_FOUND", details=details)


class DuplicateToolError(ToolError):
    """Raised when a tool is registered under an already-used name."""

    def __init__(
        self,
        name: str = "",
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        msg = f"Tool {name!r} is already registered" if name else "Duplicate tool registration"
        super().__init__(msg, code="DUPLICATE_TOOL", details=details)


class ToolValidationError(ToolError):
    """Raised when tool argument validation fails."""

    def __init__(
        self,
        name: str = "",
        message: str = "Tool validation failed",
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        full_msg = f"Tool {name!r}: {message}" if name else message
        super().__init__(full_msg, code="TOOL_VALIDATION_ERROR", details=details)


class ToolInvalidArgumentsError(ToolError):
    """Raised when required arguments are missing or unknown arguments provided."""

    def __init__(
        self,
        name: str = "",
        message: str = "Invalid tool arguments",
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        full_msg = f"Tool {name!r}: {message}" if name else message
        super().__init__(full_msg, code="TOOL_INVALID_ARGUMENTS", details=details)


class ToolExecutionError(ToolError):
    """Raised when a tool function raises an exception during execution.

    The original exception is captured in ``details`` and never exposed
    as a raw traceback to callers.
    """

    def __init__(
        self,
        name: str = "",
        message: str = "Tool execution failed",
        *,
        original_exception: Exception | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        full_msg = f"Tool {name!r}: {message}" if name else message
        merged_details = dict(details or {})
        if original_exception is not None:
            merged_details["original_error"] = f"{type(original_exception).__name__}: {original_exception}"
        super().__init__(full_msg, code="TOOL_EXECUTION_ERROR", details=merged_details)
