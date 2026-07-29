"""Tests for tool error types."""

from __future__ import annotations

import pytest

from app.tools.errors import (
    DuplicateToolError,
    ToolError,
    ToolExecutionError,
    ToolInvalidArgumentsError,
    ToolNotFoundError,
    ToolValidationError,
)


class TestToolErrorHierarchy:
    def test_tool_error_is_base(self) -> None:
        assert issubclass(ToolNotFoundError, ToolError)
        assert issubclass(DuplicateToolError, ToolError)
        assert issubclass(ToolValidationError, ToolError)
        assert issubclass(ToolInvalidArgumentsError, ToolError)
        assert issubclass(ToolExecutionError, ToolError)

    def test_tool_error_has_code(self) -> None:
        error = ToolError("test")
        assert error.code == "TOOL_ERROR"

    def test_tool_error_has_details(self) -> None:
        error = ToolError("test", details={"key": "value"})
        assert error.details == {"key": "value"}


class TestToolNotFoundError:
    def test_with_name(self) -> None:
        error = ToolNotFoundError(name="my_tool")
        assert "my_tool" in str(error)
        assert error.code == "TOOL_NOT_FOUND"

    def test_without_name(self) -> None:
        error = ToolNotFoundError()
        assert "not found" in str(error).lower() or str(error) == "Tool not found"

    def test_with_details(self) -> None:
        error = ToolNotFoundError("ghost", details={"available": ["a", "b"]})
        assert error.details["available"] == ["a", "b"]


class TestDuplicateToolError:
    def test_with_name(self) -> None:
        error = DuplicateToolError(name="calc")
        assert "calc" in str(error)
        assert error.code == "DUPLICATE_TOOL"

    def test_without_name(self) -> None:
        error = DuplicateToolError()
        assert "Duplicate" in str(error)


class TestToolValidationError:
    def test_with_name(self) -> None:
        error = ToolValidationError(name="calc", message="type mismatch")
        assert "calc" in str(error)
        assert "type mismatch" in str(error)
        assert error.code == "TOOL_VALIDATION_ERROR"

    def test_without_name(self) -> None:
        error = ToolValidationError(message="validation failed")
        assert "validation failed" in str(error)


class TestToolInvalidArgumentsError:
    def test_with_name_and_message(self) -> None:
        error = ToolInvalidArgumentsError(
            name="calc",
            message="Missing required parameter 'x'",
        )
        assert "calc" in str(error)
        assert "Missing required" in str(error)
        assert error.code == "TOOL_INVALID_ARGUMENTS"

    def test_defaults(self) -> None:
        error = ToolInvalidArgumentsError()
        assert error.message is not None


class TestToolExecutionError:
    def test_with_name(self) -> None:
        error = ToolExecutionError(name="calc")
        assert "calc" in str(error)
        assert error.code == "TOOL_EXECUTION_ERROR"

    def test_with_original_exception(self) -> None:
        original = ValueError("something broke")
        error = ToolExecutionError(
            name="crash",
            original_exception=original,
        )
        assert error.details.get("original_error") is not None
        assert "ValueError" in error.details["original_error"]

    def test_no_traceback_exposure(self) -> None:
        """Original exception is captured, not the raw traceback."""
        original = RuntimeError("secret internals")
        error = ToolExecutionError(
            name="safe",
            original_exception=original,
        )
        # The original traceback should NOT be in the message
        assert "Traceback" not in str(error)
        assert "secret internals" in str(error) or "secret" in error.details.get("original_error", "")
