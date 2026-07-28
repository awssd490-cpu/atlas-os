"""Request and response mappers for the Claude API.

Maps between Atlas domain models and Anthropic Messages API format.
"""

from __future__ import annotations

from typing import Any

from app.provider.models import (
    ContentType,
    ProviderMessage,
    ProviderRequest,
    ProviderResponse,
    ProviderUsage,
    Role,
    StopReason,
    StreamingChunk,
    ToolCallRequest,
)


# ---------------------------------------------------------------------------
# Request mapping
# ---------------------------------------------------------------------------


class ClaudeRequestMapper:
    """Maps ``ProviderRequest`` → Anthropic Messages API request body."""

    ANTHROPIC_VERSION = "2023-06-01"

    def to_dict(self, request: ProviderRequest) -> dict[str, Any]:
        """Convert *request* to an Anthropic Messages API payload.

        Returns:
            A dict ready for JSON serialization and HTTP POST.
        """
        body: dict[str, Any] = {
            "model": request.metadata.get("model", "claude-sonnet-4-20250514"),
            "max_tokens": request.max_tokens,
            "messages": self._map_messages(request),
        }

        if request.system:
            body["system"] = request.system

        if request.temperature != 0.7 or "temperature" in request.metadata:
            body["temperature"] = request.metadata.get("temperature", request.temperature)

        if request.top_p != 1.0 or "top_p" in request.metadata:
            body["top_p"] = request.metadata.get("top_p", request.top_p)

        if request.stop_sequences:
            body["stop_sequences"] = request.stop_sequences

        if request.metadata.get("anthropic_version"):
            body["anthropic_version"] = request.metadata["anthropic_version"]

        return body

    def _map_messages(self, request: ProviderRequest) -> list[dict[str, Any]]:
        """Map ``ProviderMessage`` list to Anthropic messages format."""
        messages: list[dict[str, Any]] = []
        for msg in request.messages:
            mapped = self._map_message(msg)
            if mapped:
                messages.append(mapped)
        return messages

    @staticmethod
    def _map_message(msg: ProviderMessage) -> dict[str, Any] | None:
        """Map a single ``ProviderMessage`` to Anthropic format."""
        role = msg.role.value if isinstance(msg.role, Role) else msg.role

        if role == "system":
            return None  # system handled separately

        entry: dict[str, Any] = {"role": role, "content": msg.content}
        if msg.name:
            entry["name"] = msg.name
        return entry


# ---------------------------------------------------------------------------
# Response mapping
# ---------------------------------------------------------------------------


class ClaudeResponseMapper:
    """Maps Anthropic Messages API response → ``ProviderResponse``."""

    STOP_REASON_MAP: dict[str, StopReason] = {
        "end_turn": StopReason.STOP,
        "max_tokens": StopReason.LENGTH,
        "stop_sequence": StopReason.STOP,
        "tool_use": StopReason.TOOL_CALL,
        "error": StopReason.ERROR,
    }

    def to_response(self, data: dict[str, Any]) -> ProviderResponse:
        """Convert a Claude API response dict to ``ProviderResponse``."""
        content = self._extract_text(data)
        stop_reason = self._map_stop_reason(data.get("stop_reason", ""))
        usage = self._map_usage(data.get("usage", {}))
        tool_calls = self._extract_tool_calls(data)

        return ProviderResponse(
            content=content,
            message=ProviderMessage(
                role=Role.ASSISTANT,
                content=content,
                tool_calls=tool_calls,
            ),
            stop_reason=stop_reason,
            usage=usage,
            tool_calls=tool_calls,
            metadata={"raw_stop_reason": data.get("stop_reason")},
        )

    def to_chunk(self, event_type: str, data: dict[str, Any]) -> StreamingChunk | None:
        """Convert a Claude streaming event to a ``StreamingChunk``.

        Returns ``None`` for events that should be skipped (ping, etc.).
        """
        if event_type == "ping":
            return None

        if event_type == "message_start":
            message = data.get("message", {})
            usage = self._map_usage(message.get("usage", {}))
            return StreamingChunk(content="", usage=usage if usage.total_tokens > 0 else None)

        if event_type == "content_block_delta":
            delta = data.get("delta", {})
            if delta.get("type") == "text_delta":
                return StreamingChunk(content=delta.get("text", ""))
            return None

        if event_type == "content_block_start":
            block = data.get("content_block", {})
            if block.get("type") == "text":
                return StreamingChunk(content=block.get("text", ""))
            return None

        if event_type == "message_delta":
            delta = data.get("delta", {})
            stop_reason = self._map_stop_reason(delta.get("stop_reason", ""))
            usage = self._map_usage(data.get("usage", {}))
            return StreamingChunk(
                content="",
                stop_reason=stop_reason,
                usage=usage if usage.total_tokens > 0 else None,
            )

        return None

    @staticmethod
    def _extract_text(data: dict[str, Any]) -> str:
        """Extract concatenated text from Claude response content blocks."""
        parts: list[str] = []
        for block in data.get("content", []):
            if block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "".join(parts)

    def _map_stop_reason(self, reason: str) -> StopReason:
        return self.STOP_REASON_MAP.get(reason, StopReason.UNKNOWN)

    @staticmethod
    def _map_usage(usage: dict[str, Any]) -> ProviderUsage:
        return ProviderUsage(
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            total_tokens=usage.get("input_tokens", 0) + usage.get("output_tokens", 0),
        )

    @staticmethod
    def _extract_tool_calls(data: dict[str, Any]) -> list[ToolCallRequest]:
        """Extract tool calls from Claude response content blocks."""
        calls: list[ToolCallRequest] = []
        for block in data.get("content", []):
            if block.get("type") == "tool_use":
                calls.append(ToolCallRequest(
                    id=block.get("id", ""),
                    name=block.get("name", ""),
                    arguments=block.get("input", {}),
                ))
        return calls
