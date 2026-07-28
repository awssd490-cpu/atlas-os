"""Provider-agnostic domain models.

Every model in this module is immutable and independent of any specific
provider implementation.  They represent the canonical data types that
flow through the Universal Provider Runtime.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class StopReason(str, enum.Enum):
    """Why generation stopped."""

    STOP = "stop"
    LENGTH = "length"
    TOOL_CALL = "tool_call"
    CONTENT_FILTER = "content_filter"
    ERROR = "error"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    UNKNOWN = "unknown"


FinishReason = StopReason  # alias for semantic clarity


class Role(str, enum.Enum):
    """Message role in a conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ContentType(str, enum.Enum):
    """Type of content within a message."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


# ---------------------------------------------------------------------------
# Capability declarations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderCapability:
    """A single capability a provider supports."""

    name: str = ""
    version: str = "1.0"
    description: str = ""


@dataclass(frozen=True)
class ProviderMetadata:
    """Static metadata about a provider."""

    name: str = ""
    version: str = ""
    description: str = ""
    website: str = ""
    documentation: str = ""


@dataclass(frozen=True)
class ProviderInfo:
    """Full provider descriptor combining metadata and capabilities."""

    metadata: ProviderMetadata = field(default_factory=ProviderMetadata)
    capabilities: list[ProviderCapability] = field(default_factory=list)

    def has_capability(self, name: str) -> bool:
        return any(c.name == name for c in self.capabilities)

    @property
    def capability_names(self) -> list[str]:
        return [c.name for c in self.capabilities]


# ---------------------------------------------------------------------------
# Messages
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolCallRequest:
    """A request from the provider to call a tool."""

    id: str = ""
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCallResponse:
    """The result of a tool call, to be sent back to the provider."""

    call_id: str = ""
    output: str = ""


@dataclass(frozen=True)
class ProviderMessage:
    """A single message in a conversation exchange."""

    role: Role = Role.USER
    content: str = ""
    content_type: ContentType = ContentType.TEXT
    name: str = ""
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    tool_call_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProviderUsage:
    """Token usage for a provider request."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @property
    def ratio(self) -> float:
        if self.prompt_tokens == 0:
            return 0.0
        return self.completion_tokens / self.prompt_tokens


# ---------------------------------------------------------------------------
# Request / Response
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProviderRequest:
    """A complete request to any provider.

    Every field is provider-agnostic.  Adapters translate this into
    provider-specific payloads.
    """

    messages: list[ProviderMessage] = field(default_factory=list)
    system: str = ""
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 1.0
    stop_sequences: list[str] = field(default_factory=list)
    tools: list[dict[str, Any]] = field(default_factory=list)
    stream: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def message_count(self) -> int:
        return len(self.messages)


@dataclass(frozen=True)
class ProviderResponse:
    """A complete (non-streaming) response from any provider."""

    content: str = ""
    message: ProviderMessage = field(default_factory=ProviderMessage)
    stop_reason: StopReason = StopReason.STOP
    usage: ProviderUsage = field(default_factory=ProviderUsage)
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "ProviderResponse":
        return cls()


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StreamingChunk:
    """A single chunk from a streaming response."""

    content: str = ""
    stop_reason: StopReason | None = None
    usage: ProviderUsage | None = None
    tool_call: ToolCallRequest | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    index: int = 0


@dataclass(frozen=True)
class StreamingResult:
    """The accumulated result of a streaming request."""

    full_content: str = ""
    chunks: list[StreamingChunk] = field(default_factory=list)
    stop_reason: StopReason = StopReason.STOP
    usage: ProviderUsage = field(default_factory=ProviderUsage)
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Capability constants
# ---------------------------------------------------------------------------


class Capabilities:
    """Well-known capability names for discovery and filtering."""

    STREAMING = "streaming"
    TOOL_CALLING = "tool_calling"
    VISION = "vision"
    AUDIO = "audio"
    EMBEDDINGS = "embeddings"
    JSON_MODE = "json_mode"
    REASONING = "reasoning"
    FUNCTION_CALLING = "function_calling"
    TEMPERATURE = "temperature"
    SYSTEM_PROMPTS = "system_prompts"
    STOP_SEQUENCES = "stop_sequences"
    MULTIPLE_TOOL_CALLS = "multiple_tool_calls"
    PARALLEL_TOOL_CALLS = "parallel_tool_calls"
    CONTEXT_CACHING = "context_caching"
    BATCHING = "batching"
