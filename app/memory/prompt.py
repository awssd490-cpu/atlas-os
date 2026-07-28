"""Prompt Assembly Engine — converts ContextPackage into structured PromptDocument.

This module knows NOTHING about:

- Claude, GPT, Gemini, Ollama, OpenRouter
- Provider APIs
- LLM clients
- Tool calling

It produces a canonical, structured prompt representation that any
provider formatter can consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from app.memory.context import ContextPackage, ContextSection, ContextSource
from app.memory.memory import Memory


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptSection:
    """A single section in a prompt document.

    Each section has a type, label, content lines, and metadata.
    Sections are ordered by priority.
    """

    section_type: str = ""
    label: str = ""
    content_lines: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    token_count: int = 0

    @property
    def content(self) -> str:
        return "\n".join(self.content_lines)

    @property
    def is_empty(self) -> bool:
        return not self.content_lines


@dataclass(frozen=True)
class PromptStatistics:
    """Aggregated statistics for a prompt assembly run."""

    total_sections: int = 0
    total_lines: int = 0
    total_chars: int = 0
    source_section_types: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class PromptDocument:
    """The final, structured, provider-agnostic prompt.

    This is the output of the Prompt Assembly Engine.  Renderers
    convert this to text, markdown, or structured formats.

    The document preserves section structure — renderers decide
    how to flatten.
    """

    request_id: str = ""
    sections: list[PromptSection] = field(default_factory=list)
    statistics: PromptStatistics = field(default_factory=PromptStatistics)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_sections(self) -> int:
        return len(self.sections)

    @property
    def section_types(self) -> list[str]:
        return [s.section_type for s in self.sections]

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "sections": [
                {
                    "section_type": s.section_type,
                    "label": s.label,
                    "line_count": len(s.content_lines),
                    "content": s.content,
                }
                for s in self.sections
            ],
            "statistics": {
                "total_sections": self.statistics.total_sections,
                "total_lines": self.statistics.total_lines,
                "total_chars": self.statistics.total_chars,
            },
        }


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class PromptConfig:
    """Configuration for the Prompt Assembly Engine."""

    # Section labels (overrides)
    section_labels: dict[str, str] = field(default_factory=lambda: {
        "user_query": "User Query",
        "working_memory": "Working Memory",
        "relevant_memories": "Relevant Memories",
        "related_memories": "Related Memories",
        "long_term_facts": "Long-Term Facts",
        "conversation_history": "Conversation History",
        "retrieved_metadata": "Metadata",
    })

    # Sections to include (empty = include all present)
    include_section_types: list[str] = field(default_factory=list)

    # Sections to exclude
    exclude_section_types: list[str] = field(default_factory=list)

    # Maximum content lines per section (0 = unlimited)
    max_lines_per_section: int = 0

    # Include source metadata in content
    include_source_metadata: bool = True

    # Include token count estimates
    include_token_estimates: bool = True

    # Include section labels as headers
    include_section_labels: bool = True

    # Separator between sections
    section_separator: str = "---"

    # Maximum line length (0 = no limit)
    max_line_length: int = 0

    # Whether assembly is deterministic (always True for this engine)
    deterministic: bool = True


# ---------------------------------------------------------------------------
# Renderer protocol
# ---------------------------------------------------------------------------


class PromptRenderer(Protocol):
    """Converts a PromptDocument into an output format.

    Implementations are independent of provider-specific formatting.
    """

    def render(self, document: PromptDocument) -> str:
        """Render *document* to a string.

        Args:
            document: The prompt document to render.

        Returns:
            The rendered prompt as a string.
        """
        ...


# ---------------------------------------------------------------------------
# Built-in renderers
# ---------------------------------------------------------------------------


class TextRenderer:
    """Renders PromptDocument as plain text.

    Each section is rendered with its label as a header line followed
    by its content lines.  Sections are separated by a configurable
    separator.
    """

    def __init__(self, separator: str = "\n---\n") -> None:
        self._separator = separator

    def render(self, document: PromptDocument) -> str:
        parts: list[str] = []
        for section in document.sections:
            if section.is_empty:
                continue
            section_parts: list[str] = []
            if section.label:
                section_parts.append(section.label)
                section_parts.append("")
            section_parts.extend(section.content_lines)
            parts.append("\n".join(section_parts))
        return self._separator.join(parts)


class MarkdownRenderer:
    """Renders PromptDocument as GitHub-flavoured Markdown.

    Section labels become ``##`` headings.  Memory content is rendered
    as paragraphs or list items.  Metadata is rendered as blockquotes.
    """

    def __init__(self, separator: str = "\n---\n") -> None:
        self._separator = separator

    def render(self, document: PromptDocument) -> str:
        parts: list[str] = []
        for section in document.sections:
            if section.is_empty:
                continue
            section_parts: list[str] = []
            if section.label:
                section_parts.append(f"## {section.label}")
                section_parts.append("")
            for line in section.content_lines:
                if line.startswith("[metadata]") or line.startswith("score=") or line.startswith("importance="):
                    section_parts.append(f"> {line}")
                else:
                    section_parts.append(line)
            parts.append("\n".join(section_parts))
        return self._separator.join(parts)


class StructuredRenderer:
    """Renders PromptDocument as a structured dict.

    The output retains section structure and metadata.  Useful for
    programmatic consumption by provider adapters.
    """

    def render(self, document: PromptDocument) -> str:
        """Render as a pretty-printed JSON string."""
        import json

        data = document.to_dict()
        return json.dumps(data, indent=2, default=str)


# ---------------------------------------------------------------------------
# Default section templates
# ---------------------------------------------------------------------------


class SectionTemplate:
    """Converts a ContextSection into content lines for a PromptSection."""

    def render_memory(self, memory: Memory, *, include_metadata: bool = True) -> list[str]:
        """Render a single Memory to content lines."""
        lines: list[str] = [memory.content]
        if include_metadata:
            meta_parts: list[str] = []
            meta_parts.append(f"importance={memory.importance:.2f}")
            if memory.tags:
                meta_parts.append(f"tags={','.join(memory.tags)}")
            if memory.source:
                meta_parts.append(f"source={memory.source}")
            if memory.namespace:
                meta_parts.append(f"namespace={memory.namespace}")
            if meta_parts:
                lines.append(f"[metadata] {' | '.join(meta_parts)}")
        return lines

    def render_section(
        self,
        section: ContextSection,
        *,
        max_lines: int = 0,
        include_metadata: bool = True,
    ) -> list[str]:
        """Convert a ContextSection to content lines."""
        lines: list[str] = []
        for mem in section.memories:
            mem_lines = self.render_memory(mem, include_metadata=include_metadata)
            lines.extend(mem_lines)
            if max_lines > 0 and len(lines) >= max_lines:
                lines = lines[:max_lines]
                break
        return lines


# ---------------------------------------------------------------------------
# PromptAssemblyEngine
# ---------------------------------------------------------------------------


class PromptAssemblyEngine:
    """Transforms a ContextPackage into a structured PromptDocument.

    Handles section ordering, filtering, template application,
    metadata injection, and structure assembly.

    Usage::

        engine = PromptAssemblyEngine(config=...)
        doc = engine.assemble(package)
        text = TextRenderer().render(doc)
    """

    def __init__(self, config: PromptConfig | None = None) -> None:
        self._config = config or PromptConfig()
        self._template = SectionTemplate()

    @property
    def config(self) -> PromptConfig:
        return self._config

    def assemble(self, package: ContextPackage) -> PromptDocument:
        """Assemble *package* into a ``PromptDocument``.

        Args:
            package: The context package to assemble.

        Returns:
            A structured PromptDocument.
        """
        cfg = self._config
        seen_types: set[str] = set()
        prompt_sections: list[PromptSection] = []

        for ctx_section in package.sections:
            # Filter by include/exclude lists
            st = ctx_section.section_type
            if cfg.include_section_types and st not in cfg.include_section_types:
                continue
            if st in cfg.exclude_section_types:
                continue

            # Skip empty sections
            if not ctx_section.memories:
                continue

            # Deduplicate by section type (keep first occurrence)
            if st in seen_types:
                continue
            seen_types.add(st)

            # Render content
            content_lines = self._template.render_section(
                ctx_section,
                max_lines=cfg.max_lines_per_section,
                include_metadata=cfg.include_source_metadata,
            )

            if not content_lines:
                continue

            # Determine label
            label = cfg.section_labels.get(st, ctx_section.label or st)

            # Estimate tokens
            token_count = ctx_section.token_count

            prompt_sections.append(PromptSection(
                section_type=st,
                label=label if cfg.include_section_labels else "",
                content_lines=content_lines,
                token_count=token_count,
            ))

        total_lines = sum(len(s.content_lines) for s in prompt_sections)
        total_chars = sum(len(s.content) for s in prompt_sections)

        stats = PromptStatistics(
            total_sections=len(prompt_sections),
            total_lines=total_lines,
            total_chars=total_chars,
            source_section_types=[s.section_type for s in prompt_sections],
        )

        return PromptDocument(
            request_id=package.request_id,
            sections=prompt_sections,
            statistics=stats,
            metadata={
                "source_package_tokens": package.total_tokens,
                **(package.metadata if cfg.include_token_estimates else {}),
            },
        )
