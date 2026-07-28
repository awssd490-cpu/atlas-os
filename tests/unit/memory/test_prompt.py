"""Tests for the Prompt Assembly Engine.

Verifies:
- PromptSection, PromptStatistics, PromptDocument domain models
- PromptConfig defaults and customisation
- SectionTemplate: memory rendering, metadata formatting
- PromptAssemblyEngine: assembly, filtering, ordering, dedup
- TextRenderer: plain text output
- MarkdownRenderer: markdown output
- StructuredRenderer: JSON output
- Deterministic output
- Empty context
- Config propagation
- ContextPackage integration
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from app.memory.memory import Memory, MemoryId, MemoryState
from app.memory.context import ContextPackage, ContextSection, ContextSource, ContextStatistics
from app.memory.prompt import (
    MarkdownRenderer,
    PromptAssemblyEngine,
    PromptConfig,
    PromptDocument,
    PromptSection,
    PromptStatistics,
    SectionTemplate,
    StructuredRenderer,
    TextRenderer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_memory(content: str, memory_id: str = "", importance: float = 0.5, **kw: Any) -> Memory:
    return Memory(content=content, memory_id=MemoryId(memory_id or ""), importance=importance, **kw)


def _make_section(
    section_type: str,
    contents: list[str],
    *,
    label: str = "",
) -> ContextSection:
    mems = [_make_memory(c, f"{section_type}-{i}") for i, c in enumerate(contents)]
    sources = [ContextSource(source_type=section_type, memory_id=m.id.value) for m in mems]
    return ContextSection(
        section_type=section_type,
        label=label or section_type,
        memories=mems,
        sources=sources,
        token_count=sum(len(c) for c in contents),
    )


def _make_package(sections: list[ContextSection], request_id: str = "test") -> ContextPackage:
    tokens = sum(s.token_count for s in sections)
    stats = ContextStatistics(total_tokens=tokens, total_sections=len(sections))
    return ContextPackage(
        request_id=request_id,
        sections=sections,
        statistics=stats,
        metadata={"source": "test"},
    )


# ---------------------------------------------------------------------------
# Domain model tests
# ---------------------------------------------------------------------------


class TestPromptSection:
    def test_create(self) -> None:
        ps = PromptSection(section_type="memories", label="Memories", content_lines=["line1", "line2"])
        assert ps.section_type == "memories"
        assert ps.content == "line1\nline2"
        assert ps.is_empty is False

    def test_empty(self) -> None:
        ps = PromptSection(section_type="empty")
        assert ps.is_empty is True
        assert ps.content == ""

    def test_immutable(self) -> None:
        ps = PromptSection(section_type="test")
        with pytest.raises(AttributeError):
            ps.section_type = "changed"  # type: ignore[misc]


class TestPromptStatistics:
    def test_defaults(self) -> None:
        ps = PromptStatistics()
        assert ps.total_sections == 0
        assert ps.total_lines == 0

    def test_with_values(self) -> None:
        ps = PromptStatistics(total_sections=3, total_lines=15, total_chars=100)
        assert ps.total_sections == 3


class TestPromptDocument:
    def test_empty(self) -> None:
        doc = PromptDocument()
        assert doc.total_sections == 0
        assert doc.section_types == []

    def test_with_sections(self) -> None:
        sections = [
            PromptSection(section_type="a", content_lines=["x"]),
            PromptSection(section_type="b", content_lines=["y", "z"]),
        ]
        doc = PromptDocument(sections=sections)
        assert doc.total_sections == 2
        assert doc.section_types == ["a", "b"]

    def test_to_dict(self) -> None:
        sections = [PromptSection(section_type="test", label="Test", content_lines=["hello"])]
        stats = PromptStatistics(total_sections=1, total_lines=1, total_chars=5)
        doc = PromptDocument(request_id="r1", sections=sections, statistics=stats)
        d = doc.to_dict()
        assert d["request_id"] == "r1"
        assert len(d["sections"]) == 1
        assert d["sections"][0]["content"] == "hello"
        assert d["statistics"]["total_lines"] == 1


# ---------------------------------------------------------------------------
# PromptConfig
# ---------------------------------------------------------------------------


class TestPromptConfig:
    def test_defaults(self) -> None:
        cfg = PromptConfig()
        assert "user_query" in cfg.section_labels
        assert cfg.deterministic is True
        assert cfg.include_section_labels is True

    def test_custom(self) -> None:
        cfg = PromptConfig(
            include_section_types=["user_query", "relevant_memories"],
            exclude_section_types=["metadata"],
            include_section_labels=False,
        )
        assert cfg.include_section_types == ["user_query", "relevant_memories"]
        assert cfg.include_section_labels is False


# ---------------------------------------------------------------------------
# SectionTemplate
# ---------------------------------------------------------------------------


class TestSectionTemplate:
    def test_render_memory(self) -> None:
        template = SectionTemplate()
        mem = _make_memory("hello world", importance=0.8, tags=["a", "b"], source="user", namespace="ns")
        lines = template.render_memory(mem)
        assert lines[0] == "hello world"
        assert "importance=0.80" in lines[1]
        assert "tags=a,b" in lines[1]
        assert "source=user" in lines[1]

    def test_render_memory_no_metadata(self) -> None:
        template = SectionTemplate()
        mem = _make_memory("just content")
        lines = template.render_memory(mem, include_metadata=False)
        assert len(lines) == 1
        assert lines[0] == "just content"

    def test_render_section(self) -> None:
        template = SectionTemplate()
        section = _make_section("test", ["hello", "world"])
        lines = template.render_section(section)
        assert len(lines) >= 2

    def test_render_section_max_lines(self) -> None:
        template = SectionTemplate()
        section = _make_section("test", ["a", "b", "c", "d"])
        lines = template.render_section(section, max_lines=3)
        assert len(lines) <= 3


# ---------------------------------------------------------------------------
# PromptAssemblyEngine
# ---------------------------------------------------------------------------


class TestPromptAssemblyEngine:
    def test_assemble_empty_package(self) -> None:
        engine = PromptAssemblyEngine()
        pkg = _make_package([])
        doc = engine.assemble(pkg)
        assert doc.total_sections == 0

    def test_assemble_preserves_sections(self) -> None:
        engine = PromptAssemblyEngine()
        sections = [
            _make_section("user_query", ["hello"]),
            _make_section("relevant_memories", ["memory1"]),
        ]
        pkg = _make_package(sections)
        doc = engine.assemble(pkg)
        assert doc.total_sections == 2
        assert doc.section_types == ["user_query", "relevant_memories"]

    def test_assemble_deduplicates_by_type(self) -> None:
        engine = PromptAssemblyEngine()
        sections = [
            _make_section("relevant", ["first"]),
            _make_section("relevant", ["second"]),
        ]
        pkg = _make_package(sections)
        doc = engine.assemble(pkg)
        # Only the first "relevant" should be kept
        assert doc.total_sections == 1
        assert "first" in doc.sections[0].content_lines[0]

    def test_assemble_exclude_filter(self) -> None:
        cfg = PromptConfig(exclude_section_types=["metadata"])
        engine = PromptAssemblyEngine(config=cfg)
        sections = [
            _make_section("user_query", ["hi"]),
            _make_section("metadata", ["some data"]),
        ]
        pkg = _make_package(sections)
        doc = engine.assemble(pkg)
        assert doc.total_sections == 1
        assert doc.section_types == ["user_query"]

    def test_assemble_include_filter(self) -> None:
        cfg = PromptConfig(include_section_types=["user_query"])
        engine = PromptAssemblyEngine(config=cfg)
        sections = [
            _make_section("user_query", ["hi"]),
            _make_section("relevant", ["data"]),
        ]
        pkg = _make_package(sections)
        doc = engine.assemble(pkg)
        assert doc.total_sections == 1

    def test_assemble_skips_empty_sections(self) -> None:
        engine = PromptAssemblyEngine()
        empty_sec = ContextSection(section_type="empty")
        non_empty = _make_section("data", ["content"])
        pkg = _make_package([empty_sec, non_empty])
        doc = engine.assemble(pkg)
        assert doc.total_sections == 1
        assert doc.section_types == ["data"]

    def test_assemble_custom_labels(self) -> None:
        cfg = PromptConfig(section_labels={"user_query": "Custom Query Label"})
        engine = PromptAssemblyEngine(config=cfg)
        sections = [_make_section("user_query", ["hello"])]
        pkg = _make_package(sections)
        doc = engine.assemble(pkg)
        assert doc.sections[0].label == "Custom Query Label"

    def test_assemble_no_labels(self) -> None:
        cfg = PromptConfig(include_section_labels=False)
        engine = PromptAssemblyEngine(config=cfg)
        sections = [_make_section("user_query", ["hello"])]
        pkg = _make_package(sections)
        doc = engine.assemble(pkg)
        assert doc.sections[0].label == ""

    def test_deterministic_output(self) -> None:
        engine = PromptAssemblyEngine()
        sections = [
            _make_section("a", ["x"]),
            _make_section("b", ["y"]),
        ]
        pkg = _make_package(sections)
        doc1 = engine.assemble(pkg)
        doc2 = engine.assemble(pkg)
        assert len(doc1.sections) == len(doc2.sections)
        for s1, s2 in zip(doc1.sections, doc2.sections):
            assert s1.content == s2.content

    def test_assemble_produces_statistics(self) -> None:
        engine = PromptAssemblyEngine()
        sections = [_make_section("a", ["hello world"]), _make_section("b", ["foo", "bar"])]
        pkg = _make_package(sections)
        doc = engine.assemble(pkg)
        assert doc.statistics.total_sections == 2
        assert doc.statistics.total_lines >= 2
        assert doc.statistics.total_chars > 0

    def test_assemble_max_lines_per_section(self) -> None:
        cfg = PromptConfig(max_lines_per_section=2)
        engine = PromptAssemblyEngine(config=cfg)
        sections = [_make_section("data", ["a", "b", "c", "d", "e"])]
        pkg = _make_package(sections)
        doc = engine.assemble(pkg)
        # Each memory generates content + metadata = 2 lines
        # But max_lines_per_section=2 means at most 2 content lines total
        assert len(doc.sections[0].content_lines) <= 2


# ---------------------------------------------------------------------------
# TextRenderer
# ---------------------------------------------------------------------------


class TestTextRenderer:
    def test_render(self) -> None:
        renderer = TextRenderer()
        sections = [PromptSection(section_type="test", label="Test", content_lines=["hello", "world"])]
        doc = PromptDocument(sections=sections)
        output = renderer.render(doc)
        assert "Test" in output
        assert "hello" in output
        assert "world" in output

    def test_render_empty_section(self) -> None:
        renderer = TextRenderer()
        sections = [
            PromptSection(section_type="empty", label="Empty"),
            PromptSection(section_type="data", label="Data", content_lines=["content"]),
        ]
        doc = PromptDocument(sections=sections)
        output = renderer.render(doc)
        assert "Empty" not in output
        assert "Data" in output

    def test_render_empty_document(self) -> None:
        renderer = TextRenderer()
        doc = PromptDocument()
        output = renderer.render(doc)
        assert output == ""


# ---------------------------------------------------------------------------
# MarkdownRenderer
# ---------------------------------------------------------------------------


class TestMarkdownRenderer:
    def test_render(self) -> None:
        renderer = MarkdownRenderer()
        sections = [PromptSection(section_type="test", label="Test Section", content_lines=["hello", "world"])]
        doc = PromptDocument(sections=sections)
        output = renderer.render(doc)
        assert "## Test Section" in output
        assert "hello" in output
        assert "---" in output or "\n\n" in output

    def test_render_metadata_blockquote(self) -> None:
        renderer = MarkdownRenderer()
        sections = [PromptSection(section_type="stats", label="Stats", content_lines=[
            "some content",
            "importance=0.90",
            "[metadata] source=user",
        ])]
        doc = PromptDocument(sections=sections)
        output = renderer.render(doc)
        assert "> importance=0.90" in output
        assert "> [metadata]" in output

    def test_render_empty(self) -> None:
        renderer = MarkdownRenderer()
        doc = PromptDocument()
        output = renderer.render(doc)
        assert output == ""


# ---------------------------------------------------------------------------
# StructuredRenderer
# ---------------------------------------------------------------------------


class TestStructuredRenderer:
    def test_render(self) -> None:
        renderer = StructuredRenderer()
        sections = [PromptSection(section_type="test", label="Test", content_lines=["hello"])]
        doc = PromptDocument(request_id="r1", sections=sections)
        output = renderer.render(doc)
        data = json.loads(output)
        assert data["request_id"] == "r1"
        assert len(data["sections"]) == 1
        assert data["sections"][0]["content"] == "hello"

    def test_render_empty(self) -> None:
        renderer = StructuredRenderer()
        output = renderer.render(PromptDocument())
        data = json.loads(output)
        assert data["sections"] == []


# ---------------------------------------------------------------------------
# End-to-end assembly + rendering
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_full_pipeline(self) -> None:
        """ContextPackage → PromptAssemblyEngine → TextRenderer."""
        sections = [
            _make_section("user_query", ["What is the status?"]),
            _make_section("relevant_memories", ["Project is on track", "Sprint complete"]),
            _make_section("working_memory", ["Current focus: testing"]),
        ]
        pkg = _make_package(sections)
        engine = PromptAssemblyEngine()
        doc = engine.assemble(pkg)
        renderer = TextRenderer()
        output = renderer.render(doc)
        assert "What is the status?" in output
        assert "Project is on track" in output
        assert "Sprint complete" in output
        assert "Current focus: testing" in output

    def test_markdown_pipeline(self) -> None:
        """ContextPackage → PromptAssemblyEngine → MarkdownRenderer."""
        sections = [
            _make_section("user_query", ["Hello"]),
            _make_section("data", ["Important fact"]),
        ]
        pkg = _make_package(sections)
        engine = PromptAssemblyEngine()
        doc = engine.assemble(pkg)
        renderer = MarkdownRenderer()
        output = renderer.render(doc)
        assert "## " in output
        assert "Hello" in output
        assert "Important fact" in output

    def test_structured_pipeline(self) -> None:
        """ContextPackage → PromptAssemblyEngine → StructuredRenderer."""
        sections = [_make_section("query", ["test"])]
        pkg = _make_package(sections, request_id="e2e")
        engine = PromptAssemblyEngine()
        doc = engine.assemble(pkg)
        renderer = StructuredRenderer()
        output = renderer.render(doc)
        data = json.loads(output)
        assert data["request_id"] == "e2e"

    def test_preserves_content_order(self) -> None:
        """Section order in the package should be preserved in the document."""
        sections = [
            _make_section("first", ["alpha"]),
            _make_section("second", ["beta"]),
            _make_section("third", ["gamma"]),
        ]
        pkg = _make_package(sections)
        engine = PromptAssemblyEngine()
        doc = engine.assemble(pkg)
        types = doc.section_types
        assert types == ["first", "second", "third"]
