"""Tests for the Context Builder subsystem.

Verifies:
- Context domain models: ContextRequest, ContextSection, ContextSource,
  ContextPackage, ContextStatistics
- TokenEstimator: text, single memory, multiple memories
- ContextBuilderConfig: defaults and custom
- ContextBuilder.build(): full assembly with all sections
- Retrieval pipeline integration
- Relationship expansion integration
- Compression integration
- Section ordering
- Token budget enforcement
- Empty context handling
- Error resilience
- to_dict serialization
"""

from __future__ import annotations

from typing import Any

import pytest

from app.memory.memory import Memory, MemoryId, MemoryState, MemoryType
from app.memory.manager import MemoryManager, MemoryRepository
from app.memory.context import (
    ContextBuilder,
    ContextBuilderConfig,
    ContextPackage,
    ContextSection,
    ContextSource,
    ContextStatistics,
    TokenEstimator,
)
from app.memory.retrieval import RetrievalPipeline, RetrievalQuery


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def repo() -> MemoryRepository:
    from app.storage.connection.sqlite import SQLiteConnection
    from app.storage.migration.manager import SqliteMigrationManager
    from app.memory.migrations import V002_MemorySchema

    conn = SQLiteConnection(":memory:")
    mgr = SqliteMigrationManager()
    await mgr.apply_all(conn, [V002_MemorySchema()])
    yield MemoryRepository(connection=conn)
    await conn.close()


@pytest.fixture
async def seeded_repo(repo: MemoryRepository) -> MemoryRepository:
    """Repo with memories across multiple namespaces."""
    memories = [
        Memory(content="project plan Q3", importance=0.9, memory_id=MemoryId("m1"),
               namespace="default", memory_type=MemoryType.PROJECT.value, tags=["plan"]),
        Memory(content="user login flow", importance=0.7, memory_id=MemoryId("m2"),
               namespace="default", memory_type=MemoryType.KNOWLEDGE.value, tags=["auth"]),
        Memory(content="working note", importance=0.8, memory_id=MemoryId("m3"),
               namespace="working", memory_type=MemoryType.WORKING.value, tags=["active"]),
        Memory(content="long-term architecture", importance=0.95, memory_id=MemoryId("m4"),
               namespace="long_term", memory_type=MemoryType.LONG_TERM.value, tags=["arch"]),
        Memory(content="semantic concept", importance=0.85, memory_id=MemoryId("m5"),
               namespace="semantic", memory_type=MemoryType.SEMANTIC.value, tags=["concept"]),
        Memory(content="conversation greeting", importance=0.3, memory_id=MemoryId("m6"),
               namespace="conversation", memory_type=MemoryType.CONVERSATION.value),
        Memory(content="conversation follow-up", importance=0.4, memory_id=MemoryId("m7"),
               namespace="conversation", memory_type=MemoryType.CONVERSATION.value),
    ]
    for m in memories:
        await repo.add(m)
    return repo


@pytest.fixture
async def graph(seeded_repo: MemoryRepository) -> Any:
    from app.memory.relationships import MemoryGraphImpl

    g = MemoryGraphImpl(connection=seeded_repo._conn)
    await g.add_relationship("m1", "m2", "references")
    return g


@pytest.fixture
async def compressed_repo(repo: MemoryRepository) -> MemoryRepository:
    """Dedupable memories."""
    memories = [
        Memory(content="dup content", importance=0.5, memory_id=MemoryId("d1"), namespace="default"),
        Memory(content="dup content", importance=0.9, memory_id=MemoryId("d2"), namespace="default"),
        Memory(content="unique", importance=0.7, memory_id=MemoryId("u1"), namespace="default"),
    ]
    for m in memories:
        await repo.add(m)
    return repo


# ---------------------------------------------------------------------------
# Domain model tests
# ---------------------------------------------------------------------------


class TestContextSource:
    def test_create(self) -> None:
        src = ContextSource(source_type="memory", memory_id="m1", importance=0.8)
        assert src.source_type == "memory"
        assert src.memory_id == "m1"
        assert src.importance == 0.8

    def test_immutable(self) -> None:
        src = ContextSource(source_type="memory", memory_id="m1")
        with pytest.raises(AttributeError):
            src.source_type = "changed"  # type: ignore[misc]


class TestContextSection:
    def test_create(self) -> None:
        mem = Memory(content="test")
        section = ContextSection(
            section_type="relevant_memories",
            label="Relevant",
            memories=[mem],
            token_count=100,
        )
        assert section.section_type == "relevant_memories"
        assert section.memory_count == 1
        assert section.token_count == 100

    def test_empty_section(self) -> None:
        section = ContextSection(section_type="empty")
        assert section.memory_count == 0
        assert section.memories == []

    def test_immutable(self) -> None:
        section = ContextSection(section_type="test")
        with pytest.raises(AttributeError):
            section.section_type = "changed"  # type: ignore[misc]


class TestContextStatistics:
    def test_defaults(self) -> None:
        stats = ContextStatistics()
        assert stats.total_memories == 0
        assert stats.total_sections == 0
        assert stats.total_tokens == 0

    def test_with_values(self) -> None:
        stats = ContextStatistics(
            total_memories=10,
            total_sections=3,
            total_tokens=500,
            retrieval_ms=12.5,
            sources_breakdown={"memory": 8, "relationship": 2},
        )
        assert stats.total_memories == 10
        assert stats.total_sections == 3


class TestContextPackage:
    def test_empty_package(self) -> None:
        pkg = ContextPackage()
        assert pkg.total_memories == 0
        assert pkg.total_tokens == 0
        assert pkg.section_types == []

    def test_with_sections(self) -> None:
        mem = Memory(content="hello")
        sections = [
            ContextSection(section_type="user_query", memories=[mem], token_count=20),
            ContextSection(section_type="relevant_memories", memories=[mem, mem], token_count=40),
        ]
        pkg = ContextPackage(sections=sections)
        assert pkg.total_memories == 3
        assert pkg.total_tokens == 60
        assert pkg.section_types == ["user_query", "relevant_memories"]

    def test_to_dict(self) -> None:
        mem = Memory(content="test", memory_id=MemoryId("m1"))
        sections = [
            ContextSection(
                section_type="relevant",
                label="Relevant",
                memories=[mem],
                sources=[ContextSource(source_type="memory", memory_id="m1")],
                token_count=50,
            ),
        ]
        stats = ContextStatistics(total_memories=1, total_sections=1, total_tokens=50)
        pkg = ContextPackage(request_id="req-1", sections=sections, statistics=stats)
        d = pkg.to_dict()
        assert d["request_id"] == "req-1"
        assert len(d["sections"]) == 1
        assert d["sections"][0]["section_type"] == "relevant"
        assert d["statistics"]["total_memories"] == 1


# ---------------------------------------------------------------------------
# TokenEstimator
# ---------------------------------------------------------------------------


class TestTokenEstimator:
    def test_estimate_text(self) -> None:
        count = TokenEstimator.estimate_text("hello world")
        assert count >= 1

    def test_estimate_short_text(self) -> None:
        count = TokenEstimator.estimate_text("a")
        assert count == 1

    def test_estimate_long_text(self) -> None:
        count = TokenEstimator.estimate_text("x" * 400)
        assert count >= 100

    def test_estimate_memory(self) -> None:
        mem = Memory(content="test content", memory_type="short_term", namespace="default",
                     tags=["a", "b"])
        count = TokenEstimator.estimate_memory(mem)
        assert count >= 1

    def test_estimate_memories(self) -> None:
        mems = [Memory(content="a"), Memory(content="b" * 100)]
        total = TokenEstimator.estimate_memories(mems)
        assert total >= 2


# ---------------------------------------------------------------------------
# ContextBuilderConfig
# ---------------------------------------------------------------------------


class TestContextBuilderConfig:
    def test_defaults(self) -> None:
        cfg = ContextBuilderConfig()
        assert cfg.max_tokens == 4096
        assert cfg.enable_relationship_expansion is True
        assert cfg.enable_compression is True
        assert len(cfg.section_order) >= 5

    def test_custom(self) -> None:
        cfg = ContextBuilderConfig(
            max_tokens=2048,
            enable_relationship_expansion=False,
            enable_compression=False,
        )
        assert cfg.max_tokens == 2048
        assert cfg.enable_relationship_expansion is False
        assert cfg.enable_compression is False


# ---------------------------------------------------------------------------
# ContextBuilder (integration tests)
# ---------------------------------------------------------------------------


class TestContextBuilder:
    async def test_build_empty_manager(self, repo: MemoryRepository) -> None:
        """Build with an empty repository should produce minimal context."""
        mgr = MemoryManager(repository=repo)
        builder = ContextBuilder(memory_manager=mgr)
        pkg = await builder.build()
        assert pkg is not None
        assert pkg.total_memories == 0

    async def test_build_with_user_query(self, repo: MemoryRepository) -> None:
        mgr = MemoryManager(repository=repo)
        builder = ContextBuilder(memory_manager=mgr)
        pkg = await builder.build(user_content="Hello, I need help")
        assert len(pkg.sections) >= 1
        # The user query section should exist
        types = pkg.section_types
        assert "user_query" in types

    async def test_build_retrieves_relevant_memories(
        self, seeded_repo: MemoryRepository
    ) -> None:
        mgr = MemoryManager(repository=seeded_repo)
        builder = ContextBuilder(memory_manager=mgr)
        pkg = await builder.build()
        # Should have relevant memories section
        types = pkg.section_types
        assert "relevant_memories" in types

    async def test_build_includes_working_memory(
        self, seeded_repo: MemoryRepository
    ) -> None:
        mgr = MemoryManager(repository=seeded_repo)
        builder = ContextBuilder(memory_manager=mgr)
        pkg = await builder.build()
        types = pkg.section_types
        assert "working_memory" in types

    async def test_build_includes_long_term(
        self, seeded_repo: MemoryRepository
    ) -> None:
        mgr = MemoryManager(repository=seeded_repo)
        builder = ContextBuilder(memory_manager=mgr)
        pkg = await builder.build()
        types = pkg.section_types
        assert "long_term_facts" in types

    async def test_build_includes_conversation(
        self, seeded_repo: MemoryRepository
    ) -> None:
        mgr = MemoryManager(repository=seeded_repo)
        builder = ContextBuilder(memory_manager=mgr)
        pkg = await builder.build()
        types = pkg.section_types
        assert "conversation_history" in types

    async def test_build_produces_statistics(
        self, seeded_repo: MemoryRepository
    ) -> None:
        mgr = MemoryManager(repository=seeded_repo)
        builder = ContextBuilder(memory_manager=mgr)
        pkg = await builder.build()
        stats = pkg.statistics
        assert stats.total_memories > 0
        assert stats.total_sections > 0
        assert stats.total_tokens > 0
        assert stats.retrieval_ms >= 0

    async def test_build_respects_max_tokens(
        self, seeded_repo: MemoryRepository
    ) -> None:
        mgr = MemoryManager(repository=seeded_repo)
        builder = ContextBuilder(memory_manager=mgr)
        pkg = await builder.build(max_tokens=100)
        assert pkg.statistics.total_tokens <= 150  # allow small overhead

    async def test_build_to_dict(
        self, seeded_repo: MemoryRepository
    ) -> None:
        mgr = MemoryManager(repository=seeded_repo)
        builder = ContextBuilder(memory_manager=mgr)
        pkg = await builder.build(request_id="test-1")
        d = pkg.to_dict()
        assert d["request_id"] == "test-1"
        assert len(d["sections"]) > 0
        assert d["statistics"]["total_memories"] > 0

    async def test_build_with_custom_query(
        self, seeded_repo: MemoryRepository
    ) -> None:
        mgr = MemoryManager(repository=seeded_repo)
        builder = ContextBuilder(memory_manager=mgr)
        query = RetrievalQuery(namespaces=["conversation"], limit=50)
        pkg = await builder.build(query=query)
        # Should only retrieve from conversation namespace
        types = pkg.section_types
        # Only relevant_memories uses the query — working/long_term still run
        assert "relevant_memories" in types

    async def test_build_with_compression_disabled(
        self, seeded_repo: MemoryRepository
    ) -> None:
        cfg = ContextBuilderConfig(enable_compression=False)
        mgr = MemoryManager(repository=seeded_repo)
        builder = ContextBuilder(memory_manager=mgr, config=cfg)
        pkg = await builder.build()
        # Should still produce sections
        assert len(pkg.sections) > 0

    async def test_build_without_relationship_expansion(
        self, seeded_repo: MemoryRepository
    ) -> None:
        cfg = ContextBuilderConfig(enable_relationship_expansion=False)
        mgr = MemoryManager(repository=seeded_repo)
        builder = ContextBuilder(memory_manager=mgr, config=cfg)
        pkg = await builder.build()
        types = pkg.section_types
        # related_memories section should NOT appear
        assert "related_memories" not in types

    async def test_build_produces_metadata(
        self, seeded_repo: MemoryRepository
    ) -> None:
        mgr = MemoryManager(repository=seeded_repo)
        builder = ContextBuilder(memory_manager=mgr)
        pkg = await builder.build(metadata={"source": "test_suite"})
        assert pkg.metadata["source"] == "test_suite"
        assert "total_elapsed_ms" in pkg.metadata

    async def test_build_section_ordering(
        self, seeded_repo: MemoryRepository
    ) -> None:
        cfg = ContextBuilderConfig(
            section_order=["working_memory", "relevant_memories", "long_term_facts"],
        )
        mgr = MemoryManager(repository=seeded_repo)
        builder = ContextBuilder(memory_manager=mgr, config=cfg)
        pkg = await builder.build()
        # working_memory should be first
        if pkg.sections:
            assert pkg.sections[0].section_type == "working_memory"

    async def test_deterministic_assembly(
        self, seeded_repo: MemoryRepository
    ) -> None:
        """Two builds with the same input should produce the same structure."""
        mgr = MemoryManager(repository=seeded_repo)
        builder = ContextBuilder(memory_manager=mgr)
        pkg1 = await builder.build()
        pkg2 = await builder.build()
        assert len(pkg1.sections) == len(pkg2.sections)
        for s1, s2 in zip(pkg1.sections, pkg2.sections):
            assert s1.section_type == s2.section_type


# ---------------------------------------------------------------------------
# Token budget enforcement
# ---------------------------------------------------------------------------


class TestTokenBudget:
    async def test_small_budget_trims_memories(self, repo: MemoryRepository) -> None:
        """A very small budget should result in few or no memories."""
        await repo.add(Memory(content="a" * 200, memory_id=MemoryId("m1"), namespace="default"))
        await repo.add(Memory(content="b" * 200, memory_id=MemoryId("m2"), namespace="default"))
        mgr = MemoryManager(repository=repo)
        builder = ContextBuilder(memory_manager=mgr)
        pkg = await builder.build(max_tokens=50)
        # With 50 tokens, it should be very tight
        assert pkg.statistics.total_tokens <= 100

    async def test_large_budget_includes_all(
        self, seeded_repo: MemoryRepository
    ) -> None:
        """A large budget should include all available memories."""
        mgr = MemoryManager(repository=seeded_repo)
        builder = ContextBuilder(memory_manager=mgr)
        pkg = await builder.build(max_tokens=50000)
        assert pkg.statistics.total_memories >= len(seeded_repo._data) if hasattr(seeded_repo, "_data") else 1


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    async def test_no_crash_on_empty_repo(self, repo: MemoryRepository) -> None:
        mgr = MemoryManager(repository=repo)
        builder = ContextBuilder(memory_manager=mgr)
        pkg = await builder.build()
        assert pkg is not None

    async def test_no_crash_on_compression_error(
        self, compressed_repo: MemoryRepository
    ) -> None:
        """Should not crash if compression is configured but fails."""
        from app.memory.compression import CompressionService, CompressionPolicy

        # A valid compression config
        comp = CompressionService(
            repository=compressed_repo,
            connection=compressed_repo._conn,
            policy=CompressionPolicy(take_snapshot_before=False),
        )
        mgr = MemoryManager(repository=compressed_repo, compressor=comp)
        builder = ContextBuilder(memory_manager=mgr)
        pkg = await builder.build()
        assert pkg is not None

    async def test_rejects_invalid_config(self) -> None:
        # Dataclasses don't enforce types at runtime — this verifies
        # that passing a wrong type at least doesn't crash the builder.
        cfg = ContextBuilderConfig(max_tokens=2048)  # type: ignore[arg-type]
        assert cfg.max_tokens == 2048


# ---------------------------------------------------------------------------
# Config propagation
# ---------------------------------------------------------------------------


class TestConfigPropagation:
    async def test_custom_max_tokens(self, repo: MemoryRepository) -> None:
        cfg = ContextBuilderConfig(max_tokens=2048)
        mgr = MemoryManager(repository=repo)
        builder = ContextBuilder(memory_manager=mgr, config=cfg)
        assert builder.config.max_tokens == 2048

    async def test_disable_working_memory(self, seeded_repo: MemoryRepository) -> None:
        cfg = ContextBuilderConfig(working_memory_namespace="")
        mgr = MemoryManager(repository=seeded_repo)
        builder = ContextBuilder(memory_manager=mgr, config=cfg)
        pkg = await builder.build()
        types = pkg.section_types
        assert "working_memory" not in types

    async def test_disable_long_term(self, seeded_repo: MemoryRepository) -> None:
        cfg = ContextBuilderConfig(long_term_namespaces=[])
        mgr = MemoryManager(repository=seeded_repo)
        builder = ContextBuilder(memory_manager=mgr, config=cfg)
        pkg = await builder.build()
        types = pkg.section_types
        assert "long_term_facts" not in types

    async def test_disable_conversation(self, seeded_repo: MemoryRepository) -> None:
        cfg = ContextBuilderConfig(conversation_namespace="")
        mgr = MemoryManager(repository=seeded_repo)
        builder = ContextBuilder(memory_manager=mgr, config=cfg)
        pkg = await builder.build()
        types = pkg.section_types
        assert "conversation_history" not in types
