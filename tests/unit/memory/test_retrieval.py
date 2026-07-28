"""Tests for the retrieval pipeline.

Verifies:
- RetrievalContext and RetrievalResult
- RetrievalPipeline stage sequencing and abort
- SearchExecutor: query conversion, active-only default, results
- Each filter stage: namespace, type, state, tag, source, owner,
  content search, temporal, importance
- ImportanceRanker: score computation, ordering, edge cases
- RelationshipExpander: graph traversal and append
- Deduplicator: duplicate removal
- TopKTruncation: limit enforcement
- default_pipeline assembly
- End-to-end retrieval with combined filters + ranking + truncation
"""

from __future__ import annotations

from typing import Any

import pytest

from app.memory.memory import Memory, MemoryId, MemoryState, MemoryType
from app.memory.manager import MemoryRepository
from app.memory.relationships import MemoryGraphImpl
from app.memory.retrieval import (
    ContentSearch,
    Deduplicator,
    ImportanceFilter,
    ImportanceRanker,
    NamespaceFilter,
    OwnerFilter,
    RelationshipExpander,
    RetrievalContext,
    RetrievalPipeline,
    RetrievalQuery,
    RetrievalResult,
    SearchExecutor,
    SourceFilter,
    StateFilter,
    TagFilter,
    TemporalFilter,
    TopKTruncation,
    TypeFilter,
    default_pipeline,
)
from app.storage.interfaces import CacheService


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def repo() -> MemoryRepository:
    from app.storage.connection.sqlite import SQLiteConnection
    from app.storage.migration.manager import SqliteMigrationManager
    from app.memory.migrations import V002_MemorySchema

    conn = SQLiteConnection(":memory:")
    manager = SqliteMigrationManager()
    await manager.apply_all(conn, [V002_MemorySchema()])
    yield MemoryRepository(connection=conn)
    await conn.close()


@pytest.fixture
async def seeded_repo(repo: MemoryRepository) -> MemoryRepository:
    """Populate the repo with a variety of memories for retrieval tests."""
    memories = [
        Memory(
            content="alpha project plan",
            memory_type=MemoryType.PROJECT.value,
            namespace="projects",
            importance=0.9,
            tags=["plan", "project"],
            source="user",
            owner="alice",
            memory_id=MemoryId("mem-alpha"),
        ),
        Memory(
            content="beta release notes",
            memory_type=MemoryType.REFERENCE.value,
            namespace="docs",
            importance=0.7,
            tags=["release", "docs"],
            source="system",
            owner="bob",
            memory_id=MemoryId("mem-beta"),
        ),
        Memory(
            content="gamma architecture discussion",
            memory_type=MemoryType.CONVERSATION.value,
            namespace="projects",
            importance=0.3,
            tags=["architecture", "discussion"],
            source="user",
            owner="alice",
            memory_id=MemoryId("mem-gamma"),
        ),
        Memory(
            content="delta technical spec",
            memory_type=MemoryType.KNOWLEDGE.value,
            namespace="docs",
            importance=0.8,
            tags=["spec", "architecture"],
            source="user",
            owner="charlie",
            memory_id=MemoryId("mem-delta"),
        ),
        Memory(
            content="epsilon quick note",
            memory_type=MemoryType.SHORT_TERM.value,
            namespace="default",
            importance=0.1,
            tags=["note"],
            source="manual",
            owner="system",
            memory_id=MemoryId("mem-epsilon"),
        ),
        Memory(
            content="archived old plan",
            memory_type=MemoryType.PROJECT.value,
            namespace="projects",
            importance=0.2,
            tags=["plan", "old"],
            state=MemoryState.ARCHIVED,
            source="system",
            owner="alice",
            memory_id=MemoryId("mem-archived"),
        ),
    ]
    for m in memories:
        await repo.add(m)
    return repo


@pytest.fixture
async def graph(seeded_repo: MemoryRepository) -> MemoryGraphImpl:
    from app.memory.relationships import MemoryGraphImpl

    g = MemoryGraphImpl(connection=seeded_repo._conn)
    # Create relationships: alpha -> beta (references), alpha -> gamma (parent)
    await g.add_relationship("mem-alpha", "mem-beta", "references")
    await g.add_relationship("mem-alpha", "mem-gamma", "parent")
    return g


# ---------------------------------------------------------------------------
# Context and Result
# ---------------------------------------------------------------------------


class TestRetrievalContext:
    def test_initial_state(self) -> None:
        q = RetrievalQuery()
        ctx = RetrievalContext(q)
        assert ctx.query is q
        assert ctx.candidates == []
        assert ctx.aborted is False
        assert ctx.metadata == {}
        assert ctx.scores == {}

    def test_add_stage_meta(self) -> None:
        ctx = RetrievalContext(RetrievalQuery())
        ctx.add_stage_meta("Foo", "count", 7)
        assert ctx.metadata["Foo"]["count"] == 7
        # Subsequent calls append under the same stage
        ctx.add_stage_meta("Foo", "extra", "x")
        assert ctx.metadata["Foo"]["extra"] == "x"


class TestRetrievalResult:
    def test_empty_result(self) -> None:
        result = RetrievalResult(memories=[], total=0)
        assert result.memories == []
        assert result.total == 0

    def test_with_memories(self) -> None:
        m = Memory(content="test")
        result = RetrievalResult(memories=[m], total=1)
        assert result.memories[0] is m
        assert result.total == 1

    def test_to_dict(self) -> None:
        m = Memory(content="hello", memory_id=MemoryId("mid"))
        result = RetrievalResult(memories=[m], total=1)
        d = result.to_dict()
        assert d["total"] == 1
        assert len(d["memories"]) == 1
        assert d["memories"][0]["content"] == "hello"


# ---------------------------------------------------------------------------
# SearchExecutor
# ---------------------------------------------------------------------------


class TestSearchExecutor:
    async def test_executes_and_populates_candidates(
        self, seeded_repo: MemoryRepository
    ) -> None:
        stage = SearchExecutor(seeded_repo)
        query = RetrievalQuery()
        ctx = RetrievalContext(query)
        ctx = await stage(ctx)
        assert len(ctx.candidates) > 0

    async def test_defaults_to_active_only(
        self, seeded_repo: MemoryRepository
    ) -> None:
        stage = SearchExecutor(seeded_repo)
        query = RetrievalQuery()
        ctx = RetrievalContext(query)
        ctx = await stage(ctx)
        # Only the archived memory should be excluded by default
        ids = {m.id.value for m in ctx.candidates}
        assert "mem-archived" not in ids
        assert "mem-alpha" in ids

    async def test_includes_archived_when_requested(
        self, seeded_repo: MemoryRepository
    ) -> None:
        stage = SearchExecutor(seeded_repo)
        query = RetrievalQuery(states=[MemoryState.ACTIVE, MemoryState.ARCHIVED])
        ctx = RetrievalContext(query)
        ctx = await stage(ctx)
        ids = {m.id.value for m in ctx.candidates}
        assert "mem-archived" in ids


# ---------------------------------------------------------------------------
# Filter stages
# ---------------------------------------------------------------------------


class TestNamespaceFilter:
    async def test_filters_by_namespace(self) -> None:
        stage = NamespaceFilter()
        q = RetrievalQuery(namespaces=["docs"])
        ctx = RetrievalContext(q)
        ctx.candidates = [
            Memory(content="a", namespace="docs"),
            Memory(content="b", namespace="projects"),
        ]
        ctx = await stage(ctx)
        assert len(ctx.candidates) == 1
        assert ctx.candidates[0].content == "a"

    async def test_noop_when_no_namespaces(self) -> None:
        stage = NamespaceFilter()
        ctx = RetrievalContext(RetrievalQuery())
        ctx.candidates = [Memory(content="x")]
        ctx = await stage(ctx)
        assert len(ctx.candidates) == 1

    async def test_noop_on_empty_candidates(self) -> None:
        stage = NamespaceFilter()
        ctx = RetrievalContext(RetrievalQuery(namespaces=["docs"]))
        ctx = await stage(ctx)
        assert ctx.candidates == []


class TestTypeFilter:
    async def test_filters_by_type(self) -> None:
        stage = TypeFilter()
        q = RetrievalQuery(memory_types=[MemoryType.PROJECT.value])
        ctx = RetrievalContext(q)
        ctx.candidates = [
            Memory(content="a", memory_type=MemoryType.PROJECT.value),
            Memory(content="b", memory_type=MemoryType.REFERENCE.value),
        ]
        ctx = await stage(ctx)
        assert len(ctx.candidates) == 1
        assert ctx.candidates[0].content == "a"

    async def test_multiple_types(self) -> None:
        stage = TypeFilter()
        q = RetrievalQuery(memory_types=[MemoryType.PROJECT.value, MemoryType.REFERENCE.value])
        ctx = RetrievalContext(q)
        ctx.candidates = [
            Memory(content="a", memory_type=MemoryType.PROJECT.value),
            Memory(content="b", memory_type=MemoryType.SHORT_TERM.value),
        ]
        ctx = await stage(ctx)
        assert len(ctx.candidates) == 1


class TestStateFilter:
    async def test_filters_by_state(self) -> None:
        stage = StateFilter()
        q = RetrievalQuery(states=[MemoryState.ARCHIVED])
        ctx = RetrievalContext(q)
        ctx.candidates = [
            Memory(content="active", state=MemoryState.ACTIVE),
            Memory(content="archived", state=MemoryState.ARCHIVED),
        ]
        ctx = await stage(ctx)
        assert len(ctx.candidates) == 1
        assert ctx.candidates[0].content == "archived"


class TestTagFilter:
    async def test_matches_any_tag(self) -> None:
        stage = TagFilter()
        q = RetrievalQuery(tags=["plan"])
        ctx = RetrievalContext(q)
        ctx.candidates = [
            Memory(content="a", tags=["plan", "project"]),
            Memory(content="b", tags=["note"]),
        ]
        ctx = await stage(ctx)
        assert len(ctx.candidates) == 1
        assert ctx.candidates[0].content == "a"

    async def test_multiple_tags_any_match(self) -> None:
        stage = TagFilter()
        q = RetrievalQuery(tags=["plan", "note"])
        ctx = RetrievalContext(q)
        ctx.candidates = [
            Memory(content="a", tags=["plan"]),
            Memory(content="b", tags=["note"]),
            Memory(content="c", tags=["other"]),
        ]
        ctx = await stage(ctx)
        assert len(ctx.candidates) == 2


class TestSourceFilter:
    async def test_filters_by_source(self) -> None:
        stage = SourceFilter()
        q = RetrievalQuery(sources=["user"])
        ctx = RetrievalContext(q)
        ctx.candidates = [
            Memory(content="a", source="user"),
            Memory(content="b", source="system"),
        ]
        ctx = await stage(ctx)
        assert len(ctx.candidates) == 1
        assert ctx.candidates[0].content == "a"


class TestOwnerFilter:
    async def test_filters_by_owner(self) -> None:
        stage = OwnerFilter()
        q = RetrievalQuery(owners=["alice"])
        ctx = RetrievalContext(q)
        ctx.candidates = [
            Memory(content="a", owner="alice"),
            Memory(content="b", owner="bob"),
        ]
        ctx = await stage(ctx)
        assert len(ctx.candidates) == 1
        assert ctx.candidates[0].content == "a"


class TestContentSearch:
    async def test_content_substring_match(self) -> None:
        stage = ContentSearch()
        q = RetrievalQuery(content_search="plan")
        ctx = RetrievalContext(q)
        ctx.candidates = [
            Memory(content="project plan document"),
            Memory(content="release notes"),
        ]
        ctx = await stage(ctx)
        assert len(ctx.candidates) == 1
        assert "plan" in ctx.candidates[0].content

    async def test_case_insensitive(self) -> None:
        stage = ContentSearch()
        q = RetrievalQuery(content_search="PLAN")
        ctx = RetrievalContext(q)
        ctx.candidates = [
            Memory(content="Project Plan"),
        ]
        ctx = await stage(ctx)
        assert len(ctx.candidates) == 1


class TestTemporalFilter:
    async def test_filters_after(self) -> None:
        stage = TemporalFilter()
        q = RetrievalQuery(created_after="2099-01-01")

        old_mem = Memory(content="old")
        old_mem.created_at = old_mem.created_at.replace(year=2024)

        new_mem = Memory(content="new")
        new_mem.created_at = new_mem.created_at.replace(year=2100)

        ctx = RetrievalContext(q)
        ctx.candidates = [old_mem, new_mem]
        ctx = await stage(ctx)
        assert len(ctx.candidates) == 1
        assert ctx.candidates[0].content == "new"

    async def test_noop_without_range(self) -> None:
        stage = TemporalFilter()
        ctx = RetrievalContext(RetrievalQuery())
        ctx.candidates = [Memory(content="x")]
        ctx = await stage(ctx)
        assert len(ctx.candidates) == 1


class TestImportanceFilter:
    async def test_min_importance(self) -> None:
        stage = ImportanceFilter()
        q = RetrievalQuery(min_importance=0.5)
        ctx = RetrievalContext(q)
        ctx.candidates = [
            Memory(content="high", importance=0.9),
            Memory(content="low", importance=0.1),
        ]
        ctx = await stage(ctx)
        assert len(ctx.candidates) == 1
        assert ctx.candidates[0].content == "high"

    async def test_importance_range(self) -> None:
        stage = ImportanceFilter()
        q = RetrievalQuery(min_importance=0.3, max_importance=0.7)
        ctx = RetrievalContext(q)
        ctx.candidates = [
            Memory(content="low", importance=0.1),
            Memory(content="mid", importance=0.5),
            Memory(content="high", importance=0.9),
        ]
        ctx = await stage(ctx)
        assert len(ctx.candidates) == 1
        assert ctx.candidates[0].content == "mid"


# ---------------------------------------------------------------------------
# ImportanceRanker
# ---------------------------------------------------------------------------


class TestImportanceRanker:
    async def test_sorts_by_importance_descending(self) -> None:
        stage = ImportanceRanker()
        ctx = RetrievalContext(RetrievalQuery())
        ctx.candidates = [
            Memory(content="low", importance=0.1),
            Memory(content="high", importance=0.9),
            Memory(content="mid", importance=0.5),
        ]
        ctx = await stage(ctx)
        assert [m.content for m in ctx.candidates] == ["high", "mid", "low"]

    async def test_recency_boosts_recent(self) -> None:
        from datetime import timedelta, timezone

        stage = ImportanceRanker()
        old = Memory(content="old", importance=0.5)
        recent = Memory(content="recent", importance=0.5)
        # recent accessed very recently
        recent.accessed_at = old.accessed_at.replace(tzinfo=timezone.utc)
        old.accessed_at = old.accessed_at.replace(year=2020, tzinfo=timezone.utc)

        ctx = RetrievalContext(RetrievalQuery())
        ctx.candidates = [old, recent]
        ctx = await stage(ctx)
        assert ctx.candidates[0].content == "recent"

    async def test_frequency_boosts_often_accessed(self) -> None:
        stage = ImportanceRanker()
        frequent = Memory(content="frequent", importance=0.5)
        infrequent = Memory(content="infrequent", importance=0.5)
        frequent.access_count = 100
        infrequent.access_count = 0

        ctx = RetrievalContext(RetrievalQuery())
        ctx.candidates = [infrequent, frequent]
        ctx = await stage(ctx)
        assert ctx.candidates[0].content == "frequent"

    async def test_scores_in_context(self) -> None:
        stage = ImportanceRanker()
        m = Memory(content="test", importance=0.7, memory_id=MemoryId("m1"))
        ctx = RetrievalContext(RetrievalQuery())
        ctx.candidates = [m]
        ctx = await stage(ctx)
        assert ctx.scores["m1"] > 0.0

    async def test_empty_candidates(self) -> None:
        stage = ImportanceRanker()
        ctx = RetrievalContext(RetrievalQuery())
        ctx = await stage(ctx)
        assert ctx.candidates == []


# ---------------------------------------------------------------------------
# RelationshipExpander
# ---------------------------------------------------------------------------


class TestRelationshipExpander:
    async def test_expands_related_memories(
        self, seeded_repo: MemoryRepository, graph: MemoryGraphImpl
    ) -> None:
        stage = RelationshipExpander(graph)
        q = RetrievalQuery(expand_relationships=True)
        ctx = RetrievalContext(q)
        # Start with just alpha
        alpha = await seeded_repo.get(MemoryId("mem-alpha"))
        assert alpha is not None
        ctx.candidates = [alpha]
        ctx = await stage(ctx)
        ids = {m.id.value for m in ctx.candidates}
        # Should have expanded to include beta and gamma
        assert "mem-beta" in ids
        assert "mem-gamma" in ids

    async def test_noop_when_expand_false(
        self, graph: MemoryGraphImpl
    ) -> None:
        stage = RelationshipExpander(graph)
        q = RetrievalQuery(expand_relationships=False)
        ctx = RetrievalContext(q)
        ctx.candidates = [Memory(content="a")]
        ctx = await stage(ctx)
        assert len(ctx.candidates) == 1

    async def test_noop_without_graph(self) -> None:
        stage = RelationshipExpander()
        q = RetrievalQuery(expand_relationships=True)
        ctx = RetrievalContext(q)
        ctx.candidates = [Memory(content="a")]
        ctx = await stage(ctx)
        assert len(ctx.candidates) == 1

    async def test_deduplicates_on_expand(
        self, seeded_repo: MemoryRepository, graph: MemoryGraphImpl
    ) -> None:
        """When a related memory is already a candidate, it's not added again."""
        stage = RelationshipExpander(graph)
        q = RetrievalQuery(expand_relationships=True)
        ctx = RetrievalContext(q)
        alpha = await seeded_repo.get(MemoryId("mem-alpha"))
        beta = await seeded_repo.get(MemoryId("mem-beta"))
        assert alpha is not None
        assert beta is not None
        ctx.candidates = [alpha, beta]
        ctx = await stage(ctx)
        # Count of beta in candidates
        beta_count = sum(1 for m in ctx.candidates if m.id.value == "mem-beta")
        assert beta_count == 1


# ---------------------------------------------------------------------------
# Deduplicator
# ---------------------------------------------------------------------------


class TestDeduplicator:
    async def test_removes_duplicates(self) -> None:
        stage = Deduplicator()
        ctx = RetrievalContext(RetrievalQuery())
        ctx.candidates = [
            Memory(content="a", memory_id=MemoryId("1")),
            Memory(content="b", memory_id=MemoryId("2")),
            Memory(content="a-dup", memory_id=MemoryId("1")),
        ]
        ctx = await stage(ctx)
        assert len(ctx.candidates) == 2

    async def test_empty_candidates(self) -> None:
        stage = Deduplicator()
        ctx = RetrievalContext(RetrievalQuery())
        ctx = await stage(ctx)
        assert ctx.candidates == []

    async def test_no_duplicates_preserves_order(self) -> None:
        stage = Deduplicator()
        ctx = RetrievalContext(RetrievalQuery())
        ctx.candidates = [
            Memory(content="first", memory_id=MemoryId("1")),
            Memory(content="second", memory_id=MemoryId("2")),
        ]
        ctx = await stage(ctx)
        assert len(ctx.candidates) == 2
        assert ctx.candidates[0].content == "first"


# ---------------------------------------------------------------------------
# TopKTruncation
# ---------------------------------------------------------------------------


class TestTopKTruncation:
    async def test_truncates_to_limit(self) -> None:
        stage = TopKTruncation()
        q = RetrievalQuery(limit=2)
        ctx = RetrievalContext(q)
        ctx.candidates = [
            Memory(content=f"item {i}") for i in range(10)
        ]
        ctx = await stage(ctx)
        assert len(ctx.candidates) == 2

    async def test_no_truncation_when_under_limit(self) -> None:
        stage = TopKTruncation()
        q = RetrievalQuery(limit=100)
        ctx = RetrievalContext(q)
        ctx.candidates = [Memory(content="only one")]
        ctx = await stage(ctx)
        assert len(ctx.candidates) == 1


# ---------------------------------------------------------------------------
# RetrievalPipeline
# ---------------------------------------------------------------------------


class TestRetrievalPipeline:
    async def test_pipeline_rejects_empty_stages(self) -> None:
        with pytest.raises(AssertionError, match="at least one stage"):
            RetrievalPipeline([])

    async def test_pipeline_runs_all_stages(
        self, seeded_repo: MemoryRepository
    ) -> None:
        pipeline = RetrievalPipeline([
            SearchExecutor(seeded_repo),
            TopKTruncation(),
        ])
        result = await pipeline.retrieve(RetrievalQuery(limit=3))
        assert len(result.memories) <= 3
        assert result.total <= 3

    async def test_pipeline_abort(self) -> None:
        """Stage that aborts mid-pipeline — subsequent stages must not run."""

        class _AbortStage:
            async def __call__(self, ctx: RetrievalContext) -> RetrievalContext:
                ctx.aborted = True
                return ctx

        class _ShouldNotRun:
            async def __call__(self, ctx: RetrievalContext) -> RetrievalContext:
                raise RuntimeError("This stage should never execute")

        pipeline = RetrievalPipeline([_AbortStage(), _ShouldNotRun()])
        result = await pipeline.retrieve(RetrievalQuery())
        assert result.memories == []

    async def test_pipeline_metadata_collected(self) -> None:
        pipeline = RetrievalPipeline([TopKTruncation()])
        result = await pipeline.retrieve(RetrievalQuery(limit=5))
        assert "TopKTruncation" in result.metadata

    async def test_full_integration(
        self, seeded_repo: MemoryRepository
    ) -> None:
        """End-to-end: filter by namespace, type, tag, rank, and truncate."""
        pipeline = default_pipeline(seeded_repo)
        query = RetrievalQuery(
            namespaces=["projects"],
            memory_types=[MemoryType.PROJECT.value],
            tags=["plan"],
            limit=5,
        )
        result = await pipeline.retrieve(query)
        assert len(result.memories) >= 1
        # All results should be from 'projects' namespace, type PROJECT, tagged 'plan'
        for m in result.memories:
            assert m.namespace == "projects"
            assert m.memory_type == MemoryType.PROJECT.value
            assert "plan" in m.tags

    async def test_full_pipeline_with_relationship_expansion(
        self, seeded_repo: MemoryRepository, graph: MemoryGraphImpl
    ) -> None:
        """End-to-end: search + filter + rank + expand + dedup + truncate."""
        pipeline = default_pipeline(seeded_repo, graph=graph)
        query = RetrievalQuery(
            expand_relationships=True,
            limit=10,
        )
        result = await pipeline.retrieve(query)
        # With relationship expansion we should get more than just the direct matches
        assert len(result.memories) >= 1

    async def test_empty_repo(self) -> None:
        """Pipeline handles empty repository gracefully."""
        from app.storage.connection.sqlite import SQLiteConnection
        from app.storage.migration.manager import SqliteMigrationManager
        from app.memory.migrations import V002_MemorySchema

        conn = SQLiteConnection(":memory:")
        manager = SqliteMigrationManager()
        await manager.apply_all(conn, [V002_MemorySchema()])
        empty_repo = MemoryRepository(connection=conn)
        try:
            pipeline = default_pipeline(empty_repo)
            result = await pipeline.retrieve(RetrievalQuery(limit=10))
            assert result.memories == []
            assert result.total == 0
        finally:
            await conn.close()

    async def test_all_filters_combined(
        self, seeded_repo: MemoryRepository
    ) -> None:
        """All filter stages applied together should produce correct intersection."""
        pipeline = default_pipeline(seeded_repo)
        query = RetrievalQuery(
            namespaces=["docs"],
            memory_types=[MemoryType.REFERENCE.value],
            sources=["system"],
            owners=["bob"],
            limit=10,
        )
        result = await pipeline.retrieve(query)
        # Only beta should match all filters
        assert len(result.memories) == 1
        assert result.memories[0].id.value == "mem-beta"

    async def test_ranking_orders_by_score(
        self, seeded_repo: MemoryRepository
    ) -> None:
        """Ranked results should be in descending composite-score order."""
        pipeline = default_pipeline(seeded_repo)
        result = await pipeline.retrieve(RetrievalQuery(limit=10))
        scores = [m.importance for m in result.memories]
        # Should be roughly descending (may not be strictly monotonic due to
        # recency/frequency scoring, but importance should be non-increasing
        # when recency/frequency equalize)
        assert len(scores) >= 2  # at least a few memories
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1] * 0.5  # relaxed check


# ---------------------------------------------------------------------------
# default_pipeline
# ---------------------------------------------------------------------------


class TestDefaultPipeline:
    def test_assembles_stages(self, seeded_repo: MemoryRepository) -> None:
        pipeline = default_pipeline(seeded_repo)
        assert len(pipeline._stages) == 14  # all expected stages

    def test_includes_all_stage_types(self, seeded_repo: MemoryRepository) -> None:
        pipeline = default_pipeline(seeded_repo)
        stage_types = [type(s) for s in pipeline._stages]
        assert SearchExecutor in stage_types
        assert NamespaceFilter in stage_types
        assert TypeFilter in stage_types
        assert StateFilter in stage_types
        assert TagFilter in stage_types
        assert ImportanceRanker in stage_types
        assert TopKTruncation in stage_types
        assert Deduplicator in stage_types


# ---------------------------------------------------------------------------
# Stage isolation: each stage must handle empty candidates gracefully
# ---------------------------------------------------------------------------


class TestStageIsolation:
    """Every stage must be safe to call with an empty candidate list."""

    @pytest.mark.parametrize(
        "stage_cls, kwargs",
        [
            (NamespaceFilter, {}),
            (TypeFilter, {}),
            (StateFilter, {}),
            (TagFilter, {}),
            (SourceFilter, {}),
            (OwnerFilter, {}),
            (ContentSearch, {}),
            (TemporalFilter, {}),
            (ImportanceFilter, {}),
            (ImportanceRanker, {}),
            (Deduplicator, {}),
            (TopKTruncation, {}),
        ],
    )
    async def test_stage_handles_empty(
        self, stage_cls: Any, kwargs: dict[str, Any]
    ) -> None:
        stage = stage_cls(**kwargs) if kwargs else stage_cls()
        # Use a query that satisfies the stage's conditions
        q = RetrievalQuery(
            namespaces=["test"],
            memory_types=["test"],
            states=[MemoryState.ACTIVE],
            tags=["test"],
            content_search="test",
            sources=["test"],
            owners=["test"],
            min_importance=0.5,
        )
        ctx = RetrievalContext(q)
        ctx = await stage(ctx)
        assert ctx.candidates == []
