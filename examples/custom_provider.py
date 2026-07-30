#!/usr/bin/env python3
"""
Custom provider example.

Demonstrates:
  - Implementing a custom EmbeddingProvider
  - Implementing a custom Reranker
  - Using them in a KnowledgeBase with hybrid retrieval
"""

import asyncio
import math
from collections.abc import Sequence
from typing import Any

from app.rag.embeddings import EmbeddingProvider, EmbeddingConfig
from app.rag.embeddings.models import EmbeddingResult, EmbeddingVector
from app.rag.knowledge_base import KnowledgeBase
from app.rag.models import KnowledgeChunk, KnowledgeDocument, KnowledgeQuery
from app.rag.rerank import Reranker
from app.rag.rerank.models import RerankResponse, RerankedResult
from app.rag.retriever import KnowledgeRetriever


# ---------------------------------------------------------------------------
# Custom embedding provider
# ---------------------------------------------------------------------------

class SimpleEmbeddingProvider(EmbeddingProvider):
    """A minimal embedding provider that creates simple hash-based vectors.

    This is NOT suitable for production — it exists to demonstrate the
    EmbeddingProvider interface.
    """

    def __init__(self, config: EmbeddingConfig) -> None:
        super().__init__(config)

    @property
    def name(self) -> str:
        return "simple"

    async def embed(self, text: str) -> EmbeddingResult:
        vector = self._make_vector(text)
        vec = EmbeddingVector(
            vector=vector,
            dimensions=len(vector),
            provider=self.name,
        )
        return EmbeddingResult(
            embeddings=(vec,),
            provider=self.name,
            config=self.config,
            total_texts=1,
        )

    async def embed_batch(self, texts: Sequence[str]) -> EmbeddingResult:
        vectors: list[EmbeddingVector] = []
        for text in texts:
            vector = self._make_vector(text)
            vectors.append(EmbeddingVector(
                vector=vector,
                dimensions=len(vector),
                provider=self.name,
            ))
        return EmbeddingResult(
            embeddings=tuple(vectors),
            provider=self.name,
            config=self.config,
            total_texts=len(vectors),
        )

    def _make_vector(self, text: str) -> tuple[float, ...]:
        """Create a simple deterministic vector from text."""
        dims = self.config.dimensions
        raw = []
        for dim in range(dims):
            # Use a hash of each character position
            val = sum(ord(c) * (dim + 1) for c in text) % 1000 / 1000.0
            raw.append(val)
        norm = math.sqrt(sum(v * v for v in raw))
        if norm > 0:
            raw = [v / norm for v in raw]
        return tuple(raw)


# ---------------------------------------------------------------------------
# Custom reranker
# ---------------------------------------------------------------------------

class LengthReranker(Reranker):
    """A reranker that boosts shorter chunks."""

    async def rerank(
        self,
        query: str,
        results: list[tuple[str, float]],
    ) -> RerankResponse:
        reranked: list[RerankedResult] = []
        for chunk_id, score in results:
            # Boosts chunks with "France" in the ID — just for demo
            length_bonus = 0.1 if "paris" in chunk_id.lower() else 0.0
            final_score = score + length_bonus
            reranked.append(RerankedResult(
                chunk_id=chunk_id,
                original_score=score,
                rerank_score=length_bonus,
                final_score=final_score,
            ))
        reranked.sort(key=lambda r: r.final_score, reverse=True)
        return RerankResponse(results=tuple(reranked))


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

async def main() -> None:
    print("=" * 60)
    print("Custom Provider Demo")
    print("=" * 60)

    # Step 1: Create custom provider
    config = EmbeddingConfig(
        provider_name="simple",
        dimensions=4,
        normalize_embeddings=True,
    )
    provider = SimpleEmbeddingProvider(config)
    print(f"\nEmbedding provider: {provider.name}")

    # Step 2: Create custom reranker
    reranker = LengthReranker()
    print(f"Reranker: LengthReranker")

    # Step 3: Create knowledge base with both
    from app.rag.vectorstore import MemoryVectorStore
    vs = MemoryVectorStore()

    kb = KnowledgeBase(
        embedding_provider=provider,
        vector_store=vs,
        reranker=reranker,
    )

    doc = KnowledgeDocument(
        document_id="paris",
        title="Paris",
        content="Paris is the capital of France.",
        chunks=(
            KnowledgeChunk(
                chunk_id="paris:0", document_id="paris",
                content="Paris is the capital of France.", index=0,
            ),
        ),
    )
    kb.register(doc)

    # Step 4: Embed and index
    for chunk in kb.list_chunks():
        result = await provider.embed(chunk.content)
        vec = result.embeddings[0]
        kb._embeddings[chunk.chunk_id] = vec
        vs.add(chunk.chunk_id, vec.vector)

    print(f"Indexed {vs.count()} vector(s)")

    # Step 5: Retrieve with reranking
    from app.rag.context import KnowledgeContextBuilder
    builder = KnowledgeContextBuilder(kb, KnowledgeRetriever(kb))
    context = await builder.build("capital of France")
    print(f"\nReranked context:")
    print(context.text)

    print(f"\nReranking enabled: {kb.reranker is not None}")


if __name__ == "__main__":
    asyncio.run(main())
