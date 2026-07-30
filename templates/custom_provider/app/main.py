#!/usr/bin/env python3
"""
Demo: Custom providers in a KnowledgeBase with hybrid retrieval.
"""

import asyncio

from app.rag.embeddings import EmbeddingConfig
from app.rag.knowledge_base import KnowledgeBase
from app.rag.models import KnowledgeChunk, KnowledgeDocument
from app.rag.retriever import KnowledgeRetriever
from app.rag.context import KnowledgeContextBuilder
from app.rag.vectorstore import MemoryVectorStore

from app.embedding_provider import SimpleHashProvider
from app.reranker import LengthReranker


async def main() -> None:
    print("=" * 60)
    print("Custom Provider Demo")
    print("=" * 60)

    # Create custom providers
    config = EmbeddingConfig(provider_name="simple_hash", dimensions=4)
    provider = SimpleHashProvider(config)
    reranker = LengthReranker()
    store = MemoryVectorStore()

    print(f"  Embedding: {provider.name}")
    print(f"  Reranker:  {reranker.__class__.__name__}")

    # Create KB with custom providers
    knowledge_base = KnowledgeBase(
        embedding_provider=provider,
        vector_store=store,
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
    knowledge_base.register(doc)

    # Embed and index
    for chunk in knowledge_base.list_chunks():
        result = await provider.embed(chunk.content)
        vec = result.embeddings[0]
        knowledge_base._embeddings[chunk.chunk_id] = vec
        store.add(chunk.chunk_id, vec.vector)

    # Search with reranking
    builder = KnowledgeContextBuilder(knowledge_base, KnowledgeRetriever(knowledge_base))
    context = await builder.build("capital of France")
    print(f"\n  Result: {context.text}")


if __name__ == "__main__":
    asyncio.run(main())
