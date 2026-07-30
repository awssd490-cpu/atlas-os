#!/usr/bin/env python3
"""
Basic RAG pipeline example.

Demonstrates:
  - Creating a KnowledgeBase
  - Registering documents with pre-built chunks
  - Setting up a deterministic embedding provider and vector store
  - Keyword retrieval via KnowledgeRetriever
  - Context building via KnowledgeContextBuilder
"""

import asyncio

from app.rag.context import KnowledgeContextBuilder
from app.rag.knowledge_base import KnowledgeBase
from app.rag.models import KnowledgeChunk, KnowledgeDocument, KnowledgeQuery
from app.rag.retriever import KnowledgeRetriever


def create_sample_kb() -> KnowledgeBase:
    """Create and populate a knowledge base with sample documents."""
    kb = KnowledgeBase()

    # Document 1: Paris
    paris_doc = KnowledgeDocument(
        document_id="paris",
        title="Paris",
        content="Paris is the capital of France. The Eiffel Tower is a famous landmark.",
        chunks=(
            KnowledgeChunk(
                chunk_id="paris:0",
                document_id="paris",
                content="Paris is the capital of France.",
                index=0,
            ),
            KnowledgeChunk(
                chunk_id="paris:1",
                document_id="paris",
                content="The Eiffel Tower is a famous landmark.",
                index=1,
            ),
        ),
    )
    kb.register(paris_doc)

    # Document 2: London
    london_doc = KnowledgeDocument(
        document_id="london",
        title="London",
        content="London is the capital of the UK. The Thames flows through London.",
        chunks=(
            KnowledgeChunk(
                chunk_id="london:0",
                document_id="london",
                content="London is the capital of the UK.",
                index=0,
            ),
            KnowledgeChunk(
                chunk_id="london:1",
                document_id="london",
                content="The Thames flows through London.",
                index=1,
            ),
        ),
    )
    kb.register(london_doc)

    return kb


async def main() -> None:
    print("=" * 60)
    print("Basic RAG Pipeline Demo")
    print("=" * 60)

    # Step 1: Create knowledge base
    kb = create_sample_kb()
    print(f"\nRegistered {kb.count()} document(s)")
    print(f"Total chunks: {len(kb.list_chunks())}")

    # Step 2: Create a retriever and search
    retriever = KnowledgeRetriever(kb)

    query = KnowledgeQuery(query="capital of France", max_results=5)
    result = await retriever.retrieve(query)

    print(f"\nKeyword search for 'capital of France':")
    print(f"  Found {result.total} matching chunks")
    for chunk in result.chunks:
        print(f"  - {chunk.content}")

    # Step 3: Build context for provider injection
    builder = KnowledgeContextBuilder(kb, retriever)
    context = await builder.build("capital of France")
    print(f"\nFormatted context ({len(context.chunks)} chunks):")
    print(context.text)


if __name__ == "__main__":
    asyncio.run(main())
