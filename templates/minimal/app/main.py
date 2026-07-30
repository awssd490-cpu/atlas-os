#!/usr/bin/env python3
"""
Minimal Atlas RAG example — smallest runnable project.
"""

import asyncio
from app.rag.knowledge_base import KnowledgeBase
from app.rag.models import KnowledgeChunk, KnowledgeDocument, KnowledgeQuery
from app.rag.retriever import KnowledgeRetriever
from app.rag.context import KnowledgeContextBuilder


def create_kb() -> KnowledgeBase:
    kb = KnowledgeBase()
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
    return kb


async def main() -> None:
    kb = create_kb()
    print(f"Registered {kb.count()} document(s)")

    retriever = KnowledgeRetriever(kb)
    query = KnowledgeQuery(query="capital of France", max_results=5)
    result = await retriever.retrieve(query)
    print(f"Keyword search: found {result.total} chunk(s)")

    builder = KnowledgeContextBuilder(kb, retriever)
    context = await builder.build("capital of France")
    print(f"Context: {context.text}")


if __name__ == "__main__":
    asyncio.run(main())
