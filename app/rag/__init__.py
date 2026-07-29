"""Knowledge Layer for Atlas RAG (Retrieval-Augmented Generation).

Provides a provider-independent knowledge retrieval system that
integrates with the Agent Runtime.  Knowledge and Memory are separate
systems — Memory stores conversations, Knowledge stores external data.

Architecture::

    Agent Runtime
        │
        ├─ Memory Retrieval  (conversations)
        ├─ Knowledge Retrieval  (external data)  ← NEW
        ├─ Context Builder (merges both)
        ▼
    Provider Runtime
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.rag.models import (
        KnowledgeDocument,
        KnowledgeChunk,
        KnowledgeSource,
        KnowledgeQuery,
        KnowledgeResult,
        KnowledgeContext,
        KnowledgeMetadata,
    )
    from app.rag.knowledge_base import KnowledgeBase
    from app.rag.retriever import KnowledgeRetriever
    from app.rag.context import KnowledgeContextBuilder

__all__ = [
    "KnowledgeDocument",
    "KnowledgeChunk",
    "KnowledgeSource",
    "KnowledgeQuery",
    "KnowledgeResult",
    "KnowledgeContext",
    "KnowledgeMetadata",
    "KnowledgeBase",
    "KnowledgeRetriever",
    "KnowledgeContextBuilder",
]
