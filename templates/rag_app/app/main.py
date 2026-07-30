#!/usr/bin/env python3
"""
Main entry point for the RAG application.
"""

import asyncio

from app.rag.chunking import ChunkingEngine, ChunkingConfig
from app.rag.knowledge_base import KnowledgeBase
from app.rag.pipeline import PipelineBuilder, PipelineConfig

from app.core.config import ConfigLoader
from app.core.log import AtlasLogger

from app.loader import load_documents


log = AtlasLogger("rag-app", level="INFO")


async def main() -> None:
    log.info("Starting RAG application")

    # Load configuration
    cfg = ConfigLoader.from_json("config.json")
    log.info("Configuration loaded", environment=cfg.environment)

    # Build pipeline
    pipeline = (
        PipelineBuilder()
        .loader(load_documents)
        .chunker(ChunkingEngine(ChunkingConfig(strategy="whole_document", min_chunk_size=1)))
        .knowledge_base(KnowledgeBase())
        .config(PipelineConfig(auto_embed=False, auto_index=False))
        .build()
    )

    # Ingest documents
    count = await pipeline.ingest("data")
    log.info("Documents ingested", count=count)

    # Search
    result = await pipeline.search("capital of France")
    print(f"Search results:\n{result.context}")
    log.info(
        "Search completed",
        query="capital of France",
        chunks_returned=result.metadata["chunks_returned"],
        mode=result.metadata["retrieval_mode"],
    )


if __name__ == "__main__":
    asyncio.run(main())
