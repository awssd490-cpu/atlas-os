"""
Search orchestration for the RAG app.
"""

from app.rag.pipeline import PipelineResult


async def execute_search(pipeline, query: str, top_k: int = 5) -> PipelineResult:
    """Execute a search and return a formatted result."""
    result = await pipeline.search(query, max_chunks=top_k)
    return result
