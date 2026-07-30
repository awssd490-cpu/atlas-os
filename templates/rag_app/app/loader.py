"""
Custom document loader for the RAG app.
"""

from app.rag.models import KnowledgeChunk, KnowledgeDocument


def load_documents(path: str) -> list[KnowledgeDocument]:
    """Load documents from a path.

    This is a sample loader. Replace with your actual document loading logic.
    """
    documents = {
        "paris": {
            "title": "Paris",
            "content": "Paris is the capital of France. The Eiffel Tower is a famous landmark.",
        },
        "london": {
            "title": "London",
            "content": "London is the capital of the UK. The Thames flows through London.",
        },
        "tokyo": {
            "title": "Tokyo",
            "content": "Tokyo is the capital of Japan. It is a bustling metropolis.",
        },
    }

    result = []
    for doc_id, info in documents.items():
        doc = KnowledgeDocument(
            document_id=doc_id,
            title=info["title"],
            content=info["content"],
            chunks=(
                KnowledgeChunk(
                    chunk_id=f"{doc_id}:0",
                    document_id=doc_id,
                    content=info["content"],
                    index=0,
                ),
            ),
        )
        result.append(doc)
    return result
