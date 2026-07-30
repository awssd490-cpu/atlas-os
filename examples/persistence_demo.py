#!/usr/bin/env python3
"""
Persistence demo — save, load, and update a knowledge base to/from JSON.

Demonstrates:
  - Saving a KnowledgeBase to a JSON file
  - Loading it back
  - Incremental update after adding a document
"""

import asyncio
import os
import tempfile

from app.rag.knowledge_base import KnowledgeBase
from app.rag.models import KnowledgeChunk, KnowledgeDocument
from app.rag.persistence import JsonPersistenceBackend, PersistenceConfig


def create_kb() -> KnowledgeBase:
    """Create a sample knowledge base with two documents."""
    kb = KnowledgeBase()

    doc1 = KnowledgeDocument(
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
    kb.register(doc1)

    doc2 = KnowledgeDocument(
        document_id="london",
        title="London",
        content="London is the capital of the UK.",
        chunks=(
            KnowledgeChunk(
                chunk_id="london:0", document_id="london",
                content="London is the capital of the UK.", index=0,
            ),
        ),
    )
    kb.register(doc2)

    return kb


async def main() -> None:
    print("=" * 60)
    print("Persistence Demo")
    print("=" * 60)

    # Use a temporary file for the snapshot
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        snapshot_path = f.name

    try:
        # Step 1: Create and save
        kb = create_kb()
        print(f"\n1. Created KB with {kb.count()} document(s)")

        backend = JsonPersistenceBackend(
            PersistenceConfig(overwrite=True, include_embeddings=False),
        )
        save_result = await backend.save(snapshot_path, kb)
        print(f"   Saved to {snapshot_path}")
        print(f"   Size: {save_result.metadata['size_bytes']} bytes")

        # Step 2: Load back
        load_result = await backend.load(snapshot_path)
        kb_loaded = load_result.metadata["knowledge_base"]
        print(f"\n2. Loaded KB with {kb_loaded.count()} document(s)")
        for doc in kb_loaded.list_documents():
            print(f"   - {doc.document_id}: {doc.title} ({len(doc.chunks)} chunk(s))")

        # Step 3: Add a document and update
        new_doc = KnowledgeDocument(
            document_id="tokyo",
            title="Tokyo",
            content="Tokyo is the capital of Japan.",
            chunks=(
                KnowledgeChunk(
                    chunk_id="tokyo:0", document_id="tokyo",
                    content="Tokyo is the capital of Japan.", index=0,
                ),
            ),
        )
        kb.register(new_doc)
        print(f"\n3. Added document '{new_doc.document_id}'")

        update_result = await backend.update(snapshot_path, kb)
        changes = update_result.metadata["changes"]
        print(f"   Update changes: {changes}")

        # Step 4: Verify by loading again
        final = await backend.load(snapshot_path)
        final_kb = final.metadata["knowledge_base"]
        print(f"\n4. Final KB has {final_kb.count()} document(s)")
        for doc_name in ["paris", "london", "tokyo"]:
            print(f"   - {doc_name}: {final_kb.get(doc_name) is not None}")

        # Step 5: Check stats
        stats = await backend.stats(snapshot_path)
        print(f"\n5. Snapshot stats: {stats.documents} docs, {stats.size_bytes} bytes")

    finally:
        os.unlink(snapshot_path)
        print(f"\nCleaned up temporary file: {snapshot_path}")


if __name__ == "__main__":
    asyncio.run(main())
