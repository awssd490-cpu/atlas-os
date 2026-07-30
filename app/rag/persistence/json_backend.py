"""JsonPersistenceBackend — a concrete persistence backend using JSON.

Serialises a knowledge base (documents, chunks, embeddings, vectors)
to a JSON file.  All data is kept in plain JSON — no compression
(that is reserved for a future checkpoint).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from app.rag.knowledge_base import KnowledgeBase
from app.rag.persistence.base import PersistenceBackend
from app.rag.persistence.config import PersistenceConfig
from app.rag.persistence.errors import (
    PersistenceError,
)
from app.rag.persistence.models import PersistenceResult, PersistenceStats


class JsonPersistenceBackend(PersistenceBackend):
    """Persistence backend that serialises a ``KnowledgeBase`` to JSON.

    The saved file is a plain JSON document with a deterministic key
    order::

        {
            "version": 1,
            "documents": [...],
            "chunks": [...],
            "embeddings": [...],
            "vectors": [...],
            "metadata": { ... }
        }

    Usage::

        backend = JsonPersistenceBackend()
        result = await backend.save("/path/to/snapshot.json", knowledge_base)
    """

    CURRENT_VERSION: int = 1

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    async def save(
        self,
        path: str,
        data: object,
        **kwargs: object,
    ) -> PersistenceResult:
        """Persist a knowledge base to a JSON file.

        Args:
            path: Target file path ending in ``.json``.
            data: A ``KnowledgeBase`` instance to serialise.
            **kwargs: Unused.

        Returns:
            A ``PersistenceResult`` with success status, file path,
            byte count, and statistics.

        Raises:
            PersistenceError: If the target path already exists and
                ``overwrite`` is ``False``, or on write failures.
        """
        if not isinstance(data, KnowledgeBase):
            raise PersistenceError(
                "JsonPersistenceBackend.save() requires a KnowledgeBase instance",
                details={"received_type": type(data).__name__},
            )

        kb: KnowledgeBase = data
        config = self._config
        start = time.monotonic()

        # --- Overwrite protection ---
        target = Path(path)
        if not config.overwrite and target.exists():
            raise PersistenceError(
                f"Target path already exists: {path}",
                code="PERSISTENCE_TARGET_EXISTS",
                details={"path": path, "overwrite": False},
            )

        # --- Serialise documents ---
        documents: list[dict[str, Any]] = []
        chunks: list[dict[str, Any]] = []
        for doc in sorted(kb.list_documents(), key=lambda d: d.document_id):
            documents.append(doc.to_dict())

            for chunk in sorted(doc.chunks, key=lambda c: c.index):
                chunks.append(self._serialise_chunk(chunk))

        # --- Serialise embeddings ---
        embeddings: list[dict[str, Any]] = []
        if config.include_embeddings:
            for chunk in sorted(kb.list_chunks(), key=lambda c: c.chunk_id):
                vec = kb.get_embedding(chunk.chunk_id)
                if vec is not None:
                    embeddings.append({
                        "chunk_id": chunk.chunk_id,
                        "vector": list(vec.vector),
                        "dimensions": vec.dimensions,
                        "provider": vec.provider,
                        "metadata": dict(vec.metadata),
                    })

        # --- Serialise vectors ---
        vectors: list[dict[str, Any]] = []
        if config.include_vectors:
            vs = kb.vector_store
            if vs is not None:
                # MemoryVectorStore stores its data in _vectors,
                # a dict[str, tuple[float, ...]].
                raw_vectors: dict[str, tuple[float, ...]] = {}
                if hasattr(vs, "_vectors"):
                    raw_vectors = vs._vectors  # type: ignore[attr-defined]

                for chunk_id in sorted(raw_vectors):
                    vectors.append({
                        "chunk_id": chunk_id,
                        "vector": list(raw_vectors[chunk_id]),
                    })

        # --- Build payload ---
        payload: dict[str, Any] = {
            "version": self.CURRENT_VERSION,
            "documents": documents,
            "chunks": chunks,
        }

        if config.include_embeddings:
            payload["embeddings"] = embeddings
        if config.include_vectors:
            payload["vectors"] = vectors

        payload["metadata"] = {
            "saved_at": time.time(),
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "embedding_count": len(embeddings),
            "vector_count": len(vectors),
        }

        # --- Write file ---
        try:
            json_bytes = json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise PersistenceError(
                f"Failed to serialise knowledge base: {exc}",
                details={"path": path},
            ) from exc

        try:
            target.write_bytes(json_bytes)
        except OSError as exc:
            raise PersistenceError(
                f"Failed to write file: {exc}",
                details={"path": path},
            ) from exc

        elapsed = time.monotonic() - start
        file_size = len(json_bytes)

        return PersistenceResult(
            success=True,
            metadata={
                "path": path,
                "size_bytes": file_size,
                "elapsed_time": round(elapsed, 4),
                "documents": len(documents),
                "chunks": len(chunks),
                "embeddings": len(embeddings),
                "vectors": len(vectors),
            },
        )

    # ------------------------------------------------------------------
    # Stub methods for ABC compliance
    # ------------------------------------------------------------------

    async def load(
        self,
        path: str,
        **kwargs: object,
    ) -> PersistenceResult:
        """Load is not yet implemented."""
        raise PersistenceError(
            "JsonPersistenceBackend.load() is not implemented",
            code="PERSISTENCE_LOAD_NOT_IMPLEMENTED",
        )

    async def exists(
        self,
        path: str,
        **kwargs: object,
    ) -> bool:
        """Check whether a JSON snapshot exists at *path*."""
        return Path(path).exists()

    async def delete(
        self,
        path: str,
        **kwargs: object,
    ) -> PersistenceResult:
        """Delete a JSON snapshot at *path*."""
        target = Path(path)
        if not target.exists():
            return PersistenceResult(
                success=False,
                metadata={"path": path, "reason": "not_found"},
            )

        try:
            os.remove(path)
        except OSError as exc:
            raise PersistenceError(
                f"Failed to delete file: {exc}",
                details={"path": path},
            ) from exc

        return PersistenceResult(
            success=True,
            metadata={"path": path},
        )

    async def stats(
        self,
        path: str,
        **kwargs: object,
    ) -> PersistenceStats:
        """Return statistics about a JSON snapshot at *path*.

        The file must have been produced by this backend.
        """
        target = Path(path)
        if not target.exists():
            raise PersistenceError(
                f"Path does not exist: {path}",
                code="PERSISTENCE_PATH_NOT_FOUND",
                details={"path": path},
            )

        try:
            payload = json.loads(target.read_bytes())
        except (json.JSONDecodeError, OSError) as exc:
            raise PersistenceError(
                f"Failed to read snapshot: {exc}",
                details={"path": path},
            ) from exc

        meta = payload.get("metadata", {})
        return PersistenceStats(
            documents=meta.get("document_count", 0),
            chunks=meta.get("chunk_count", 0),
            embeddings=meta.get("embedding_count", 0),
            vectors=meta.get("vector_count", 0),
            size_bytes=target.stat().st_size,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _serialise_chunk(chunk: object) -> dict[str, Any]:
        """Serialise a KnowledgeChunk to a plain dict."""
        from app.rag.models import KnowledgeChunk

        c = chunk  # type: KnowledgeChunk
        result: dict[str, Any] = {
            "chunk_id": c.chunk_id,
            "document_id": c.document_id,
            "content": c.content,
            "index": c.index,
            "metadata": {
                "source": c.metadata.source,
                "author": c.metadata.author,
                "version": c.metadata.version,
                "tags": list(c.metadata.tags),
                "category": c.metadata.category,
                "language": c.metadata.language,
            },
        }
        return result
