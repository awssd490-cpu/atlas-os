"""JsonPersistenceBackend — a concrete persistence backend using JSON.

Serialises and deserialises a knowledge base (documents, chunks,
embeddings, vectors) to and from a JSON file.  All data is kept in
plain JSON — no compression (that is reserved for a future
checkpoint).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from app.rag.knowledge_base import KnowledgeBase
from app.rag.models import KnowledgeChunk, KnowledgeDocument, KnowledgeMetadata
from app.rag.persistence.base import PersistenceBackend
from app.rag.persistence.config import PersistenceConfig
from app.rag.persistence.errors import (
    PersistenceError,
)
from app.rag.persistence.models import PersistenceResult, PersistenceStats


class JsonPersistenceBackend(PersistenceBackend):
    """Persistence backend that serialises/deserialises a
    ``KnowledgeBase`` to/from JSON.

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
        result = await backend.load("/path/to/snapshot.json")
    """

    CURRENT_VERSION: int = 1
    SUPPORTED_VERSIONS: tuple[int, ...] = (1,)

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

        # --- Serialise documents (include full content & metadata for round-trip) ---
        documents: list[dict[str, Any]] = []
        chunks: list[dict[str, Any]] = []
        for doc in sorted(kb.list_documents(), key=lambda d: d.document_id):
            doc_entry = doc.to_dict()
            doc_entry["content"] = doc.content
            doc_entry["metadata"] = self._serialise_metadata(doc.metadata)
            documents.append(doc_entry)

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
    # Load
    # ------------------------------------------------------------------

    async def load(
        self,
        path: str,
        **kwargs: object,
    ) -> PersistenceResult:
        """Load a knowledge base from a JSON file.

        Args:
            path: Source file path (``.json``) produced by
                :meth:`save`.
            **kwargs: Unused.

        Returns:
            A ``PersistenceResult`` with ``success=True`` and the
            reconstructed ``KnowledgeBase`` stored in ``metadata``
            under the key ``"knowledge_base"``, plus statistics.

        Raises:
            PersistenceError: If the file does not exist, contains
                invalid JSON, has an unsupported version, is missing
                required fields, or has corrupted data.
        """
        target = Path(path)

        # --- File existence ---
        if not target.exists():
            raise PersistenceError(
                f"File does not exist: {path}",
                code="PERSISTENCE_PATH_NOT_FOUND",
                details={"path": path},
            )

        # --- Parse JSON ---
        try:
            raw = target.read_bytes()
            payload = json.loads(raw)
        except (json.JSONDecodeError, OSError) as exc:
            raise PersistenceError(
                f"Failed to parse JSON snapshot: {exc}",
                details={"path": path},
            ) from exc

        if not isinstance(payload, dict):
            raise PersistenceError(
                "JSON root must be an object",
                details={"path": path, "received_type": type(payload).__name__},
            )

        # --- Validate version ---
        version = payload.get("version")
        if version is None:
            raise PersistenceError(
                "Missing required field: version",
                details={"path": path},
            )
        if not isinstance(version, int) or version not in self.SUPPORTED_VERSIONS:
            raise PersistenceError(
                f"Unsupported snapshot version: {version}. "
                f"Supported versions: {self.SUPPORTED_VERSIONS}",
                details={"path": path, "version": version, "supported": list(self.SUPPORTED_VERSIONS)},
            )

        # --- Validate required fields ---
        if "documents" not in payload:
            raise PersistenceError(
                "Missing required field: documents",
                details={"path": path, "version": version},
            )
        if "chunks" not in payload:
            raise PersistenceError(
                "Missing required field: chunks",
                details={"path": path, "version": version},
            )
        if not isinstance(payload["documents"], list):
            raise PersistenceError(
                "Field 'documents' must be a list",
                details={"path": path},
            )
        if not isinstance(payload["chunks"], list):
            raise PersistenceError(
                "Field 'chunks' must be a list",
                details={"path": path},
            )

        doc_list: list[dict[str, Any]] = payload["documents"]
        chunk_list: list[dict[str, Any]] = payload["chunks"]
        emb_list: list[dict[str, Any]] = payload.get("embeddings") or []
        vec_list: list[dict[str, Any]] = payload.get("vectors") or []

        # --- Validate and build chunk index ---
        chunk_map: dict[str, dict[str, Any]] = {}
        chunk_doc_ids: set[str] = set()
        seen_chunk_ids: set[str] = set()

        for i, c in enumerate(chunk_list):
            if not isinstance(c, dict):
                raise PersistenceError(
                    f"Chunk at index {i} must be an object",
                    details={"path": path, "index": i},
                )
            cid = c.get("chunk_id")
            if not cid or not isinstance(cid, str):
                raise PersistenceError(
                    f"Chunk at index {i} is missing a valid 'chunk_id'",
                    details={"path": path, "index": i},
                )
            if cid in seen_chunk_ids:
                raise PersistenceError(
                    f"Duplicate chunk_id: {cid!r}",
                    details={"path": path, "chunk_id": cid, "index": i},
                )
            seen_chunk_ids.add(cid)

            did = c.get("document_id", "")
            if not did:
                raise PersistenceError(
                    f"Chunk {cid!r} is missing a valid 'document_id'",
                    details={"path": path, "chunk_id": cid},
                )
            chunk_doc_ids.add(did)
            chunk_map[cid] = c

        # --- Reconstruct KnowledgeBase ---
        kb = KnowledgeBase()

        # Build documents with their chunks
        seen_doc_ids: set[str] = set()
        reconstructed_docs: list[KnowledgeDocument] = []

        for i, d in enumerate(doc_list):
            if not isinstance(d, dict):
                raise PersistenceError(
                    f"Document at index {i} must be an object",
                    details={"path": path, "index": i},
                )
            did = d.get("document_id")
            if not did or not isinstance(did, str):
                raise PersistenceError(
                    f"Document at index {i} is missing a valid 'document_id'",
                    details={"path": path, "index": i},
                )
            if did in seen_doc_ids:
                raise PersistenceError(
                    f"Duplicate document_id: {did!r}",
                    details={"path": path, "document_id": did, "index": i},
                )
            seen_doc_ids.add(did)

            title = d.get("title", "")
            content = d.get("content", "")

            # Reconstruct metadata
            meta = self._deserialise_metadata(d.get("metadata"))

            # Collect chunks belonging to this document
            doc_chunks: list[KnowledgeChunk] = []
            doc_chunk_ids: set[str] = set()
            for cid in sorted(chunk_map):
                c = chunk_map[cid]
                if c.get("document_id") == did:
                    if cid in doc_chunk_ids:
                        # Already added — skip (safety)
                        continue
                    doc_chunk_ids.add(cid)
                    doc_chunks.append(self._deserialise_chunk(c))

            # Sort by index for deterministic order
            doc_chunks.sort(key=lambda c: c.index)

            doc = KnowledgeDocument(
                document_id=did,
                title=title,
                content=content,
                chunks=tuple(doc_chunks),
                metadata=meta,
            )
            reconstructed_docs.append(doc)

        # Register all documents in the knowledge base
        for doc in reconstructed_docs:
            # Use the internal store method to bypass duplicate checks
            # since we've already validated uniqueness.
            kb._store_document(doc)

        # --- Restore embeddings ---
        if emb_list:
            from app.rag.embeddings.models import EmbeddingVector

            for i, e in enumerate(emb_list):
                if not isinstance(e, dict):
                    continue
                cid = e.get("chunk_id")
                if not cid or cid not in chunk_map:
                    raise PersistenceError(
                        f"Embedding at index {i} references unknown chunk: {cid!r}",
                        details={"path": path, "chunk_id": cid, "index": i},
                    )
                vector = tuple(e.get("vector", []))
                dimensions = e.get("dimensions", len(vector))
                provider = e.get("provider", "")
                emb_meta = e.get("metadata", {})

                kb._embeddings[cid] = EmbeddingVector(
                    vector=vector,
                    dimensions=dimensions,
                    provider=provider,
                    metadata=dict(emb_meta),
                )

        # --- Restore vector store ---
        if vec_list:
            from app.rag.vectorstore import MemoryVectorStore

            vs = MemoryVectorStore()
            for i, v in enumerate(vec_list):
                if not isinstance(v, dict):
                    continue
                cid = v.get("chunk_id")
                if not cid or cid not in chunk_map:
                    raise PersistenceError(
                        f"Vector at index {i} references unknown chunk: {cid!r}",
                        details={"path": path, "chunk_id": cid, "index": i},
                    )
                vector = tuple(v.get("vector", []))
                vs.add(cid, vector)

            # Wire the reconstructed vector store onto the KB
            kb._vector_store = vs  # type: ignore[attr-defined]

        file_size = target.stat().st_size

        return PersistenceResult(
            success=True,
            metadata={
                "path": path,
                "knowledge_base": kb,
                "size_bytes": file_size,
                "documents": len(reconstructed_docs),
                "chunks": len(chunk_map),
                "embeddings": len(emb_list),
                "vectors": len(vec_list),
            },
        )

    # ------------------------------------------------------------------
    # Exists / Delete / Stats
    # ------------------------------------------------------------------

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
    # Internal helpers — serialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _serialise_metadata(meta: KnowledgeMetadata) -> dict[str, Any]:
        """Serialise a KnowledgeMetadata to a plain dict."""
        return {
            "source": meta.source,
            "author": meta.author,
            "version": meta.version,
            "tags": list(meta.tags),
            "category": meta.category,
            "language": meta.language,
        }

    @staticmethod
    def _serialise_chunk(chunk: object) -> dict[str, Any]:
        """Serialise a KnowledgeChunk to a plain dict."""
        c = chunk  # type: KnowledgeChunk
        return {
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

    # ------------------------------------------------------------------
    # Internal helpers — deserialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _deserialise_metadata(
        data: dict[str, Any] | None,
    ) -> KnowledgeMetadata:
        """Deserialise a plain dict back to a KnowledgeMetadata."""
        if not data:
            return KnowledgeMetadata()
        return KnowledgeMetadata(
            source=data.get("source", ""),
            author=data.get("author", ""),
            version=data.get("version", "1.0"),
            tags=tuple(data.get("tags", [])),
            category=data.get("category", ""),
            language=data.get("language", ""),
        )

    @staticmethod
    def _deserialise_chunk(data: dict[str, Any]) -> KnowledgeChunk:
        """Deserialise a plain dict back to a KnowledgeChunk."""
        meta = JsonPersistenceBackend._deserialise_metadata(data.get("metadata"))
        return KnowledgeChunk(
            chunk_id=data.get("chunk_id", ""),
            document_id=data.get("document_id", ""),
            content=data.get("content", ""),
            index=data.get("index", 0),
            metadata=meta,
        )
