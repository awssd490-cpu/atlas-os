"""Local filesystem object store.

Stores binary blobs as files on disk with SHA-256 checksums and
metadata stored alongside.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from app.storage.interfaces import ObjectMetadata, ObjectStore


class LocalFileObjectStore(ObjectStore):
    """Object store backed by local filesystem.

    Files are stored at ``{base_path}/{key}``.  Metadata is stored at
    ``{base_path}/{key}.meta.json`` (JSON).  Checksums are SHA-256.

    Keys may contain path separators to organise objects into
    subdirectories.
    """

    def __init__(self, base_path: str) -> None:
        self._base = Path(base_path)

    async def upload(
        self,
        key: str,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
        metadata: dict[str, Any] | None = None,
    ) -> ObjectMetadata:
        """Store a binary object and its checksum/metadata.

        Returns the object's metadata after storage.
        """
        file_path = self._base / key
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the file
        file_path.write_bytes(data)

        # Compute checksum
        checksum = hashlib.sha256(data).hexdigest()

        # Write metadata
        meta = ObjectMetadata(
            key=key,
            size=len(data),
            content_type=content_type,
            checksum_sha256=checksum,
            metadata=metadata or {},
        )
        self._write_metadata(key, meta)

        return meta

    async def download(self, key: str) -> bytes | None:
        """Retrieve a binary object by key, or ``None``."""
        file_path = self._base / key
        if not file_path.exists():
            return None
        return file_path.read_bytes()

    async def download_stream(self, key: str, chunk_size: int = 65536) -> AsyncIterator[bytes]:
        """Stream a binary object in chunks."""
        file_path = self._base / key
        if not file_path.exists():
            return

        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                yield chunk

    async def delete(self, key: str) -> bool:
        """Delete an object and its metadata.  Returns ``True`` if it existed."""
        file_path = self._base / key
        meta_path = self._meta_path(key)

        existed = file_path.exists()

        if file_path.exists():
            file_path.unlink()
        if meta_path.exists():
            meta_path.unlink()

        # Remove empty parent directories
        self._cleanup_empty_parents(file_path.parent)

        return existed

    async def exists(self, key: str) -> bool:
        """Return ``True`` when the object file exists."""
        return (self._base / key).exists()

    async def list_keys(self, prefix: str = "") -> list[str]:
        """Return all object keys with an optional prefix.

        Excludes metadata files (``.meta.json``).
        """
        keys: list[str] = []
        base_len = len(str(self._base)) + 1  # +1 for separator

        for root, _dirs, files in os.walk(str(self._base)):
            for filename in files:
                if filename.endswith(".meta.json"):
                    continue
                full_path = os.path.join(root, filename)
                relative_key = full_path[base_len:].replace("\\", "/")
                if relative_key.startswith(prefix):
                    keys.append(relative_key)

        return sorted(keys)

    async def get_metadata(self, key: str) -> ObjectMetadata | None:
        """Return an object's metadata without downloading the file."""
        meta_path = self._meta_path(key)
        if not meta_path.exists():
            return None

        data = meta_path.read_text()
        raw = json.loads(data)
        return ObjectMetadata(
            key=raw.get("key", key),
            size=raw.get("size", 0),
            content_type=raw.get("content_type", "application/octet-stream"),
            checksum_sha256=raw.get("checksum_sha256"),
            created_at=raw.get("created_at"),
            metadata=raw.get("metadata", {}),
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _meta_path(self, key: str) -> Path:
        """Return the path to the metadata file for *key*."""
        return self._base / f"{key}.meta.json"

    def _write_metadata(self, key: str, meta: ObjectMetadata) -> None:
        """Write metadata as JSON adjacent to the object file."""
        meta_path = self._meta_path(key)
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(
            json.dumps(
                {
                    "key": meta.key,
                    "size": meta.size,
                    "content_type": meta.content_type,
                    "checksum_sha256": meta.checksum_sha256,
                    "created_at": meta.created_at,
                    "metadata": meta.metadata,
                },
                indent=2,
            )
        )

    def _cleanup_empty_parents(self, path: Path) -> None:
        """Remove empty parent directories up to the base path."""
        current = path
        while current != self._base:
            try:
                if any(current.iterdir()):
                    break  # not empty
                current.rmdir()
                current = current.parent
            except (OSError, PermissionError):
                break
