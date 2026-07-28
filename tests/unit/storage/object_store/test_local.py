"""Tests for LocalFileObjectStore.

Verifies:
- Upload and download binary data
- Upload returns metadata with checksum
- Exists returns correct state
- Delete removes files
- List keys with prefix
- Get metadata
- Streaming download
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import AsyncIterator

import pytest

from app.storage.object_store.local import LocalFileObjectStore


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def store() -> LocalFileObjectStore:
    """Object store backed by a temporary directory."""
    tmp = tempfile.mkdtemp()
    return LocalFileObjectStore(base_path=tmp)


# ---------------------------------------------------------------------------
# Upload / Download
# ---------------------------------------------------------------------------


class TestUploadDownload:
    async def test_upload_and_download(self, store: LocalFileObjectStore) -> None:
        data = b"hello world"
        meta = await store.upload("test/hello.txt", data, content_type="text/plain")
        assert meta.key == "test/hello.txt"
        assert meta.size == len(data)
        assert meta.content_type == "text/plain"
        assert meta.checksum_sha256 is not None

        downloaded = await store.download("test/hello.txt")
        assert downloaded == data

    async def test_download_nonexistent(self, store: LocalFileObjectStore) -> None:
        result = await store.download("nonexistent")
        assert result is None

    async def test_upload_with_metadata(self, store: LocalFileObjectStore) -> None:
        meta = await store.upload(
            "test/meta.txt",
            b"data",
            metadata={"author": "test", "version": 1},
        )
        assert meta.metadata["author"] == "test"


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


class TestStream:
    async def test_download_stream(self, store: LocalFileObjectStore) -> None:
        data = b"chunked data for streaming test"
        await store.upload("test/stream.txt", data)
        chunks: list[bytes] = []
        async for chunk in store.download_stream("test/stream.txt", chunk_size=4):
            chunks.append(chunk)
        assert b"".join(chunks) == data

    async def test_download_stream_nonexistent(self, store: LocalFileObjectStore) -> None:
        chunks: list[bytes] = []
        async for chunk in store.download_stream("nonexistent"):
            chunks.append(chunk)
        assert chunks == []


# ---------------------------------------------------------------------------
# Exists / Delete
# ---------------------------------------------------------------------------


class TestExistsDelete:
    async def test_exists_after_upload(self, store: LocalFileObjectStore) -> None:
        await store.upload("test/exists.txt", b"data")
        assert await store.exists("test/exists.txt") is True

    async def test_not_exists(self, store: LocalFileObjectStore) -> None:
        assert await store.exists("nothing") is False

    async def test_delete_removes(self, store: LocalFileObjectStore) -> None:
        await store.upload("test/delete_me.txt", b"data")
        assert await store.exists("test/delete_me.txt") is True
        deleted = await store.delete("test/delete_me.txt")
        assert deleted is True
        assert await store.exists("test/delete_me.txt") is False

    async def test_delete_nonexistent(self, store: LocalFileObjectStore) -> None:
        deleted = await store.delete("nothing")
        assert deleted is False


# ---------------------------------------------------------------------------
# List keys
# ---------------------------------------------------------------------------


class TestListKeys:
    async def test_list_keys(self, store: LocalFileObjectStore) -> None:
        await store.upload("prefix/a.txt", b"a")
        await store.upload("prefix/b.txt", b"b")
        await store.upload("other/c.txt", b"c")
        keys = await store.list_keys(prefix="prefix/")
        assert len(keys) == 2
        assert all(k.startswith("prefix/") for k in keys)

    async def test_list_keys_empty(self, store: LocalFileObjectStore) -> None:
        keys = await store.list_keys()
        assert keys == []


# ---------------------------------------------------------------------------
# Get metadata
# ---------------------------------------------------------------------------


class TestGetMetadata:
    async def test_get_metadata(self, store: LocalFileObjectStore) -> None:
        data = b"test data for metadata"
        await store.upload("test/meta_check.txt", data)
        meta = await store.get_metadata("test/meta_check.txt")
        assert meta is not None
        assert meta.size == len(data)
        assert meta.checksum_sha256 is not None

    async def test_get_metadata_nonexistent(self, store: LocalFileObjectStore) -> None:
        meta = await store.get_metadata("nonexistent")
        assert meta is None
