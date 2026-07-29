"""Knowledge layer error hierarchy."""

from __future__ import annotations

from typing import Any

from app.core.errors import AtlasError


class KnowledgeError(AtlasError):
    """Base class for all knowledge layer errors."""


class DuplicateDocumentError(KnowledgeError):
    """Raised when a document is registered under an already-used ID."""

    def __init__(
        self,
        name: str = "",
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        msg = f"Document {name!r} is already registered" if name else "Duplicate document"
        super().__init__(msg, code="KNOWLEDGE_DUPLICATE_DOCUMENT", details=details)


class DocumentNotFoundError(KnowledgeError):
    """Raised when a requested document is not found."""

    def __init__(
        self,
        name: str = "",
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        msg = f"Document {name!r} not found" if name else "Document not found"
        super().__init__(msg, code="KNOWLEDGE_DOCUMENT_NOT_FOUND", details=details)
