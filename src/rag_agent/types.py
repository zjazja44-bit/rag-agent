"""Core data models shared across ingestion, storage and retrieval."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A whole source document after loading."""

    id: str
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A retrievable slice of a document."""

    id: str
    text: str
    doc_id: str
    source: str
    index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "doc_id": self.doc_id,
            "source": self.source,
            "index": self.index,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Chunk":
        return cls(
            id=data["id"],
            text=data["text"],
            doc_id=data["doc_id"],
            source=data["source"],
            index=data["index"],
            metadata=data.get("metadata", {}),
        )
