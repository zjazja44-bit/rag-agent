"""Ingestion pipeline: load -> split -> embed -> store."""
from __future__ import annotations

from typing import Sequence

from ..config import IngestConfig
from ..llm.base import BaseEmbedder
from ..logging_utils import get_logger
from ..store.base import BaseVectorStore
from ..types import Chunk, Document
from .loader import load_path
from .splitter import split_text

logger = get_logger(__name__)


def split_document(doc: Document, cfg: IngestConfig) -> list[Chunk]:
    texts = split_text(
        doc.text,
        chunk_size=cfg.chunk_size,
        overlap=cfg.chunk_overlap,
        min_chunk_size=cfg.min_chunk_size,
        strategy=cfg.splitter,
    )
    return [
        Chunk(
            id=f"{doc.id}::{i:04d}",
            text=text,
            doc_id=doc.id,
            source=doc.source,
            index=i,
            metadata=dict(doc.metadata),
        )
        for i, text in enumerate(texts)
    ]


class IngestionPipeline:
    """Wires a text splitter, an embedder and a vector store together."""

    def __init__(
        self,
        embedder: BaseEmbedder,
        store: BaseVectorStore,
        cfg: IngestConfig,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.cfg = cfg

    def ingest_documents(self, documents: Sequence[Document]) -> int:
        chunks: list[Chunk] = []
        for doc in documents:
            chunks.extend(split_document(doc, self.cfg))
        if not chunks:
            logger.warning("No chunks produced from %d document(s)", len(documents))
            return 0

        vectors = self.embedder.embed([c.text for c in chunks])
        added = self.store.add(chunks, vectors)
        self.store.persist()
        logger.info(
            "Ingested %d new chunk(s) of %d produced, from %d document(s)",
            added,
            len(chunks),
            len(documents),
        )
        return added

    def ingest_path(self, path: str) -> int:
        documents = load_path(path)
        return self.ingest_documents(documents)
