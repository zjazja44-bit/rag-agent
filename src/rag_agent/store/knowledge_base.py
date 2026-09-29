"""A knowledge store pairing a dense vector index and a sparse BM25 index.

Keeping both indexes behind one object guarantees they stay consistent: the same
chunks are written to both, and BM25 is rebuilt from the persisted chunks on
load (so only the vectors need to be stored on disk).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from ..types import Chunk
from .bm25 import BM25Index
from .numpy_store import NumpyVectorStore


class KnowledgeStore:
    def __init__(self, store_dir: str | Path) -> None:
        self.dir = Path(store_dir)
        self.vector_store = NumpyVectorStore(self.dir)
        self.bm25 = BM25Index()

    @property
    def chunks(self) -> list[Chunk]:
        return self.vector_store.chunks

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> int:
        """Add only chunks not already present; returns the number added."""
        pairs = [
            (chunk, vec)
            for chunk, vec in zip(chunks, np.asarray(vectors, dtype=np.float32))
            if not self.vector_store.has(chunk.id)
        ]
        if not pairs:
            return 0
        new_chunks = [c for c, _ in pairs]
        new_vectors = np.vstack([v for _, v in pairs]).astype(np.float32)
        self.vector_store.add(new_chunks, new_vectors)
        self.bm25.add(new_chunks)
        return len(new_chunks)

    def persist(self) -> None:
        self.vector_store.persist()

    def load(self) -> bool:
        loaded = self.vector_store.load()
        if loaded:
            self.bm25 = BM25Index()
            self.bm25.add(self.vector_store.chunks)
        return loaded

    def __len__(self) -> int:
        return len(self.vector_store)
