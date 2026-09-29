"""Hand-written dense vector store backed by a NumPy matrix.

Vectors are L2-normalised on the way in, so a query is a single matrix-vector
product and cosine similarity is just the dot product. Top-k uses
``argpartition`` (O(n)) rather than a full sort. State persists as
``chunks.json`` (metadata + text) plus ``vectors.npy`` (the raw matrix).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from ..types import Chunk
from .base import BaseVectorStore, SearchHit


class NumpyVectorStore(BaseVectorStore):
    def __init__(self, store_dir: str | Path) -> None:
        self.dir = Path(store_dir)
        self.chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None
        self._id_index: dict[str, int] = {}

    # -- mutation ----------------------------------------------------------
    def has(self, chunk_id: str) -> bool:
        return chunk_id in self._id_index

    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        vectors = _normalise(np.asarray(vectors, dtype=np.float32))
        if self._vectors is not None and vectors.shape[1] != self._vectors.shape[1]:
            raise ValueError(
                f"Embedding dimension mismatch: store has {self._vectors.shape[1]}, "
                f"new vectors have {vectors.shape[1]}. Re-ingest with the same "
                f"embedding provider (or use a fresh store directory)."
            )
        new_chunks: list[Chunk] = []
        new_rows: list[np.ndarray] = []
        for chunk, vec in zip(chunks, vectors):
            if chunk.id in self._id_index:
                continue  # idempotent: never duplicate the same chunk
            self._id_index[chunk.id] = len(self.chunks) + len(new_chunks)
            new_chunks.append(chunk)
            new_rows.append(vec)

        if not new_chunks:
            return
        block = np.vstack(new_rows).astype(np.float32)
        self._vectors = block if self._vectors is None else np.vstack([self._vectors, block])
        self.chunks.extend(new_chunks)

    # -- query -------------------------------------------------------------
    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchHit]:
        if not self.chunks or self._vectors is None:
            return []
        q = _normalise(np.asarray(query_vector, dtype=np.float32).reshape(1, -1))[0]
        if q.shape[0] != self._vectors.shape[1]:
            raise ValueError(
                f"Query embedding dimension {q.shape[0]} does not match the store's "
                f"{self._vectors.shape[1]}. Use the same embedding provider you ingested with."
            )
        sims = self._vectors @ q
        k = min(top_k, len(sims))
        if k <= 0:
            return []
        if k >= len(sims):
            order = np.argsort(-sims)
        else:
            part = np.argpartition(-sims, k - 1)[:k]
            order = part[np.argsort(-sims[part])]
        return [
            SearchHit(chunk=self.chunks[int(i)], score=float(sims[i]), retriever="vector", rank=r)
            for r, i in enumerate(order)
        ]

    # -- persistence -------------------------------------------------------
    def persist(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        payload = [c.to_dict() for c in self.chunks]
        (self.dir / "chunks.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if self._vectors is not None:
            np.save(self.dir / "vectors.npy", self._vectors)

    def load(self) -> bool:
        chunks_path = self.dir / "chunks.json"
        if not chunks_path.exists():
            return False
        data = json.loads(chunks_path.read_text(encoding="utf-8"))
        self.chunks = [Chunk.from_dict(d) for d in data]
        self._id_index = {c.id: i for i, c in enumerate(self.chunks)}
        vectors_path = self.dir / "vectors.npy"
        self._vectors = np.load(vectors_path) if vectors_path.exists() else None
        return True

    def __len__(self) -> int:
        return len(self.chunks)


def _normalise(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms
