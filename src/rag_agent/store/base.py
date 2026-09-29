"""Vector-store abstraction shared by the concrete stores and BM25 index."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..types import Chunk


@dataclass
class SearchHit:
    """One retrieved chunk plus its score and the retriever that produced it."""

    chunk: Chunk
    score: float
    retriever: str = ""
    rank: int = 0


class BaseVectorStore(ABC):
    """Dense vector store: add chunks, search by query vector, persist."""

    @abstractmethod
    def add(self, chunks: Sequence[Chunk], vectors: np.ndarray) -> None:
        ...

    @abstractmethod
    def search(self, query_vector: np.ndarray, top_k: int) -> list[SearchHit]:
        ...

    @abstractmethod
    def persist(self) -> None:
        ...

    @abstractmethod
    def load(self) -> bool:
        """Load persisted state; return True if anything was loaded."""

    def __len__(self) -> int:
        raise NotImplementedError
