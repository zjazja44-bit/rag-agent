"""Local (offline) embedder backed by sentence-transformers.

Optional dependency — only imported when ``RAG__EMBEDDING__PROVIDER=local``.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..config import EmbeddingConfig
from .base import BaseEmbedder

_DEFAULT_LOCAL_MODEL = "BAAI/bge-small-zh-v1.5"


class LocalEmbedder(BaseEmbedder):
    def __init__(self, cfg: EmbeddingConfig) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "sentence-transformers is not installed. Run "
                "`pip install sentence-transformers`, or use "
                "RAG__EMBEDDING__PROVIDER=mock / openai_compat instead."
            ) from exc

        # If the configured model looks like a hosted embedding model name,
        # fall back to a sensible local default rather than trying to download
        # something that does not exist on the Hub.
        model_name = cfg.model
        if model_name.startswith("text-embedding"):
            model_name = _DEFAULT_LOCAL_MODEL
        self._model = SentenceTransformer(model_name)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        arr = self._model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True
        )
        return np.asarray(arr, dtype=np.float32)
