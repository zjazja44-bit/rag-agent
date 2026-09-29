"""Local embedder backed by fastembed (ONNX runtime, no PyTorch).

Optional dependency: ``pip install fastembed``. Downloads a small ONNX model on
first use, then runs fully offline. Much lighter than sentence-transformers
(no torch), and supports the Chinese model ``BAAI/bge-small-zh-v1.5``.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..config import EmbeddingConfig
from .base import BaseEmbedder

_DEFAULT_FASTEMBED_MODEL = "BAAI/bge-small-zh-v1.5"


class FastEmbedEmbedder(BaseEmbedder):
    def __init__(self, cfg: EmbeddingConfig) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError(
                "fastembed is not installed. Run `pip install fastembed`, or use "
                "RAG__EMBEDDING__PROVIDER=local / mock instead."
            ) from exc

        model_name = cfg.model
        if model_name.startswith("text-embedding"):
            model_name = _DEFAULT_FASTEMBED_MODEL
        self._model = TextEmbedding(model_name=model_name)

        # Probe once to learn the output dimension (also warms up the model).
        probe = next(iter(self._model.embed(["dimension probe"])))
        self._dim = int(len(probe))

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        vectors = np.asarray(list(self._model.embed(texts)), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms
