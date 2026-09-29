"""Hand-written BM25 (Okapi) with an inverted index.

BM25 scores a document for a query as a sum over query terms of

    idf(t) * (f(t,d) * (k1 + 1)) / (f(t,d) + k1 * (1 - b + b * |d| / avgdl))

with the standard probabilistic idf ``ln(1 + (N - df + 0.5) / (df + 0.5))``.
An inverted index (term -> posting list) means we only score documents that
actually contain a query term, instead of scanning the whole corpus.
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

from ..text_utils import tokenize
from ..types import Chunk
from .base import SearchHit


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = []
        self._id_index: dict[str, int] = {}
        self._tf: list[dict[str, int]] = []       # term freq per doc
        self._doc_len: list[int] = []
        self._postings: dict[str, list[int]] = defaultdict(list)
        self._avgdl: float = 0.0

    # -- mutation ----------------------------------------------------------
    def has(self, chunk_id: str) -> bool:
        return chunk_id in self._id_index

    def add(self, chunks: Sequence[Chunk]) -> None:
        for chunk in chunks:
            if chunk.id in self._id_index:
                continue
            doc_idx = len(self.chunks)
            self._id_index[chunk.id] = doc_idx
            self.chunks.append(chunk)

            tf: dict[str, int] = {}
            for tok in tokenize(chunk.text):
                tf[tok] = tf.get(tok, 0) + 1
            self._tf.append(tf)
            self._doc_len.append(sum(tf.values()))
            for term in tf:
                self._postings[term].append(doc_idx)

        self._avgdl = (sum(self._doc_len) / len(self._doc_len)) if self._doc_len else 0.0

    # -- query -------------------------------------------------------------
    def search(self, query: str, top_k: int) -> list[SearchHit]:
        n = len(self.chunks)
        if n == 0 or self._avgdl == 0:
            return []

        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenize(query)):
            postings = self._postings.get(term)
            if not postings:
                continue
            df = len(postings)
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for doc_idx in postings:
                f = self._tf[doc_idx][term]
                dl = self._doc_len[doc_idx]
                denom = f + self.k1 * (1 - self.b + self.b * dl / self._avgdl)
                scores[doc_idx] += idf * (f * (self.k1 + 1)) / denom

        if not scores:
            return []

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [
            SearchHit(chunk=self.chunks[i], score=float(s), retriever="bm25", rank=r)
            for r, (i, s) in enumerate(ranked)
        ]

    def __len__(self) -> int:
        return len(self.chunks)
