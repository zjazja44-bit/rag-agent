"""Hybrid retriever: dense + sparse recall fused with Reciprocal Rank Fusion.

RRF combines ranked lists without needing their scores to share a scale:

    score(d) = weight * sum_over_lists( 1 / (k + rank(d)) )

which is exactly why it is robust across a cosine-similarity list and a BM25
list whose raw scores are not comparable.
"""
from __future__ import annotations

from typing import Sequence

from ..config import RetrievalConfig
from ..llm.base import BaseEmbedder
from ..store.base import SearchHit
from .reranker import build_reranker


class HybridRetriever:
    def __init__(self, store, embedder: BaseEmbedder, cfg: RetrievalConfig, reranker=None):
        self.store = store
        self.embedder = embedder
        self.cfg = cfg
        self.reranker = reranker

    @classmethod
    def from_config(cls, store, embedder, cfg: RetrievalConfig, llm=None) -> "HybridRetriever":
        return cls(store, embedder, cfg, reranker=build_reranker(cfg, llm))

    def retrieve(self, query: str, top_k: int | None = None) -> list[SearchHit]:
        top_k = top_k or self.cfg.top_k
        candidate_k = max(self.cfg.candidate_k, top_k)

        query_vector = self.embedder.embed_one(query)
        vector_hits = self.store.vector_store.search(query_vector, candidate_k)
        bm25_hits = self.store.bm25.search(query, candidate_k)

        fused = self.fuse(
            [vector_hits, bm25_hits],
            [self.cfg.vector_weight, self.cfg.bm25_weight],
        )

        if self.reranker is not None:
            return self.reranker.rerank(query, fused, top_n=top_k)
        return fused[:top_k]

    def fuse(
        self,
        hit_lists: Sequence[Sequence[SearchHit]],
        weights: Sequence[float],
    ) -> list[SearchHit]:
        scores: dict[str, float] = {}
        chunks: dict[str, object] = {}
        for hits, weight in zip(hit_lists, weights):
            for rank, hit in enumerate(hits):
                cid = hit.chunk.id
                chunks[cid] = hit.chunk
                scores[cid] = scores.get(cid, 0.0) + weight * (1.0 / (self.cfg.rrf_k + rank + 1))

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [
            SearchHit(chunk=chunks[cid], score=score, retriever="hybrid", rank=rank)
            for rank, (cid, score) in enumerate(ranked)
        ]
