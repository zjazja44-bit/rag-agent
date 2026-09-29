"""Hybrid retrieval, RRF fusion and reranking tests (no network)."""
from pathlib import Path

from rag_agent.config import IngestConfig, RetrievalConfig
from rag_agent.ingest.pipeline import IngestionPipeline
from rag_agent.llm.base import LLMResponse
from rag_agent.llm.mock import MockEmbedder, MockLLM
from rag_agent.retrieve.reranker import HeuristicReranker, LLMReranker
from rag_agent.retrieve.retriever import HybridRetriever
from rag_agent.store import KnowledgeStore
from rag_agent.store.base import SearchHit
from rag_agent.types import Chunk

DATA = Path(__file__).resolve().parent.parent / "data" / "docs"


def _chunk(cid: str) -> Chunk:
    return Chunk(id=cid, text=cid, doc_id="d", source="s", index=0)


def test_rrf_prefers_chunk_in_both_lists():
    retriever = HybridRetriever.__new__(HybridRetriever)  # no deps needed for fuse()
    retriever.cfg = RetrievalConfig(rrf_k=60, vector_weight=1.0, bm25_weight=1.0)

    vector_hits = [SearchHit(_chunk("a"), 0.9, "vector", 0), SearchHit(_chunk("b"), 0.8, "vector", 1)]
    bm25_hits = [SearchHit(_chunk("b"), 5.0, "bm25", 0), SearchHit(_chunk("c"), 4.0, "bm25", 1)]

    fused = retriever.fuse([vector_hits, bm25_hits], [1.0, 1.0])
    assert [h.chunk.id for h in fused][0] == "b"  # present in both lists -> wins


def test_llm_reranker_follows_model_order():
    llm = MockLLM(responses=[LLMResponse(content="[2, 0, 1]")])
    hits = [SearchHit(_chunk(f"c{i}"), 1.0, "hybrid", i) for i in range(3)]
    out = LLMReranker(llm).rerank("q", hits, top_n=3)
    assert [h.chunk.id for h in out] == ["c2", "c0", "c1"]
    assert all(h.retriever == "rerank" for h in out)


def test_llm_reranker_falls_back_on_garbage():
    llm = MockLLM(responses=[LLMResponse(content="I cannot help with that")])
    hits = [SearchHit(_chunk(f"c{i}"), 1.0, "hybrid", i) for i in range(3)]
    out = LLMReranker(llm).rerank("q", hits, top_n=3)
    assert [h.chunk.id for h in out] == ["c0", "c1", "c2"]  # original order kept


def test_hybrid_retrieve_over_real_data(tmp_path):
    embedder = MockEmbedder(dim=256)
    store = KnowledgeStore(tmp_path)
    IngestionPipeline(embedder, store, IngestConfig(chunk_size=180, chunk_overlap=40)).ingest_path(str(DATA))

    retriever = HybridRetriever.from_config(
        store, embedder, RetrievalConfig(top_k=3, candidate_k=10, rerank=False)
    )
    hits = retriever.retrieve("什么是混合检索，RRF 怎么融合")
    assert hits and hits[0].retriever == "hybrid"
    assert hits[0].score >= hits[-1].score
    assert len(hits) == 3


def test_rerank_enabled_uses_heuristic_offline(tmp_path):
    embedder = MockEmbedder(dim=256)
    store = KnowledgeStore(tmp_path)
    IngestionPipeline(embedder, store, IngestConfig(chunk_size=180, chunk_overlap=40)).ingest_path(str(DATA))

    retriever = HybridRetriever.from_config(
        store, embedder, RetrievalConfig(top_k=3, candidate_k=10, rerank=True), llm=MockLLM()
    )
    assert isinstance(retriever.reranker, HeuristicReranker)
    hits = retriever.retrieve("重排序 rerank 的作用")
    assert hits and hits[0].retriever == "rerank"
