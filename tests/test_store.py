"""Vector store + BM25 + ingestion integration tests (no network)."""
from pathlib import Path

from rag_agent.config import IngestConfig
from rag_agent.ingest.pipeline import IngestionPipeline
from rag_agent.llm.mock import MockEmbedder
from rag_agent.store import KnowledgeStore

DATA = Path(__file__).resolve().parent.parent / "data" / "docs"


def _build(tmp_path):
    embedder = MockEmbedder(dim=256)
    store = KnowledgeStore(tmp_path)
    pipe = IngestionPipeline(embedder, store, IngestConfig(chunk_size=180, chunk_overlap=40))
    return embedder, store, pipe


def test_ingest_populates_both_indexes(tmp_path):
    embedder, store, pipe = _build(tmp_path)
    n = pipe.ingest_path(str(DATA))
    assert n > 0
    assert len(store) == n
    assert len(store.bm25) == n


def test_vector_and_bm25_search(tmp_path):
    embedder, store, pipe = _build(tmp_path)
    pipe.ingest_path(str(DATA))

    query = "混合检索如何融合两路结果"
    vhits = store.vector_store.search(embedder.embed_one(query), top_k=3)
    assert vhits and vhits[0].retriever == "vector"
    assert vhits[0].score >= vhits[-1].score

    bhits = store.bm25.search(query, top_k=3)
    assert bhits and bhits[0].retriever == "bm25"
    assert any("BM25" in h.chunk.text or "混合检索" in h.chunk.text for h in bhits)


def test_persistence_round_trip(tmp_path):
    embedder, store, pipe = _build(tmp_path)
    n = pipe.ingest_path(str(DATA))

    reloaded = KnowledgeStore(tmp_path)
    assert reloaded.load() is True
    assert len(reloaded) == n
    # BM25 is rebuilt from persisted chunks, so it works without its own file
    assert reloaded.bm25.search("查询改写", top_k=3)


def test_reingest_is_idempotent(tmp_path):
    _, _, pipe = _build(tmp_path)
    assert pipe.ingest_path(str(DATA)) > 0
    assert pipe.ingest_path(str(DATA)) == 0  # same content -> same ids -> nothing new
