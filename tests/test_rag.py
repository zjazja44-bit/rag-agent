"""RAG chain and query-rewriting tests (no network)."""
from pathlib import Path

from rag_agent.config import IngestConfig, RetrievalConfig
from rag_agent.ingest.pipeline import IngestionPipeline
from rag_agent.llm.base import LLMResponse
from rag_agent.llm.mock import MockEmbedder, MockLLM
from rag_agent.rag.chain import RAGChain, build_context
from rag_agent.rag.rewrite import IdentityRewriter, LLMQueryRewriter
from rag_agent.retrieve.retriever import HybridRetriever
from rag_agent.store import KnowledgeStore
from rag_agent.store.base import SearchHit
from rag_agent.types import Chunk

DATA = Path(__file__).resolve().parent.parent / "data" / "docs"


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(id=cid, text=text, doc_id="d", source="docs/x.md", index=0)


def _retriever(tmp_path):
    embedder = MockEmbedder(dim=256)
    store = KnowledgeStore(tmp_path)
    IngestionPipeline(embedder, store, IngestConfig(chunk_size=180, chunk_overlap=40)).ingest_path(str(DATA))
    return HybridRetriever.from_config(store, embedder, RetrievalConfig(top_k=3, rerank=False))


def test_build_context_numbers_sources():
    hits = [
        SearchHit(_chunk("c1", "文本一"), 0.5, "hybrid", 0),
        SearchHit(_chunk("c2", "文本二"), 0.4, "hybrid", 1),
    ]
    ctx = build_context(hits)
    assert ctx.startswith("[1]")
    assert "[2]" in ctx
    assert "文本一" in ctx


def test_rag_chain_answer_has_citations(tmp_path):
    chain = RAGChain(_retriever(tmp_path), MockLLM(), rewriter=IdentityRewriter())
    result = chain.answer("什么是混合检索")
    assert result.answer
    assert result.hits
    assert len(result.citations) == len(result.hits)
    assert result.context.startswith("[1]")


def test_query_rewriter_resolves_with_scripted_model():
    llm = MockLLM(responses=[LLMResponse(content="向量检索有哪些缺点")])
    rewriter = LLMQueryRewriter(llm)
    rewritten = rewriter.rewrite(
        "它的缺点是什么",
        history=[
            {"role": "user", "content": "介绍一下向量检索"},
            {"role": "assistant", "content": "向量检索是……"},
        ],
    )
    assert rewritten == "向量检索有哪些缺点"


def test_query_rewriter_identity_without_history():
    assert LLMQueryRewriter(MockLLM()).rewrite("你好", history=None) == "你好"


def test_query_rewriter_falls_back_on_multiline_output():
    llm = MockLLM(responses=[LLMResponse(content="第一行\n第二行")])
    assert LLMQueryRewriter(llm).rewrite("它呢", history=[{"role": "user", "content": "x"}]) == "它呢"


def test_multiturn_offline_uses_latest_question():
    # The mock cooperates with the rewriter, so the standalone query is the latest question.
    rewritten = LLMQueryRewriter(MockLLM()).rewrite(
        "向量检索的缺点", history=[{"role": "user", "content": "介绍向量检索"}]
    )
    assert rewritten == "向量检索的缺点"
