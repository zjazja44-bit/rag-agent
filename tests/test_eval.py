"""Metric and evaluation-runner tests (no network)."""
from rag_agent.app import RAGApp
from rag_agent.config import load_config
from rag_agent.eval import metrics
from rag_agent.eval.runner import run_eval


def test_retrieval_metrics():
    retrieved, relevant = ["a", "b", "c"], ["b"]
    assert metrics.hit_rate(retrieved, relevant, 2) == 1.0
    assert metrics.recall_at_k(retrieved, relevant, 3) == 1.0
    assert metrics.precision_at_k(retrieved, relevant, 3) == 1 / 3
    assert metrics.mrr(retrieved, relevant) == 0.5
    assert metrics.hit_rate(["c"], ["b"], 1) == 0.0
    assert metrics.mrr(["x", "y"], ["y"]) == 0.5


def test_generation_metrics():
    assert metrics.groundedness("向量检索 语义", "资料：向量检索能捕捉语义") > 0.5
    assert metrics.answer_coverage("关于向量检索的答案", ["向量", "检索"]) == 1.0
    assert metrics.citation_rate("答案见 [1]") == 1.0
    assert metrics.citation_rate("没有引用") == 0.0


def test_run_eval_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG__LLM__PROVIDER", "mock")
    monkeypatch.setenv("RAG__EMBEDDING__PROVIDER", "mock")
    monkeypatch.setenv("RAG__STORE__DIR", str(tmp_path / "store"))

    app = RAGApp(load_config())
    app.ingest("data/docs")
    report = run_eval(app, "data/eval/qa.jsonl", verbose=False)

    assert len(report["rows"]) == 10
    assert report["aggregate"]["hit_rate"] >= 0.5
    assert set(report["aggregate"]) >= {"hit_rate", "recall", "mrr", "groundedness"}
