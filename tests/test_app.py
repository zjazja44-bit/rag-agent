"""Application wiring, HTTP API and CLI tests (all offline via mock)."""
import json

from fastapi.testclient import TestClient

from rag_agent.api.server import create_app
from rag_agent.app import RAGApp
from rag_agent.cli import main
from rag_agent.config import load_config


def _mock_env(monkeypatch, tmp_path):
    monkeypatch.setenv("RAG__LLM__PROVIDER", "mock")
    monkeypatch.setenv("RAG__EMBEDDING__PROVIDER", "mock")
    monkeypatch.setenv("RAG__STORE__DIR", str(tmp_path / "store"))


def test_app_end_to_end(tmp_path, monkeypatch):
    _mock_env(monkeypatch, tmp_path)
    app = RAGApp(load_config())

    assert app.ingest("data/docs") > 0
    payload = app.answer("什么是混合检索")
    assert payload["answer"]
    assert payload["citations"]
    assert app.stats()["chunks"] > 0


def test_http_endpoints(tmp_path, monkeypatch):
    _mock_env(monkeypatch, tmp_path)
    app = RAGApp(load_config())
    app.ingest("data/docs")
    client = TestClient(create_app(app))

    assert client.get("/health").json()["status"] == "ok"

    ask = client.post("/ask", json={"question": "什么是 RRF 融合"}).json()
    assert ask["answer"] and ask["citations"]

    ingested = client.get("/stats").json()
    assert ingested["chunks"] > 0

    stream = client.post("/chat/stream", json={"question": "重排序有什么作用"})
    assert stream.status_code == 200
    events = [line for line in stream.text.splitlines() if line.startswith("data: ")]
    types = [json.loads(e[6:])["type"] for e in events]
    assert "token" in types and types[-1] == "done"


def test_cli_ingest_and_ask(tmp_path, monkeypatch, capsys):
    _mock_env(monkeypatch, tmp_path)
    assert main(["--mock", "ingest", "data/docs"]) == 0
    assert "新增" in capsys.readouterr().out

    assert main(["--mock", "ask", "什么是向量检索"]) == 0
    out = capsys.readouterr().out
    assert out.strip()
    assert "来源" in out  # citations are printed
