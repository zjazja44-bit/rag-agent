"""Agentic RAG and tool tests (no network)."""
from pathlib import Path

from rag_agent.agent.agent import RAGAgent
from rag_agent.agent.tools import ToolRegistry, build_tools, make_calculator_tool, safe_eval
from rag_agent.config import IngestConfig, RetrievalConfig
from rag_agent.ingest.pipeline import IngestionPipeline
from rag_agent.llm.base import LLMResponse, ToolCall
from rag_agent.llm.mock import MockEmbedder, MockLLM
from rag_agent.retrieve.retriever import HybridRetriever
from rag_agent.store import KnowledgeStore

DATA = Path(__file__).resolve().parent.parent / "data" / "docs"


def _retriever(tmp_path):
    embedder = MockEmbedder(dim=256)
    store = KnowledgeStore(tmp_path)
    IngestionPipeline(embedder, store, IngestConfig(chunk_size=180, chunk_overlap=40)).ingest_path(str(DATA))
    return HybridRetriever.from_config(store, embedder, RetrievalConfig(top_k=3, rerank=False))


def test_safe_eval_arithmetic():
    assert safe_eval("(12 + 3) * 4") == 60
    assert safe_eval("2 ** 5") == 32
    assert safe_eval("-7 % 3") == 2


def test_calculator_tool_and_unknown_tool():
    tools = build_tools(retriever=None, names=["calculator"])
    assert tools.call("calculator", {"expression": "(12+3)*4"}) == "60"
    assert "unknown tool" in tools.call("nope", {}).lower()


def test_tool_registry_filters_bad_arguments():
    # an extra/unknown argument must not raise into the loop
    tools = build_tools(retriever=None, names=["calculator"])
    assert tools.call("calculator", {"expression": "1+1", "bogus": 1}) == "2"


def test_agent_retrieves_then_answers(tmp_path):
    agent = RAGAgent(MockLLM(), build_tools(_retriever(tmp_path), ["knowledge_search", "calculator"]))
    result = agent.run("什么是混合检索")
    assert len(result.steps) == 1
    assert result.steps[0].tool == "knowledge_search"
    assert result.answer


def test_agent_uses_calculator(tmp_path):
    agent = RAGAgent(MockLLM(), build_tools(_retriever(tmp_path), ["knowledge_search", "calculator"]))
    result = agent.run("帮算 12 * 3")
    assert result.steps and result.steps[0].tool == "calculator"
    assert result.steps[0].observation == "36"


def test_agent_respects_max_steps():
    # Model keeps requesting tools forever; the loop must stop at max_steps.
    loop_call = LLMResponse(tool_calls=[ToolCall(id="t", name="calculator", arguments={"expression": "1+1"})])
    llm = MockLLM(responses=[loop_call, loop_call, LLMResponse(content="final answer")])
    agent = RAGAgent(llm, ToolRegistry([make_calculator_tool()]), max_steps=2)
    result = agent.run("loop please")
    assert len(result.steps) == 2  # bounded
    assert result.answer == "final answer"
