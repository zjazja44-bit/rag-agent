"""Provider smoke tests (no network)."""
import numpy as np

from rag_agent.llm.mock import MockEmbedder, MockLLM


def test_mock_embedder_lexical_similarity():
    emb = MockEmbedder(dim=128)
    v = emb.embed(["向量检索与知识库问答", "知识库问答系统", "今天天气很不错"])
    assert v.shape == (3, 128)
    sim_close = float(np.dot(v[0], v[1]))
    sim_far = float(np.dot(v[0], v[2]))
    assert sim_close > sim_far


def test_mock_llm_plain_answer():
    r = MockLLM().chat([{"role": "user", "content": "你好"}])
    assert r.content
    assert not r.tool_calls


def test_mock_llm_requests_tool_then_answers():
    llm = MockLLM()
    tools = [
        {"type": "function", "function": {"name": "knowledge_search", "description": "", "parameters": {}}}
    ]
    first = llm.chat([{"role": "user", "content": "什么是 RAG"}], tools=tools)
    assert first.tool_calls and first.tool_calls[0].name == "knowledge_search"

    second = llm.chat(
        [
            {"role": "user", "content": "什么是 RAG"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_search", "type": "function", "function": {"name": "knowledge_search", "arguments": "{}"}}
                ],
            },
            {"role": "tool", "tool_call_id": "call_search", "content": "RAG 是检索增强生成"},
        ],
        tools=tools,
    )
    assert not second.tool_calls
    assert "RAG 是检索增强生成" in second.content


def test_mock_llm_calculator_intent():
    tools = [
        {"type": "function", "function": {"name": "calculator", "description": "", "parameters": {}}}
    ]
    r = MockLLM().chat([{"role": "user", "content": "帮我算 12 * 3 等于几"}], tools=tools)
    assert r.tool_calls and r.tool_calls[0].name == "calculator"
    assert r.tool_calls[0].arguments["expression"]
