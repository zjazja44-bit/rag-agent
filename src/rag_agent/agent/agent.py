"""The tool-calling agent loop.

Unlike the fixed RAG chain, the agent decides at each step whether to call a
tool or answer. The loop is bounded by ``max_steps`` so a model that keeps
calling tools can never run forever or blow the budget.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..llm.base import BaseLLM, LLMResponse, Message
from .tools import ToolRegistry

_SYSTEM = (
    "你是一个可调用工具的知识库问答智能体。面对事实性问题，请先调用 knowledge_search "
    "检索资料，再依据资料回答，并用 [序号] 标注引用。遇到算术问题可调用 calculator。"
    "当信息已足够时，直接给出最终答案，不要无意义地反复调用工具。"
)


@dataclass
class AgentStep:
    tool: str
    arguments: dict[str, Any]
    observation: str


@dataclass
class AgentResult:
    answer: str
    steps: list[AgentStep] = field(default_factory=list)
    messages: list[Message] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)


def _accumulate(total: dict[str, int], usage: dict[str, int]) -> None:
    for key, value in (usage or {}).items():
        total[key] = total.get(key, 0) + value


def _assistant_message(resp: LLMResponse) -> Message:
    return {
        "role": "assistant",
        "content": resp.content or "",
        "tool_calls": [
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
            }
            for tc in resp.tool_calls
        ],
    }


class RAGAgent:
    def __init__(self, llm: BaseLLM, tools: ToolRegistry, max_steps: int = 6) -> None:
        self.llm = llm
        self.tools = tools
        self.max_steps = max_steps

    def run(self, question: str, history: Sequence[Message] | None = None) -> AgentResult:
        messages: list[Message] = [{"role": "system", "content": _SYSTEM}]
        for turn in history or []:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append({"role": "user", "content": question})

        steps: list[AgentStep] = []
        usage_total: dict[str, int] = {}

        for _ in range(self.max_steps):
            resp = self.llm.chat(messages, tools=self.tools.schemas())
            _accumulate(usage_total, resp.usage)

            if not resp.tool_calls:
                return AgentResult(resp.content, steps, messages, usage_total)

            messages.append(_assistant_message(resp))
            for call in resp.tool_calls:
                observation = self.tools.call(call.name, call.arguments)
                steps.append(AgentStep(tool=call.name, arguments=call.arguments, observation=observation))
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "name": call.name, "content": observation}
                )

        # Budget exhausted: force a final answer with tools disabled.
        final = self.llm.chat(messages)
        _accumulate(usage_total, final.usage)
        return AgentResult(final.content, steps, messages, usage_total)
