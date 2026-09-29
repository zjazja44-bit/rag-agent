"""Offline providers: a deterministic mock LLM and a lexical mock embedder.

These make the whole project runnable with **no network and no API key** — used
by the test-suite and by the ``--mock`` demo path. The mock embedder is not a
stub returning noise: it is feature-hashing over the shared tokenizer, so
lexically similar text really does land closer in vector space.
"""
from __future__ import annotations

import re
from typing import Any, Iterator, Sequence

import numpy as np

from ..config import EmbeddingConfig
from ..text_utils import stable_hash, tokenize
from .base import BaseEmbedder, BaseLLM, LLMResponse, Message, ToolCall

_MATH_RE = re.compile(r"[0-9][0-9\s.+\-*/×÷()^]*[0-9)]")
_OP_RE = re.compile(r"[+\-*/×÷^]")


class MockEmbedder(BaseEmbedder):
    """Deterministic bag-of-tokens embedding via feature hashing + L2 norm."""

    def __init__(self, dim: int = 256) -> None:
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        mat = np.zeros((len(texts), self._dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in tokenize(text):
                mat[row, stable_hash(tok) % self._dim] += 1.0
            norm = np.linalg.norm(mat[row])
            if norm > 0:
                mat[row] /= norm
        return mat


def _last_content(messages: Sequence[Message], role: str) -> str:
    for msg in reversed(messages):
        if msg.get("role") == role:
            return str(msg.get("content") or "")
    return ""


def _last_tool_content(messages: Sequence[Message]) -> str:
    return _last_content(messages, "tool")


def _looks_like_math(text: str) -> bool:
    return bool(_MATH_RE.search(text)) and bool(_OP_RE.search(text))


def _extract_math(text: str) -> str:
    m = _MATH_RE.search(text)
    return m.group(0).strip() if m else text


def _after_marker(text: str, marker: str) -> str:
    idx = text.find(marker)
    if idx == -1:
        return ""
    rest = text[idx + len(marker) :].strip()
    return rest.splitlines()[0].strip() if rest else ""


# Prompt marker emitted by the query rewriter; lets the mock cooperate offline.
REWRITE_HINT = "standalone question"
REWRITE_MARKER = "Latest question:"


class MockLLM(BaseLLM):
    """Deterministic chat model.

    With no script supplied it picks a sensible canned behaviour:
    * if tools are offered and no tool has run yet, request one
      (``calculator`` for arithmetic queries, else ``knowledge_search``);
    * otherwise emit a final answer, quoting the last tool result if present.

    Pass ``responses`` to return a fixed sequence instead.
    """

    def __init__(
        self,
        model: str = "mock-llm",
        responses: list[LLMResponse] | None = None,
    ) -> None:
        self._model = model
        self._script = list(responses) if responses else []

    @property
    def model(self) -> str:
        return self._model

    def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        if self._script:
            return self._script.pop(0)
        return self._default_response(messages, tools)

    def stream_chat(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        answer = self.chat(messages, temperature=temperature, max_tokens=max_tokens).content
        for piece in answer.split(" "):
            yield piece + " "

    # -- default deterministic policy --------------------------------------
    def _default_response(
        self, messages: Sequence[Message], tools: list[dict[str, Any]] | None
    ) -> LLMResponse:
        query = _last_content(messages, "user")
        tool_names = {t["function"]["name"] for t in (tools or [])}
        already_used_tool = any(m.get("role") == "tool" for m in messages)

        # Query-rewrite intent: cooperate with the rewriter off (offline demo).
        if REWRITE_HINT in _last_content(messages, "system").lower():
            return LLMResponse(content=_after_marker(query, REWRITE_MARKER) or query)

        if tools and not already_used_tool:
            if "calculator" in tool_names and _looks_like_math(query):
                return LLMResponse(
                    tool_calls=[
                        ToolCall(
                            id="call_calc",
                            name="calculator",
                            arguments={"expression": _extract_math(query)},
                        )
                    ]
                )
            if "knowledge_search" in tool_names:
                return LLMResponse(
                    tool_calls=[
                        ToolCall(
                            id="call_search",
                            name="knowledge_search",
                            arguments={"query": query},
                        )
                    ]
                )

        tool_text = _last_tool_content(messages)
        if tool_text and tool_text.strip():
            return LLMResponse(content=f"根据检索到的资料：{tool_text.strip()[:200]} [1]")
        return LLMResponse(content=f"[mock] 针对「{query}」的回答。")
