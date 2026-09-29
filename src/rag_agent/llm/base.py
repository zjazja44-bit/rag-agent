"""Provider-agnostic LLM / embedding interfaces.

Concrete providers live in sibling modules (``openai_compat``, ``mock``,
``local_embedder``) and are wired up by :func:`rag_agent.llm.factory.build_llm`
and :func:`rag_agent.llm.factory.build_embedder`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterator, Sequence

import numpy as np

# A chat message in OpenAI wire format (role/content/tool_calls/tool_call_id/...).
Message = dict[str, Any]


@dataclass
class ToolCall:
    """A model-requested tool invocation."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Normalised response regardless of provider."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    raw: Any = None


class BaseLLM(ABC):
    """Chat model with optional tool-calling."""

    @property
    @abstractmethod
    def model(self) -> str:
        ...

    @abstractmethod
    def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        ...

    def stream_chat(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        """Yield answer text incrementally. Default falls back to one chunk."""
        yield self.chat(
            messages, temperature=temperature, max_tokens=max_tokens, **kwargs
        ).content


class BaseEmbedder(ABC):
    """Text -> dense vector encoder."""

    @property
    @abstractmethod
    def dim(self) -> int:
        ...

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an ``(n, dim)`` float32 matrix, L2-normalised where possible."""

    def embed_one(self, text: str) -> np.ndarray:
        return self.embed([text])[0]
