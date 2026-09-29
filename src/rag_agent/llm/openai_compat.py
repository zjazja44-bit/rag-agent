"""OpenAI-compatible provider: works with OpenAI, DeepSeek, 智谱, 通义, vLLM, ...

Only the two endpoints we need are used: ``chat.completions`` (with optional
tool-calling and streaming) and ``embeddings``. The host is fully configurable
via ``base_url`` so the same code targets every OpenAI-compatible vendor.
"""
from __future__ import annotations

import json
from typing import Any, Iterator, Sequence

import numpy as np

from ..config import EmbeddingConfig, LLMConfig
from .base import BaseEmbedder, BaseLLM, LLMResponse, Message, ToolCall


def _to_response(resp: Any) -> LLMResponse:
    choice = resp.choices[0]
    msg = choice.message

    tool_calls: list[ToolCall] = []
    for tc in getattr(msg, "tool_calls", None) or []:
        raw_args = tc.function.arguments or "{}"
        try:
            args = json.loads(raw_args)
        except json.JSONDecodeError:
            args = {"_raw": raw_args}
        tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

    usage: dict[str, int] = {}
    u = getattr(resp, "usage", None)
    if u is not None:
        usage = {
            "prompt_tokens": getattr(u, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(u, "completion_tokens", 0) or 0,
            "total_tokens": getattr(u, "total_tokens", 0) or 0,
        }

    return LLMResponse(content=msg.content or "", tool_calls=tool_calls, usage=usage, raw=resp)


class OpenAICompatLLM(BaseLLM):
    def __init__(self, cfg: LLMConfig) -> None:
        if not cfg.api_key:
            raise ValueError(
                f"No API key found. Set the environment variable {cfg.api_key_env!r} "
                f"(or run with RAG__LLM__PROVIDER=mock for an offline demo)."
            )
        from openai import OpenAI  # imported lazily so mock users skip the dependency

        self._cfg = cfg
        self._client = OpenAI(
            api_key=cfg.api_key, base_url=cfg.base_url, timeout=cfg.request_timeout
        )

    @property
    def model(self) -> str:
        return self._cfg.chat_model

    def chat(
        self,
        messages: Sequence[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        params: dict[str, Any] = {
            "model": self._cfg.chat_model,
            "messages": list(messages),
            "temperature": self._cfg.temperature if temperature is None else temperature,
            "max_tokens": self._cfg.max_tokens if max_tokens is None else max_tokens,
        }
        if tools:
            params["tools"] = tools
        params.update(kwargs)
        resp = self._client.chat.completions.create(**params)
        return _to_response(resp)

    def stream_chat(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> Iterator[str]:
        params: dict[str, Any] = {
            "model": self._cfg.chat_model,
            "messages": list(messages),
            "temperature": self._cfg.temperature if temperature is None else temperature,
            "max_tokens": self._cfg.max_tokens if max_tokens is None else max_tokens,
            "stream": True,
        }
        params.update(kwargs)
        for chunk in self._client.chat.completions.create(**params):
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content


class OpenAICompatEmbedder(BaseEmbedder):
    def __init__(self, cfg: EmbeddingConfig) -> None:
        if not cfg.api_key:
            raise ValueError(
                f"No API key found. Set the environment variable {cfg.api_key_env!r} "
                f"(or run with RAG__EMBEDDING__PROVIDER=mock for an offline demo)."
            )
        from openai import OpenAI

        self._cfg = cfg
        self._client = OpenAI(
            api_key=cfg.api_key, base_url=cfg.base_url, timeout=cfg.request_timeout
        )
        self._dim: int | None = None

    @property
    def dim(self) -> int:
        return self._dim if self._dim is not None else self._cfg.dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        vectors: list[list[float]] = []
        for i in range(0, len(texts), self._cfg.batch_size):
            batch = texts[i : i + self._cfg.batch_size]
            resp = self._client.embeddings.create(model=self._cfg.model, input=batch)
            vectors.extend(item.embedding for item in resp.data)

        arr = np.asarray(vectors, dtype=np.float32)
        self._dim = int(arr.shape[1])
        return _l2_normalise(arr)


def _l2_normalise(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return arr / norms
