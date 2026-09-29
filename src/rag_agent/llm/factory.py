"""Build providers from configuration."""
from __future__ import annotations

from ..config import EmbeddingConfig, LLMConfig
from .base import BaseEmbedder, BaseLLM
from .mock import MockEmbedder, MockLLM

__all__ = [
    "BaseLLM",
    "BaseEmbedder",
    "LLMResponse",
    "ToolCall",
    "Message",
    "build_llm",
    "build_embedder",
]


def build_llm(cfg: LLMConfig) -> BaseLLM:
    if cfg.provider == "mock":
        return MockLLM()
    if cfg.provider == "openai_compat":
        from .openai_compat import OpenAICompatLLM

        return OpenAICompatLLM(cfg)
    raise ValueError(f"Unknown llm provider: {cfg.provider!r}")


def build_embedder(cfg: EmbeddingConfig) -> BaseEmbedder:
    if cfg.provider == "mock":
        return MockEmbedder(dim=cfg.dim)
    if cfg.provider == "local":
        from .local_embedder import LocalEmbedder

        return LocalEmbedder(cfg)
    if cfg.provider == "fastembed":
        from .fastembed_embedder import FastEmbedEmbedder

        return FastEmbedEmbedder(cfg)
    if cfg.provider == "openai_compat":
        from .openai_compat import OpenAICompatEmbedder

        return OpenAICompatEmbedder(cfg)
    raise ValueError(f"Unknown embedding provider: {cfg.provider!r}")
