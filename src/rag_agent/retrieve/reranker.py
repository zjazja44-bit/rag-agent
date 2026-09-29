"""Rerankers: a two-stage retrieve-then-rerank design.

Recall stage returns many candidates cheaply; the reranker is more expensive but
more accurate and orders the final few. Two implementations:

* :class:`LLMReranker` asks the chat model to order candidates (best quality,
  costs tokens, degrades gracefully to the input order if parsing fails).
* :class:`HeuristicReranker` scores by query/chunk token overlap — fully
  offline and deterministic.
"""
from __future__ import annotations

import re
from dataclasses import replace
from typing import Sequence

from ..llm.base import BaseLLM
from ..logging_utils import get_logger
from ..store.base import SearchHit
from ..text_utils import tokenize

logger = get_logger(__name__)

_SYSTEM = (
    "You are a search reranker. Given a question and numbered candidate "
    "passages, reorder the passages from most to least relevant to the "
    "question. Respond with ONLY a JSON array of the passage indices in order, "
    "e.g. [3, 1, 2]. Do not add any explanation."
)


class LLMReranker:
    def __init__(self, llm: BaseLLM, max_candidates: int = 20) -> None:
        self.llm = llm
        self.max_candidates = max_candidates

    def rerank(self, query: str, hits: Sequence[SearchHit], top_n: int) -> list[SearchHit]:
        candidates = list(hits[: self.max_candidates])
        if len(candidates) <= 1:
            return candidates[:top_n]

        listing = "\n".join(
            f"[{i}] {h.chunk.text[:200].replace(chr(10), ' ')}"
            for i, h in enumerate(candidates)
        )
        try:
            resp = self.llm.chat(
                [
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": f"Question: {query}\n\nPassages:\n{listing}"},
                ],
                temperature=0.0,
            )
            order = _parse_order(resp.content, len(candidates))
        except Exception as exc:  # never let a reranker break the pipeline
            logger.warning("LLM rerank failed (%s); keeping retrieval order", exc)
            order = None

        if not order:
            return _reorder(candidates, list(range(len(candidates))), top_n)

        seen = set(order)
        full_order = order + [i for i in range(len(candidates)) if i not in seen]
        return _reorder(candidates, full_order, top_n)


class HeuristicReranker:
    """Order by token overlap with the query — cheap, offline, deterministic."""

    def rerank(self, query: str, hits: Sequence[SearchHit], top_n: int) -> list[SearchHit]:
        q_tokens = set(tokenize(query))
        scored = []
        for h in hits:
            overlap = len(q_tokens & set(tokenize(h.chunk.text)))
            scored.append((overlap, h))
        scored.sort(key=lambda t: t[0], reverse=True)
        ordered = [h for _, h in scored]
        return _reorder(ordered, list(range(len(ordered))), top_n)


def _parse_order(text: str, n_candidates: int) -> list[int]:
    """Extract an ordered, de-duplicated list of valid indices from model text."""
    order: list[int] = []
    for token in re.findall(r"\d+", text or ""):
        idx = int(token)
        if 0 <= idx < n_candidates and idx not in order:
            order.append(idx)
    return order


def _reorder(candidates: Sequence[SearchHit], order: Sequence[int], top_n: int) -> list[SearchHit]:
    out: list[SearchHit] = []
    for rank, idx in enumerate(order[:top_n]):
        hit = candidates[idx]
        out.append(replace(hit, rank=rank, retriever="rerank"))
    return out


def build_reranker(cfg, llm: BaseLLM | None):
    """Return a reranker when reranking is enabled, else None."""
    if not cfg.rerank:
        return None
    from ..llm.mock import MockLLM  # local import to avoid a cycle

    if llm is None or isinstance(llm, MockLLM):
        return HeuristicReranker()
    return LLMReranker(llm)
