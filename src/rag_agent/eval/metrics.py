"""Evaluation metrics.

Retrieval metrics measure whether the relevant documents were recalled. Answer
metrics are heuristic and offline by design: a real faithfulness score needs an
LLM judge, so we report a cheap lexical proxy (how much of the answer is backed
by the retrieved context) plus citation presence. Honest about what each number is.
"""
from __future__ import annotations

import re
from typing import Sequence

from ..text_utils import tokenize

_CITATION_RE = re.compile(r"\[\d+\]")


# -- retrieval -------------------------------------------------------------- #
def hit_rate(retrieved: Sequence[str], relevant: Sequence[str], k: int | None = None) -> float:
    """1.0 if any relevant doc is in the top-k, else 0.0."""
    if not relevant:
        return 0.0
    rel = set(relevant)
    return 1.0 if any(r in rel for r in _topk(retrieved, k)) else 0.0


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int | None = None) -> float:
    """Fraction of relevant docs that were retrieved."""
    if not relevant:
        return 0.0
    rel = set(relevant)
    got = rel & set(_topk(retrieved, k))
    return len(got) / len(rel)


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int | None = None) -> float:
    """Fraction of retrieved (top-k) docs that are relevant."""
    top = _topk(retrieved, k)
    if not top:
        return 0.0
    rel = set(relevant)
    return sum(1 for r in top if r in rel) / len(top)


def mrr(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    """Reciprocal rank of the first relevant doc."""
    rel = set(relevant)
    for i, doc in enumerate(retrieved, start=1):
        if doc in rel:
            return 1.0 / i
    return 0.0


# -- generation (heuristic) ------------------------------------------------- #
def citation_rate(answer: str) -> float:
    """1.0 if the answer contains at least one [n] citation marker."""
    return 1.0 if _CITATION_RE.search(answer or "") else 0.0


def answer_coverage(answer: str, keywords: Sequence[str]) -> float:
    """Fraction of expected keywords present in the answer (recall proxy)."""
    if not keywords:
        return 0.0
    low = (answer or "").lower()
    return sum(1 for kw in keywords if kw.lower() in low) / len(keywords)


def groundedness(answer: str, context: str) -> float:
    """Lexical proxy: fraction of answer tokens that appear in the context.

    A high value means the answer mostly reuses retrieved wording (grounded);
    a low value hints the model may have gone off-script. Not a substitute for
    an LLM judge — see :func:`judge_faithfulness` for that.
    """
    answer_tokens = set(tokenize(answer))
    if not answer_tokens:
        return 0.0
    context_tokens = set(tokenize(context))
    return len(answer_tokens & context_tokens) / len(answer_tokens)


JUDGE_PROMPT = (
    "你是严格的评审。给定「资料」和「回答」，判断回答是否忠实于资料，"
    "是否存在资料未支持的内容。只回复一个 0 到 1 之间的数字（1 表示完全忠实）。"
)


def judge_faithfulness(llm, answer: str, context: str) -> float | None:
    """Optional LLM-as-judge faithfulness (needs a real model). Returns None on failure."""
    try:
        resp = llm.chat(
            [
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": f"资料：\n{context}\n\n回答：\n{answer}"},
            ],
            temperature=0.0,
        )
        match = re.search(r"\d+(?:\.\d+)?", resp.content or "")
        if not match:
            return None
        return max(0.0, min(1.0, float(match.group(0))))
    except Exception:
        return None


def _topk(items: Sequence[str], k: int | None) -> list[str]:
    return list(items[:k]) if k else list(items)
