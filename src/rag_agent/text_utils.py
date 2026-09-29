"""Shared text helpers: a single tokenizer reused by BM25 and the mock embedder.

The tokenizer is intentionally language-aware: it keeps ASCII words, and for
CJK text it emits the whole run plus unigrams and bigrams so that Chinese
substring overlap produces real lexical similarity without an external segmenter.
"""
from __future__ import annotations

import hashlib
import re

_WORD_RE = re.compile(r"[a-z0-9]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """Lower-cased lexical tokens (stable across processes)."""
    text = (text or "").lower()
    tokens: list[str] = _WORD_RE.findall(text)
    for run in _CJK_RE.findall(text):
        tokens.append(run)
        tokens.extend(run)  # unigrams
        tokens.extend(run[i : i + 2] for i in range(len(run) - 1))  # bigrams
    return tokens


def stable_hash(token: str) -> int:
    """Process-independent hash (Python's builtin ``hash`` is salted per run)."""
    return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
