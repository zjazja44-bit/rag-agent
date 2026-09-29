"""Query rewriting for multi-turn conversations.

Follow-up questions like "它的缺点是什么" are unretrievable on their own — the
pronoun 它 has no referent. The rewriter folds the conversation history into the
latest question to produce a standalone query. When there is no history, or the
model output looks unusable, we fall back to the original question.
"""
from __future__ import annotations

from typing import Sequence

from ..llm.base import BaseLLM, Message

_SYSTEM = (
    "You rewrite a user's latest question into a standalone question that can be "
    "understood without the conversation history. Resolve pronouns and implicit "
    "references using the history. Output ONLY the rewritten standalone question, "
    "with no preamble and no quotes."
)


def _format_history(history: Sequence[Message]) -> str:
    return "\n".join(f"{m['role']}: {m['content']}" for m in history)


class LLMQueryRewriter:
    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm

    def rewrite(self, question: str, history: Sequence[Message] | None = None) -> str:
        if not history:
            return question
        user = (
            f"Conversation so far:\n{_format_history(history)}\n\n"
            f"Latest question: {question}\n\nStandalone question:"
        )
        resp = self.llm.chat(
            [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": user}],
            temperature=0.0,
        )
        text = (resp.content or "").strip().strip('"').strip()
        if not text or "\n" in text or len(text) > 300:
            return question  # model misbehaved -> keep the original
        return text


class IdentityRewriter:
    """No-op rewriter (used when no LLM is available)."""

    def rewrite(self, question: str, history: Sequence[Message] | None = None) -> str:
        return question


def build_query_rewriter(cfg, llm: BaseLLM | None):
    return LLMQueryRewriter(llm) if llm is not None else IdentityRewriter()
