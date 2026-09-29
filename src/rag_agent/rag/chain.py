"""The RAG chain: retrieve -> build a cited context -> generate.

This is the "reader" half of the system. It keeps the two stages explicit so the
retrieved evidence survives all the way to the response (for citations and for
the evaluation harness to score faithfulness against).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

from ..llm.base import BaseLLM, Message
from ..store.base import SearchHit

_SYSTEM_PROMPT = (
    "你是一个严谨的知识库问答助手。请严格依据「资料」回答用户问题，"
    "不要编造资料之外的内容。回答时在相应句子后用 [序号] 标注依据的资料编号"
    "（例如 [1]、[2]）。如果资料不足以回答，请明确说明「资料不足」。"
)


@dataclass
class RAGResult:
    query: str
    answer: str
    hits: list[SearchHit] = field(default_factory=list)
    context: str = ""
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def citations(self) -> list[dict]:
        return format_citations(self.hits)


@dataclass
class PreparedAnswer:
    """Retrieval result plus the assembled messages, before any generation."""

    query: str
    hits: list[SearchHit]
    context: str
    messages: list[Message]


def build_context(hits: Sequence[SearchHit]) -> str:
    blocks: list[str] = []
    for i, hit in enumerate(hits, start=1):
        name = Path(hit.chunk.source).name if hit.chunk.source else "?"
        blocks.append(f"[{i}] (来源: {name} #{hit.chunk.index})\n{hit.chunk.text.strip()}")
    return "\n\n".join(blocks)


def format_citations(hits: Sequence[SearchHit]) -> list[dict]:
    return [
        {
            "id": i,
            "source": hit.chunk.source,
            "doc_id": hit.chunk.doc_id,
            "chunk_id": hit.chunk.id,
            "index": hit.chunk.index,
            "score": round(float(hit.score), 4),
        }
        for i, hit in enumerate(hits, start=1)
    ]


class RAGChain:
    def __init__(self, retriever, llm: BaseLLM, rewriter=None) -> None:
        self.retriever = retriever
        self.llm = llm
        self.rewriter = rewriter

    def prepare(
        self,
        question: str,
        history: Sequence[Message] | None = None,
        top_k: int | None = None,
    ) -> PreparedAnswer:
        query = self.rewriter.rewrite(question, history) if self.rewriter else question
        hits = self.retriever.retrieve(query, top_k)
        context = build_context(hits)

        messages: list[Message] = [{"role": "system", "content": _SYSTEM_PROMPT}]
        for turn in history or []:
            messages.append({"role": turn["role"], "content": turn["content"]})
        messages.append(
            {
                "role": "user",
                "content": f"资料：\n{context}\n\n问题：{query}\n\n请依据资料作答，并标注引用序号。",
            }
        )
        return PreparedAnswer(query=query, hits=hits, context=context, messages=messages)

    def answer(
        self,
        question: str,
        history: Sequence[Message] | None = None,
        top_k: int | None = None,
    ) -> RAGResult:
        prepared = self.prepare(question, history, top_k)
        resp = self.llm.chat(prepared.messages)
        return RAGResult(
            query=prepared.query,
            answer=resp.content,
            hits=prepared.hits,
            context=prepared.context,
            usage=resp.usage,
        )

    def stream_answer(
        self,
        question: str,
        history: Sequence[Message] | None = None,
        top_k: int | None = None,
    ) -> tuple[PreparedAnswer, Iterator[str]]:
        prepared = self.prepare(question, history, top_k)
        return prepared, self.llm.stream_chat(prepared.messages)
