"""Application assembly: config -> providers -> store -> retriever -> chain/agent.

One object the CLI, the server and the tests can all build from a config.
"""
from __future__ import annotations

from typing import Any, Sequence

from .agent.agent import RAGAgent
from .agent.tools import build_tools
from .config import AppConfig, load_config, load_env_file
from .ingest.pipeline import IngestionPipeline
from .llm.base import Message
from .llm.factory import build_embedder, build_llm
from .logging_utils import get_logger
from .memory import SessionStore
from .rag.chain import RAGChain, RAGResult
from .rag.rewrite import build_query_rewriter
from .retrieve.retriever import HybridRetriever
from .store.knowledge_base import KnowledgeStore

logger = get_logger(__name__)


class RAGApp:
    def __init__(self, config: AppConfig | None = None) -> None:
        load_env_file()
        self.config = config or load_config()
        self.llm = build_llm(self.config.llm)
        self.embedder = build_embedder(self.config.embedding)

        self.store = KnowledgeStore(self.config.store.path)
        if self.store.load():
            logger.info("Loaded %d chunk(s) from %s", len(self.store), self.config.store.path)

        self.retriever = HybridRetriever.from_config(
            self.store, self.embedder, self.config.retrieval, llm=self.llm
        )
        self.rewriter = build_query_rewriter(self.config, self.llm)
        self.chain = RAGChain(self.retriever, self.llm, self.rewriter)
        self.agent = RAGAgent(
            self.llm,
            build_tools(self.retriever, self.config.agent.tools),
            self.config.agent.max_steps,
        )
        self.sessions = SessionStore(self.config.memory.max_turns)

    # -- ingestion ---------------------------------------------------------
    def ingest(self, path: str) -> int:
        pipeline = IngestionPipeline(self.embedder, self.store, self.config.ingest)
        return pipeline.ingest_path(path)

    # -- question answering ------------------------------------------------
    def answer(
        self,
        question: str,
        session_id: str | None = None,
        mode: str = "rag",
        top_k: int | None = None,
    ) -> dict[str, Any]:
        history = self.sessions.get(session_id).history() if session_id else None
        if mode == "agent":
            result = self.agent.run(question, history=history)
            payload = {
                "mode": "agent",
                "answer": result.answer,
                "citations": [],
                "steps": [
                    {"tool": s.tool, "arguments": s.arguments, "observation": s.observation[:800]}
                    for s in result.steps
                ],
            }
        else:
            result: RAGResult = self.chain.answer(question, history=history, top_k=top_k)
            payload = {
                "mode": "rag",
                "query": result.query,
                "answer": result.answer,
                "citations": result.citations,
            }

        if session_id:
            memory = self.sessions.get(session_id)
            memory.add_user(question)
            memory.add_assistant(payload["answer"])
        return payload

    def stream(
        self, question: str, session_id: str | None = None, top_k: int | None = None
    ):
        """Return ``(prepared, token_iterator)``; caller records memory on completion."""
        history = self.sessions.get(session_id).history() if session_id else None
        return self.chain.stream_answer(question, history=history, top_k=top_k)

    def remember_turn(self, session_id: str, question: str, answer: str) -> None:
        memory = self.sessions.get(session_id)
        memory.add_user(question)
        memory.add_assistant(answer)

    def stats(self) -> dict[str, Any]:
        return {
            "chunks": len(self.store),
            "llm_provider": self.config.llm.provider,
            "llm_model": self.llm.model,
            "embedding_provider": self.config.embedding.provider,
            "store": str(self.config.store.path),
            "tools": self.agent.tools.names(),
        }
