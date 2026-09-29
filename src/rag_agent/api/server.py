"""FastAPI service: JSON ask, streaming SSE chat, ingest and stats endpoints."""
from __future__ import annotations

import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from .. import __version__
from ..app import RAGApp
from ..rag.chain import format_citations


class AskRequest(BaseModel):
    question: str
    session_id: str | None = None
    mode: str = "rag"
    top_k: int | None = None


class IngestRequest(BaseModel):
    path: str


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def create_app(rag_app: RAGApp) -> FastAPI:
    app = FastAPI(title="rag-agent", version=__version__)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/stats")
    def stats() -> dict:
        return rag_app.stats()

    @app.post("/ingest")
    def ingest(req: IngestRequest) -> dict:
        added = rag_app.ingest(req.path)
        return {"added": added, "chunks": len(rag_app.store)}

    @app.post("/ask")
    def ask(req: AskRequest) -> dict:
        return rag_app.answer(
            req.question, session_id=req.session_id, mode=req.mode, top_k=req.top_k
        )

    @app.post("/chat/stream")
    def chat_stream(req: AskRequest) -> StreamingResponse:
        def event_stream():
            try:
                prepared, tokens = rag_app.stream(req.question, session_id=req.session_id, top_k=req.top_k)
            except Exception as exc:  # fail loudly in the stream
                yield _sse({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
                return

            collected: list[str] = []
            for token in tokens:
                collected.append(token)
                yield _sse({"type": "token", "text": token})

            answer = "".join(collected)
            if req.session_id:
                rag_app.remember_turn(req.session_id, req.question, answer)
            yield _sse({"type": "done", "answer": answer, "citations": format_citations(prepared.hits)})

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    return app


def create_app_from_config(config=None) -> FastAPI:
    from ..config import load_config

    return create_app(RAGApp(config or load_config()))
