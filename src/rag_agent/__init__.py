"""rag-agent: a knowledge-base question-answering agent.

A hand-written RAG core (chunking, vector search, BM25, hybrid retrieval,
fusion, reranking, agentic tool loop, evaluation) wired to pluggable
OpenAI-compatible providers.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
