"""Loading and splitting tests (no network, no store)."""
from pathlib import Path

from rag_agent.ingest.loader import load_file, load_path
from rag_agent.ingest.splitter import split_text

DATA = Path(__file__).resolve().parent.parent / "data" / "docs"


def test_short_text_single_chunk():
    assert split_text("很短的一句话。", chunk_size=200, overlap=20) == ["很短的一句话。"]


def test_long_text_splits_within_bound():
    text = "句子一。句子二。句子三。" * 40
    chunks = split_text(text, chunk_size=100, overlap=20)
    assert len(chunks) >= 2
    assert all(c.strip() for c in chunks)
    assert all(len(c) <= 100 + 20 for c in chunks)


def test_overlap_present_between_neighbours():
    text = "".join(f"第{i}句话内容。 " for i in range(50))
    chunks = split_text(text, chunk_size=120, overlap=30, min_chunk_size=10)
    assert len(chunks) >= 2
    # tail of chunk i should reappear at the head of chunk i+1
    assert chunks[0][-30:].strip()[:8] in chunks[1]


def test_fixed_strategy_windows():
    chunks = split_text("abcdefghij", chunk_size=4, overlap=0, min_chunk_size=0, strategy="fixed")
    assert chunks == ["abcd", "efgh", "ij"]


def test_load_directory_returns_documents():
    docs = load_path(DATA)
    assert len(docs) >= 3
    assert all(d.text.strip() for d in docs)
    assert {Path(d.source).name for d in docs} >= {
        "rag_basics.md",
        "vector_search.md",
        "llm_agent.md",
    }


def test_load_single_file_and_doc_id_stable():
    a = load_file(DATA / "rag_basics.md")[0]
    b = load_file(DATA / "rag_basics.md")[0]
    assert a.id == b.id  # deterministic id for idempotent re-ingest
