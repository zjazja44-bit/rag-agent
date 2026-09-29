"""Document loaders: PDF / DOCX / Markdown / plain text -> :class:`Document`."""
from __future__ import annotations

import hashlib
from pathlib import Path

from ..logging_utils import get_logger
from ..types import Document

logger = get_logger(__name__)

SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown", ".pdf", ".docx"}


def _doc_id(path: Path, text: str) -> str:
    h = hashlib.md5()
    h.update(str(path.resolve()).encode("utf-8", "ignore"))
    h.update(b"\x00")
    h.update(text.encode("utf-8", "ignore"))
    return h.hexdigest()[:16]


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "") for page in reader.pages]
    return "\n\n".join(pages)


def _read_docx(path: Path) -> str:
    import docx

    document = docx.Document(str(path))
    return "\n".join(p.text for p in document.paragraphs)


def load_file(path: str | Path) -> list[Document]:
    """Load a single supported file into a list of documents (usually one)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    suffix = p.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type {suffix!r} for {p}")

    if suffix == ".pdf":
        text = _read_pdf(p)
    elif suffix == ".docx":
        text = _read_docx(p)
    else:
        text = _read_text(p)

    if not text.strip():
        logger.warning("Empty document after loading: %s", p)
    return [Document(id=_doc_id(p, text), text=text, source=str(p))]


def load_path(path: str | Path) -> list[Document]:
    """Load a file, or every supported file under a directory (recursive)."""
    p = Path(path)
    if p.is_file():
        return load_file(p)
    if not p.is_dir():
        raise FileNotFoundError(p)

    documents: list[Document] = []
    for fp in sorted(p.rglob("*")):
        if fp.is_file() and fp.suffix.lower() in SUPPORTED_SUFFIXES:
            try:
                documents.extend(load_file(fp))
            except Exception as exc:  # keep going on a single bad file
                logger.warning("Failed to load %s: %s", fp, exc)
    return documents
