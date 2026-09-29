"""Text splitting: a recursive, structure-aware splitter.

``recursive`` walks a separator hierarchy (paragraph -> line -> sentence ->
clause -> character) and only falls back to a harder cut when a piece is still
too big, so sentence boundaries are preserved wherever possible. Adjacent
pieces are then greedily merged up to ``chunk_size`` with a character ``overlap``
so context that straddles a boundary is not lost.
"""
from __future__ import annotations

SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", ". ", "! ", "? ", "；", "，", ", ", " ", ""]


def split_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
    min_chunk_size: int = 64,
    strategy: str = "recursive",
) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if chunk_size <= 0:
        return [text]

    if strategy == "fixed":
        chunks = _fixed_windows(text, chunk_size, overlap)
    else:
        pieces = _split_to_pieces(text, SEPARATORS, chunk_size)
        chunks = _merge_pieces(pieces, chunk_size, overlap)

    return _postprocess(chunks, min_chunk_size)


def _fixed_windows(text: str, chunk_size: int, overlap: int) -> list[str]:
    step = max(1, chunk_size - overlap)
    return [text[i : i + chunk_size] for i in range(0, len(text), step)]


def _split_to_pieces(text: str, separators: list[str], chunk_size: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    sep = ""
    for candidate in separators:
        if candidate == "" or candidate in text:
            sep = candidate
            break

    if sep == "":
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    parts = text.split(sep)
    # Re-attach the separator to each part so we don't lose punctuation.
    rejoined = [p + sep for p in parts[:-1]] + [parts[-1]]

    out: list[str] = []
    for piece in rejoined:
        if piece == "":
            continue
        if len(piece) <= chunk_size:
            out.append(piece)
        else:
            out.extend(_split_to_pieces(piece, separators, chunk_size))
    return out


def _merge_pieces(pieces: list[str], chunk_size: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) > chunk_size:
            chunks.append(current)
            current = current[-overlap:] if overlap > 0 else ""
        current += piece
    if current.strip():
        chunks.append(current)
    return chunks


def _postprocess(chunks: list[str], min_chunk_size: int) -> list[str]:
    cleaned = [c.strip() for c in chunks if c and c.strip()]
    # Fold a too-small trailing chunk back into its neighbour.
    if min_chunk_size and len(cleaned) >= 2 and len(cleaned[-1]) < min_chunk_size:
        cleaned[-2] = f"{cleaned[-2]}\n{cleaned[-1]}"
        cleaned.pop()
    return cleaned
