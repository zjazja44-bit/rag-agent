"""Load evaluation datasets (JSONL or JSON array)."""
from __future__ import annotations

import json
from pathlib import Path


def load_dataset(path: str | Path) -> list[dict]:
    p = Path(path)
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if p.suffix == ".jsonl" or "\n{" in text or text.startswith("{"):
        items = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                items.append(json.loads(line))
        return items

    data = json.loads(text)
    return list(data) if isinstance(data, list) else []
