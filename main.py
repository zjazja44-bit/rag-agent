"""Convenience entry point so `python main.py ...` works without installing.

Adds ``src/`` to the path and delegates to the packaged CLI.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from rag_agent.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
