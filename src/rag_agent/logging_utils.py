"""Tiny logging helper so every module logs the same way."""
from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def setup_logging(level: str = "INFO") -> None:
    """Configure the root logger once, with a compact one-line format."""
    global _CONFIGURED
    if _CONFIGURED:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s", "%H:%M:%S")
    )
    root = logging.getLogger()
    root.setLevel(level.upper())
    root.addHandler(handler)

    # Third-party libs are chatty at INFO; keep our own logs, silence theirs.
    for noisy in ("httpx", "httpcore", "openai", "huggingface_hub", "urllib3", "fastembed"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
