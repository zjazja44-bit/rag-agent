"""Ingestion: loading, splitting, embedding and indexing documents."""
from .loader import load_file, load_path
from .pipeline import IngestionPipeline, split_document
from .splitter import split_text

__all__ = [
    "load_file",
    "load_path",
    "split_text",
    "split_document",
    "IngestionPipeline",
]
