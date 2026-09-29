"""Evaluation: retrieval metrics, answer heuristics and a runner."""
from .dataset import load_dataset
from .metrics import (
    answer_coverage,
    citation_rate,
    groundedness,
    hit_rate,
    judge_faithfulness,
    mrr,
    precision_at_k,
    recall_at_k,
)
from .runner import EvalReport, EvalRow, evaluate, print_report, run_eval

__all__ = [
    "load_dataset",
    "hit_rate",
    "recall_at_k",
    "precision_at_k",
    "mrr",
    "groundedness",
    "answer_coverage",
    "citation_rate",
    "judge_faithfulness",
    "evaluate",
    "run_eval",
    "print_report",
    "EvalReport",
    "EvalRow",
]
