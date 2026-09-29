"""Evaluation runner: score a RAG system against a labelled question set."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Sequence

from ..logging_utils import get_logger
from . import metrics
from .dataset import load_dataset

logger = get_logger(__name__)


@dataclass
class EvalRow:
    question: str
    answer: str
    retrieved_sources: list[str]
    relevant_docs: list[str]
    hit: float = 0.0
    recall: float = 0.0
    precision: float = 0.0
    mrr: float = 0.0
    groundedness: float = 0.0
    coverage: float = 0.0
    citation: float = 0.0


@dataclass
class EvalReport:
    mode: str
    rows: list[EvalRow] = field(default_factory=list)
    aggregate: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "aggregate": self.aggregate,
            "rows": [asdict(r) for r in self.rows],
        }


def _mean(values: Sequence[float]) -> float:
    return round(mean(values), 4) if values else 0.0


def evaluate(app, items: Sequence[dict], mode: str = "rag", top_k: int | None = None) -> EvalReport:
    k = top_k or app.config.retrieval.top_k
    rows: list[EvalRow] = []

    for item in items:
        question = item["question"]
        relevant = list(item.get("relevant_docs", []))

        if mode == "agent":
            result = app.agent.run(question)
            answer, context, sources = result.answer, "", []
        else:
            result = app.chain.answer(question, top_k=k)
            answer = result.answer
            context = result.context
            sources = [Path(h.chunk.source).name for h in result.hits]

        rows.append(
            EvalRow(
                question=question,
                answer=answer,
                retrieved_sources=sources,
                relevant_docs=relevant,
                hit=metrics.hit_rate(sources, relevant, k),
                recall=metrics.recall_at_k(sources, relevant, k),
                precision=metrics.precision_at_k(sources, relevant, k),
                mrr=metrics.mrr(sources, relevant),
                groundedness=metrics.groundedness(answer, context),
                coverage=metrics.answer_coverage(answer, item.get("answer_keywords", [])),
                citation=metrics.citation_rate(answer),
            )
        )

    aggregate = {
        "hit_rate": _mean([r.hit for r in rows]),
        "recall": _mean([r.recall for r in rows]),
        "precision": _mean([r.precision for r in rows]),
        "mrr": _mean([r.mrr for r in rows]),
        "groundedness": _mean([r.groundedness for r in rows]),
        "answer_coverage": _mean([r.coverage for r in rows]),
        "citation_rate": _mean([r.citation for r in rows]),
    }
    return EvalReport(mode=mode, rows=rows, aggregate=aggregate)


def run_eval(app, dataset_path: str | Path, mode: str = "rag", top_k: int | None = None,
             verbose: bool = True) -> dict:
    items = load_dataset(dataset_path)
    if not items:
        logger.warning("Empty dataset: %s", dataset_path)
    report = evaluate(app, items, mode=mode, top_k=top_k)
    if verbose:
        print_report(report)
    return report.to_dict()


def print_report(report: EvalReport) -> None:
    print(f"\n评测报告（mode={report.mode}，样本数={len(report.rows)}）")
    print("-" * 52)
    for key, value in report.aggregate.items():
        print(f"  {key:<16} {value:.4f}")
    print("-" * 52)
    print("逐题（问题 / 命中 / 覆盖率）：")
    for row in report.rows:
        print(f"  [hit={row.hit:.0f} cov={row.coverage:.2f}] {row.question}")
