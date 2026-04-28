"""Evaluator protocol and default implementation for retrieval evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from labelrag.evaluation.metrics import RetrievalMetrics, compute_metrics
from labelrag.pipeline.rag_pipeline import RAGPipeline


@dataclass(slots=True)
class EvalConfig:
    """Configuration for retrieval evaluation."""

    k_values: tuple[int, ...] = (1, 3, 5, 10)
    use_binary_relevance: bool = True
    relevance_threshold: int = 1


class RetrievalEvaluator(Protocol):
    """Protocol for retrieval evaluators."""

    def evaluate(
        self,
        pipeline: RAGPipeline,
        queries: list[str],
        *,
        relevance_judgments: dict[str, set[str]] | None = None,
    ) -> RetrievalMetrics:
        """Evaluate retrieval performance on a fitted pipeline."""
        ...


class DefaultRetrievalEvaluator:
    """Default retrieval evaluator using label-based relevance."""

    def __init__(self, config: EvalConfig | None = None) -> None:
        self.config = config or EvalConfig()

    def evaluate(
        self,
        pipeline: RAGPipeline,
        queries: list[str],
        *,
        relevance_judgments: dict[str, set[str]] | None = None,
    ) -> RetrievalMetrics:
        """Run retrieval for each query and compute aggregated metrics."""
        results = [pipeline.build_context(q) for q in queries]
        return compute_metrics(
            results,
            k_values=self.config.k_values,
            use_binary_relevance=self.config.use_binary_relevance,
            relevance_threshold=self.config.relevance_threshold,
            relevance_judgments=relevance_judgments,
        )
