"""Evaluation metric computation for labelrag pipeline retrieval."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from labelrag.types import RetrievalResult


@dataclass(slots=True)
class RetrievalMetrics:
    """Aggregated retrieval evaluation metrics across queries."""

    label_coverage_rate: float | None
    precision_at_k: dict[int, float]
    recall_at_k: dict[int, float]
    mrr: float
    ndcg_at_k: dict[int, float]
    map_score: float
    avg_semantic_similarity: float | None
    num_queries: int


def compute_metrics(
    results: Sequence[RetrievalResult],
    *,
    k_values: tuple[int, ...] = (1, 3, 5, 10),
    use_binary_relevance: bool = True,
    relevance_threshold: int = 1,
    relevance_judgments: dict[str, set[str]] | None = None,
) -> RetrievalMetrics:
    """Compute aggregated retrieval metrics from a sequence of retrieval results."""
    coverage_rates: list[float] = []
    precision_sums: dict[int, float] = {k: 0.0 for k in k_values}
    recall_sums: dict[int, float] = {k: 0.0 for k in k_values}
    mrr_scores: list[float] = []
    ndcg_sums: dict[int, float] = {k: 0.0 for k in k_values}
    ap_scores: list[float] = []
    similarities: list[float] = []

    for result in results:
        query_label_ids: list[str] = result.metadata.get("query_label_ids", [])
        covered_label_ids: list[str] = result.metadata.get("covered_label_ids", [])

        coverage = _label_coverage_rate(query_label_ids, covered_label_ids)
        if coverage is not None:
            coverage_rates.append(coverage)

        paragraphs = result.retrieved_paragraphs
        retrieved_ids = [p.paragraph_id for p in paragraphs]

        relevant_set = _compute_relevance_set(
            retrieved_ids=retrieved_ids,
            paragraph_labels_by_id={
                p.paragraph_id: p.paragraph_label_ids for p in paragraphs
            },
            query_label_ids=query_label_ids,
            relevance_judgments=relevance_judgments,
            result_question=result.question,
            threshold=relevance_threshold,
        )

        if relevant_set:
            relevance_grades = _build_relevance_grades(
                retrieved_ids=retrieved_ids,
                paragraph_labels_by_id={
                    p.paragraph_id: p.paragraph_label_ids for p in paragraphs
                },
                query_label_ids=query_label_ids,
                relevance_judgments=relevance_judgments,
                result_question=result.question,
                use_binary=use_binary_relevance,
            )
        else:
            relevance_grades = {pid: 0 for pid in retrieved_ids}

        for k in k_values:
            precision_sums[k] += _precision_at_k(retrieved_ids, relevant_set, k)
            recall_sums[k] += _recall_at_k(retrieved_ids, relevant_set, k)
            ndcg_sums[k] += _ndcg_at_k(relevance_grades, k)

        mrr_scores.append(_mrr(retrieved_ids, relevant_set))
        ap_scores.append(_average_precision(retrieved_ids, relevant_set))

        for p in paragraphs:
            if p.semantic_similarity is not None:
                similarities.append(p.semantic_similarity)

    n = len(results)
    return RetrievalMetrics(
        label_coverage_rate=_safe_mean(coverage_rates) if coverage_rates else None,
        precision_at_k={k: precision_sums[k] / n for k in k_values},
        recall_at_k={k: recall_sums[k] / n for k in k_values},
        mrr=_safe_mean(mrr_scores),
        ndcg_at_k={k: ndcg_sums[k] / n for k in k_values},
        map_score=_safe_mean(ap_scores),
        avg_semantic_similarity=_safe_mean(similarities) if similarities else None,
        num_queries=n,
    )


def _label_coverage_rate(
    query_label_ids: list[str],
    covered_label_ids: list[str],
) -> float | None:
    """Compute label coverage rate for a single query."""
    if not query_label_ids:
        return None
    return len(set(covered_label_ids) & set(query_label_ids)) / len(query_label_ids)


def _precision_at_k(
    retrieved_ids: list[str],
    relevant_set: set[str],
    k: int,
) -> float:
    """Compute Precision@k for a single query."""
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    return len(set(top_k) & relevant_set) / len(top_k)


def _recall_at_k(
    retrieved_ids: list[str],
    relevant_set: set[str],
    k: int,
) -> float:
    """Compute Recall@k for a single query."""
    if not relevant_set:
        return 0.0
    top_k = retrieved_ids[:k]
    return len(set(top_k) & relevant_set) / len(relevant_set)


def _mrr(
    retrieved_ids: list[str],
    relevant_set: set[str],
) -> float:
    """Compute Mean Reciprocal Rank for a single query."""
    for i, pid in enumerate(retrieved_ids, start=1):
        if pid in relevant_set:
            return 1.0 / i
    return 0.0


def _ndcg_at_k(
    relevance_grades: dict[str, int],
    k: int,
) -> float:
    """Compute NDCG@k for a single query."""
    if not relevance_grades:
        return 0.0
    sorted_grades = sorted(relevance_grades.values(), reverse=True)
    ideal_gains = sorted_grades[:k]
    ideal_dcg = sum(
        gain / math.log2(i + 2) for i, gain in enumerate(ideal_gains)
    )
    if ideal_dcg == 0.0:
        return 0.0
    retrieved_ids = list(relevance_grades.keys())[:k]
    actual_gains = [relevance_grades.get(pid, 0) for pid in retrieved_ids]
    actual_dcg = sum(
        gain / math.log2(i + 2) for i, gain in enumerate(actual_gains)
    )
    return actual_dcg / ideal_dcg


def _average_precision(
    retrieved_ids: list[str],
    relevant_set: set[str],
) -> float:
    """Compute Average Precision for a single query."""
    if not relevant_set:
        return 0.0
    num_hits = 0
    sum_precisions = 0.0
    for i, pid in enumerate(retrieved_ids, start=1):
        if pid in relevant_set:
            num_hits += 1
            sum_precisions += num_hits / i
    return sum_precisions / len(relevant_set)


def _compute_relevance_set(
    retrieved_ids: list[str],
    paragraph_labels_by_id: dict[str, list[str]],
    query_label_ids: list[str],
    relevance_judgments: dict[str, set[str]] | None,
    result_question: str,
    threshold: int,
) -> set[str]:
    """Determine which paragraph IDs are relevant to a query."""
    if relevance_judgments is not None and result_question in relevance_judgments:
        return relevance_judgments[result_question] & set(retrieved_ids)

    if not query_label_ids:
        return set()

    query_labels = set(query_label_ids)
    relevant: set[str] = set()
    for pid in retrieved_ids:
        overlap = len(set(paragraph_labels_by_id.get(pid, [])) & query_labels)
        if overlap >= threshold:
            relevant.add(pid)
    return relevant


def _build_relevance_grades(
    retrieved_ids: list[str],
    paragraph_labels_by_id: dict[str, list[str]],
    query_label_ids: list[str],
    relevance_judgments: dict[str, set[str]] | None,
    result_question: str,
    use_binary: bool,
) -> dict[str, int]:
    """Build relevance grade dict for retrieved paragraphs."""
    if relevance_judgments is not None and result_question in relevance_judgments:
        relevant = relevance_judgments[result_question]
        return {pid: (1 if pid in relevant else 0) for pid in retrieved_ids}

    query_labels = set(query_label_ids)
    grades: dict[str, int] = {}
    for pid in retrieved_ids:
        overlap = len(set(paragraph_labels_by_id.get(pid, [])) & query_labels)
        grades[pid] = 1 if (use_binary and overlap > 0) else overlap
    return grades


def _safe_mean(values: list[float]) -> float:
    """Compute the mean of a non-empty list of floats."""
    return sum(values) / len(values)
