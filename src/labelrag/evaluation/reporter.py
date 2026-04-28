"""Report formatter for retrieval evaluation results."""

from __future__ import annotations

from labelrag.evaluation.metrics import RetrievalMetrics

_WIDTH = 52


def format_report(metrics: RetrievalMetrics) -> str:
    """Format RetrievalMetrics as a human-readable bordered report."""
    lines: list[str] = []
    lines.append("┌" + "─" * _WIDTH + "┐")
    lines.append(_center("Retrieval Evaluation Report", _WIDTH))
    lines.append("├" + "─" * _WIDTH + "┤")
    lines.append(_row("Number of Queries", str(metrics.num_queries)))

    _append_metric(lines, "Label Coverage Rate", metrics.label_coverage_rate)

    for k, v in metrics.precision_at_k.items():
        _append_metric(lines, f"Precision@{k}", v)
    for k, v in metrics.recall_at_k.items():
        _append_metric(lines, f"Recall@{k}", v)

    _append_metric(lines, "MRR", metrics.mrr)

    for k, v in metrics.ndcg_at_k.items():
        _append_metric(lines, f"NDCG@{k}", v)

    _append_metric(lines, "MAP", metrics.map_score)
    _append_metric(lines, "Avg Semantic Similarity", metrics.avg_semantic_similarity)

    lines.append("└" + "─" * _WIDTH + "┘")
    return "\n".join(lines)


def _append_metric(
    lines: list[str],
    label: str,
    value: float | None,
) -> None:
    """Append a formatted metric row to the report lines."""
    if value is None:
        formatted = "N/A"
    else:
        formatted = f"{value:.4f}"
    lines.append(_row(label, formatted))


def _row(label: str, value: str) -> str:
    """Format a single report row with left label and right value."""
    return f"│  {label:<36}{value:>12}  │"


def _center(text: str, width: int) -> str:
    """Center text within a bordered row."""
    return f"│{text:^{width}}│"
