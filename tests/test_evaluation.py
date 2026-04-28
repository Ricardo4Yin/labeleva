"""Tests for retrieval evaluation metrics and evaluator."""

from __future__ import annotations

from labelrag.evaluation.evaluator import DefaultRetrievalEvaluator, EvalConfig
from labelrag.evaluation.metrics import (
    RetrievalMetrics,
    _average_precision,
    _build_relevance_grades,
    _compute_relevance_set,
    _label_coverage_rate,
    _mrr,
    _ndcg_at_k,
    _precision_at_k,
    _recall_at_k,
    _safe_mean,
    compute_metrics,
)
from labelrag.evaluation.reporter import format_report


# ── Label Coverage Rate ──────────────────────────────────────────────

def test_label_coverage_full() -> None:
    assert _label_coverage_rate(["a", "b", "c"], ["a", "b", "c"]) == 1.0


def test_label_coverage_partial() -> None:
    assert _label_coverage_rate(["a", "b", "c"], ["a"]) == 1.0 / 3.0


def test_label_coverage_no_labels() -> None:
    assert _label_coverage_rate([], ["a"]) is None


def test_label_coverage_no_covered() -> None:
    assert _label_coverage_rate(["a", "b"], []) == 0.0


# ── Precision@k ──────────────────────────────────────────────────────

def test_precision_at_k_all_relevant() -> None:
    assert _precision_at_k(["a", "b", "c"], {"a", "b", "c"}, 3) == 1.0


def test_precision_at_k_partial() -> None:
    assert _precision_at_k(["a", "b", "c", "d"], {"a", "c"}, 3) == 2.0 / 3.0


def test_precision_at_k_empty_retrieval() -> None:
    assert _precision_at_k([], {"a", "b"}, 3) == 0.0


def test_precision_at_k_zero_k() -> None:
    assert _precision_at_k(["a", "b"], {"a"}, 0) == 0.0


# ── Recall@k ─────────────────────────────────────────────────────────

def test_recall_at_k_full() -> None:
    assert _recall_at_k(["a", "b", "c"], {"a", "b", "c"}, 5) == 1.0


def test_recall_at_k_partial() -> None:
    assert _recall_at_k(["a", "b"], {"a", "b", "c", "d"}, 2) == 0.5


def test_recall_at_k_zero_relevant() -> None:
    assert _recall_at_k(["a", "b"], set(), 3) == 0.0


# ── MRR ──────────────────────────────────────────────────────────────

def test_mrr_first_rank() -> None:
    assert _mrr(["a", "b", "c"], {"a"}) == 1.0


def test_mrr_second_rank() -> None:
    assert _mrr(["a", "b", "c"], {"b"}) == 0.5


def test_mrr_no_relevant() -> None:
    assert _mrr(["a", "b"], {"c"}) == 0.0


# ── NDCG@k ───────────────────────────────────────────────────────────

def test_ndcg_perfect_ordering() -> None:
    grades = {"a": 3, "b": 2, "c": 1}
    assert _ndcg_at_k(grades, 3) == 1.0


def test_ndcg_zero_idcg() -> None:
    assert _ndcg_at_k({"a": 0, "b": 0}, 2) == 0.0


def test_ndcg_empty() -> None:
    assert _ndcg_at_k({}, 3) == 0.0


# ── Average Precision ────────────────────────────────────────────────

def test_ap_all_relevant() -> None:
    assert _average_precision(["a", "b", "c"], {"a", "b", "c"}) == 1.0


def test_ap_interleaved() -> None:
    score = _average_precision(["a", "x", "b", "y"], {"a", "b"})
    expected = (1 / 1 + 2 / 3) / 2
    assert abs(score - expected) < 1e-9


def test_ap_no_relevant() -> None:
    assert _average_precision(["a", "b"], set()) == 0.0


# ── Safe Mean ────────────────────────────────────────────────────────

def test_safe_mean_normal() -> None:
    assert _safe_mean([1.0, 2.0, 3.0]) == 2.0


# ── Relevance Set Computation ────────────────────────────────────────

def test_relevance_set_from_labels() -> None:
    result = _compute_relevance_set(
        retrieved_ids=["a", "b", "c"],
        paragraph_labels_by_id={"a": ["L1"], "b": ["L2"], "c": ["L3"]},
        query_label_ids=["L1", "L3"],
        relevance_judgments=None,
        result_question="q1",
        threshold=1,
    )
    assert result == {"a", "c"}


def test_relevance_set_threshold() -> None:
    result = _compute_relevance_set(
        retrieved_ids=["a", "b"],
        paragraph_labels_by_id={"a": ["L1", "L2"], "b": ["L1"]},
        query_label_ids=["L1", "L2"],
        relevance_judgments=None,
        result_question="q1",
        threshold=2,
    )
    assert result == {"a"}


def test_relevance_set_explicit_override() -> None:
    result = _compute_relevance_set(
        retrieved_ids=["a", "b", "c"],
        paragraph_labels_by_id={"a": ["L1"], "b": ["L2"], "c": []},
        query_label_ids=["L1", "L2"],
        relevance_judgments={"q1": {"b", "c"}},
        result_question="q1",
        threshold=1,
    )
    assert result == {"b", "c"}


def test_relevance_set_no_labels() -> None:
    result = _compute_relevance_set(
        retrieved_ids=["a", "b"],
        paragraph_labels_by_id={},
        query_label_ids=[],
        relevance_judgments=None,
        result_question="q1",
        threshold=1,
    )
    assert result == set()


# ── Relevance Grades ─────────────────────────────────────────────────

def test_build_grades_binary() -> None:
    grades = _build_relevance_grades(
        retrieved_ids=["a", "b", "c"],
        paragraph_labels_by_id={"a": ["L1"], "b": [], "c": ["L2"]},
        query_label_ids=["L1", "L2"],
        relevance_judgments=None,
        result_question="q1",
        use_binary=True,
    )
    assert grades == {"a": 1, "b": 0, "c": 1}


def test_build_grades_graded() -> None:
    grades = _build_relevance_grades(
        retrieved_ids=["a", "b"],
        paragraph_labels_by_id={"a": ["L1", "L2"], "b": ["L1"]},
        query_label_ids=["L1", "L2"],
        relevance_judgments=None,
        result_question="q1",
        use_binary=False,
    )
    assert grades == {"a": 2, "b": 1}


def test_build_grades_explicit() -> None:
    grades = _build_relevance_grades(
        retrieved_ids=["a", "b"],
        paragraph_labels_by_id={},
        query_label_ids=[],
        relevance_judgments={"q1": {"a"}},
        result_question="q1",
        use_binary=True,
    )
    assert grades == {"a": 1, "b": 0}


# ── compute_metrics integration ──────────────────────────────────────

def _make_result(
    question: str,
    paragraph_ids: list[str],
    paragraph_label_ids_list: list[list[str]],
    similarities: list[float | None],
    query_label_ids: list[str],
    covered_label_ids: list[str],
) -> object:
    """Build a minimal RetrievalResult-like object for testing."""
    from labelrag.types import QueryAnalysis, RetrievedParagraph, RetrievalResult

    paragraphs = [
        RetrievedParagraph(
            paragraph_id=pid,
            text="",
            metadata=None,
            newly_covered_label_ids=[],
            already_covered_label_ids=[],
            matched_label_ids=pls,
            matched_concept_ids=[],
            paragraph_label_ids=pls,
            paragraph_concept_ids=[],
            concept_overlap_count=0,
            marginal_gain=0,
            semantic_similarity=sim,
            retrieval_score=0.0,
        )
        for pid, pls, sim in zip(paragraph_ids, paragraph_label_ids_list, similarities)
    ]

    return RetrievalResult(
        question=question,
        query_analysis=QueryAnalysis(
            query_text=question,
            concepts=[],
            concept_ids=[],
            label_ids=query_label_ids,
            label_display_names=[],
        ),
        retrieved_paragraphs=paragraphs,
        prompt_context="",
        metadata={
            "query_label_ids": query_label_ids,
            "covered_label_ids": covered_label_ids,
        },
    )


def test_compute_metrics_standard() -> None:
    r1 = _make_result(
        question="q1",
        paragraph_ids=["a", "b"],
        paragraph_label_ids_list=[["L1"], ["L1", "L2"]],
        similarities=[0.9, 0.8],
        query_label_ids=["L1", "L2"],
        covered_label_ids=["L1", "L2"],
    )
    r2 = _make_result(
        question="q2",
        paragraph_ids=["c"],
        paragraph_label_ids_list=[["L1"]],
        similarities=[0.5],
        query_label_ids=["L1"],
        covered_label_ids=["L1"],
    )

    metrics = compute_metrics([r1, r2], k_values=(1, 3))

    assert metrics.num_queries == 2
    assert metrics.label_coverage_rate == 1.0
    assert metrics.precision_at_k[1] == 1.0
    assert metrics.mrr == 1.0
    assert metrics.map_score == 1.0
    assert metrics.avg_semantic_similarity is not None
    assert metrics.avg_semantic_similarity > 0.0


def test_compute_metrics_no_labels() -> None:
    r = _make_result(
        question="q1",
        paragraph_ids=["a"],
        paragraph_label_ids_list=[[]],
        similarities=[0.5],
        query_label_ids=[],
        covered_label_ids=[],
    )
    metrics = compute_metrics([r])

    assert metrics.label_coverage_rate is None


def test_compute_metrics_explicit_judgments() -> None:
    r = _make_result(
        question="q1",
        paragraph_ids=["a", "b", "c"],
        paragraph_label_ids_list=[[], [], []],
        similarities=[0.9, 0.8, 0.7],
        query_label_ids=[],
        covered_label_ids=[],
    )
    metrics = compute_metrics(
        [r], k_values=(2,), relevance_judgments={"q1": {"b", "c"}}
    )

    assert metrics.precision_at_k[2] == 0.5


# ── EvalConfig ───────────────────────────────────────────────────────

def test_eval_config_defaults() -> None:
    config = EvalConfig()
    assert config.k_values == (1, 3, 5, 10)
    assert config.use_binary_relevance is True
    assert config.relevance_threshold == 1


def test_eval_config_custom() -> None:
    config = EvalConfig(k_values=(5, 10), relevance_threshold=2)
    assert config.k_values == (5, 10)
    assert config.relevance_threshold == 2


# ── DefaultRetrievalEvaluator integration ────────────────────────────

def test_evaluator_with_pipeline() -> None:
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent / "examples"))
    from _demo_embedding import DemoEmbeddingProvider

    from labelrag import RAGPipeline, RAGPipelineConfig

    paragraphs = [
        "Language models help developers build applications.",
        "Developers use monitoring tools in production systems.",
        "Installation errors can be fixed by checking logs.",
    ]

    config = RAGPipelineConfig()
    config.labelgen.extractor_mode = "heuristic"
    config.labelgen.use_graph_community_detection = False
    config.embedding.normalize = False

    pipeline = RAGPipeline(config, embedding_provider=DemoEmbeddingProvider())
    pipeline.fit(paragraphs)

    evaluator = DefaultRetrievalEvaluator(EvalConfig(k_values=(1, 3)))
    metrics = evaluator.evaluate(
        pipeline,
        ["How do developers use language models?"],
    )

    assert metrics.num_queries == 1
    assert metrics.mrr >= 0.0
    assert metrics.map_score >= 0.0


# ── format_report ────────────────────────────────────────────────────

def test_format_report_contains_sections() -> None:
    metrics = RetrievalMetrics(
        label_coverage_rate=0.75,
        precision_at_k={1: 0.8, 3: 0.6},
        recall_at_k={1: 0.4, 3: 0.7},
        mrr=0.8333,
        ndcg_at_k={1: 0.8, 3: 0.72},
        map_score=0.71,
        avg_semantic_similarity=0.65,
        num_queries=5,
    )
    report = format_report(metrics)

    assert "Retrieval Evaluation Report" in report
    assert "Number of Queries" in report
    assert "0.7500" in report
    assert "0.8333" in report
    assert "0.7100" in report


def test_format_report_none_metrics() -> None:
    metrics = RetrievalMetrics(
        label_coverage_rate=None,
        precision_at_k={},
        recall_at_k={},
        mrr=0.0,
        ndcg_at_k={},
        map_score=0.0,
        avg_semantic_similarity=None,
        num_queries=0,
    )
    report = format_report(metrics)

    assert "N/A" in report
    assert "0.0000" in report
