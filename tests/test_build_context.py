"""Tests for end-to-end retrieval context construction."""

import pytest

from labelrag import RAGPipeline, RAGPipelineConfig


def test_build_context_requires_fit() -> None:
    """Context building should fail clearly before the pipeline is fitted."""

    pipeline = RAGPipeline(RAGPipelineConfig())

    with pytest.raises(RuntimeError, match="requires fit"):
        pipeline.build_context("How do developers use language models?")


def test_build_context_returns_prompt_and_metadata() -> None:
    """Context building should return retrieval results, prompt text, and trace metadata."""

    pipeline = RAGPipeline(RAGPipelineConfig())
    pipeline.fit(
        [
            "OpenAI builds language models for developers.",
            "Developers use language models in production systems.",
            "Production systems need monitoring and evaluation tooling.",
        ]
    )

    result = pipeline.build_context("How do developers use language models?")

    assert result.question == "How do developers use language models?"
    assert result.query_analysis.query_text == result.question
    assert result.metadata["retrieval_strategy"] == "greedy_label_coverage"
    assert result.metadata["query_label_ids"] == result.query_analysis.label_ids
    assert result.metadata["retrieval_limit"] == pipeline.config.retrieval.max_paragraphs
    assert "covered_label_ids" in result.metadata
    assert "uncovered_label_ids" in result.metadata
    assert result.prompt_context
    assert "[Paragraph 1" in result.prompt_context


def test_build_context_respects_prompt_configuration() -> None:
    """Context building should honor the prompt rendering configuration."""

    config = RAGPipelineConfig()
    config.prompt.include_paragraph_ids = False
    config.prompt.max_context_characters = 20
    pipeline = RAGPipeline(config)
    pipeline.fit(
        [
            "OpenAI builds language models for developers.",
            "Developers use language models in production systems.",
        ]
    )

    result = pipeline.build_context("How do developers use language models?")

    assert result.prompt_context.startswith("[Paragraph 1]")
    assert "| id=" not in result.prompt_context
    assert len(result.prompt_context) <= 20


def test_build_context_returns_empty_retrieval_for_label_free_query() -> None:
    """Queries with no fitted labels should still return a structured empty result."""

    pipeline = RAGPipeline(RAGPipelineConfig())
    pipeline.fit(
        [
            "OpenAI builds language models for developers.",
            "Developers use language models in production systems.",
        ]
    )

    result = pipeline.build_context("Quantum batteries improve starship reactors.")

    assert result.retrieved_paragraphs == []
    assert result.metadata["covered_label_ids"] == []
    assert result.metadata["uncovered_label_ids"] == []
    assert result.prompt_context == ""
