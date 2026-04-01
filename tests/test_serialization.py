"""Tests for pipeline save/load behavior."""

from dataclasses import dataclass
from pathlib import Path

from labelrag import RAGPipeline, RAGPipelineConfig
from labelrag.generation.generator import GeneratedAnswer


@dataclass
class StubGenerator:
    """Simple generator stub for load-time answer integration tests."""

    def generate(self, question: str, context: str) -> GeneratedAnswer:
        """Return a deterministic generated answer."""

        return GeneratedAnswer(
            text=f"answer for: {question}",
            metadata={"model": "stub-model", "has_context": bool(context)},
        )


def test_save_and_load_preserve_build_context_behavior(tmp_path: Path) -> None:
    """A saved and loaded pipeline should preserve retrieval behavior."""

    pipeline = RAGPipeline(RAGPipelineConfig())
    pipeline.fit(
        [
            "OpenAI builds language models for developers.",
            "Developers use language models in production systems.",
            "Production systems need monitoring and evaluation tooling.",
        ]
    )

    original = pipeline.build_context("How do developers use language models?")
    output_dir = tmp_path / "pipeline"
    pipeline.save(output_dir)

    loaded = RAGPipeline.load(output_dir)
    restored = loaded.build_context("How do developers use language models?")

    assert loaded.fit_result == pipeline.fit_result
    assert restored.query_analysis == original.query_analysis
    assert restored.retrieved_paragraphs == original.retrieved_paragraphs
    assert restored.prompt_context == original.prompt_context
    assert restored.metadata == original.metadata


def test_save_writes_expected_files(tmp_path: Path) -> None:
    """Saving should produce the expected human-inspectable file layout."""

    pipeline = RAGPipeline(RAGPipelineConfig())
    pipeline.fit(
        [
            "OpenAI builds language models for developers.",
            "Developers use language models in production systems.",
        ]
    )

    output_dir = tmp_path / "pipeline"
    pipeline.save(output_dir)

    assert (output_dir / "config.json").is_file()
    assert (output_dir / "label_generator.json").is_file()
    assert (output_dir / "fit_result.json").is_file()
    assert (output_dir / "corpus_index.json").is_file()


def test_load_supports_answer_with_generator(tmp_path: Path) -> None:
    """A loaded pipeline should still support end-to-end answer generation."""

    pipeline = RAGPipeline(RAGPipelineConfig())
    pipeline.fit(
        [
            "OpenAI builds language models for developers.",
            "Developers use language models in production systems.",
            "Production systems need monitoring and evaluation tooling.",
        ]
    )

    output_dir = tmp_path / "pipeline"
    pipeline.save(output_dir)

    loaded = RAGPipeline.load(output_dir)
    result = loaded.answer_with_generator(
        "How do developers use language models?",
        StubGenerator(),
    )

    assert result.answer_text == "answer for: How do developers use language models?"
    assert result.metadata["generator_name"] == "StubGenerator"
    assert result.metadata["generation_model"] == "stub-model"
