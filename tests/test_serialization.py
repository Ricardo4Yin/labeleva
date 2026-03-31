"""Tests for pipeline save/load behavior."""

from pathlib import Path

from labelrag import RAGPipeline, RAGPipelineConfig


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
    assert (output_dir / "corpus_index.json").is_file()
