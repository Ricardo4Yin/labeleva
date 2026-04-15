"""Tests for data loading functionality."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from labelrag import RAGPipeline, RAGPipelineConfig
from labelrag.data import DataFittingHelper, DataLoader, DataLoaderConfig
from labelrag.indexing.corpus_index import CorpusIndex
from support import StubEmbeddingProvider


def test_data_loader_config_validation(tmp_path: Path) -> None:
    """DataLoaderConfig should validate its parameters."""
    # Valid configuration
    valid_path = tmp_path / "test.json"
    valid_path.write_text("{}")
    config = DataLoaderConfig(data_path=valid_path, max_paragraph_length=500, overlap_sentences=1)
    assert config.data_path == valid_path
    assert config.max_paragraph_length == 500
    assert config.overlap_sentences == 1

    # Should accept string path
    config_str = DataLoaderConfig(data_path=str(valid_path))
    assert config_str.data_path == valid_path

    # Should raise RuntimeError for non-existent file
    with pytest.raises(RuntimeError, match="Data file not found"):
        DataLoaderConfig(data_path=tmp_path / "nonexistent.json")

    # Should raise RuntimeError for invalid max_paragraph_length
    with pytest.raises(RuntimeError, match="max_paragraph_length must be positive"):
        DataLoaderConfig(data_path=valid_path, max_paragraph_length=0)

    # Should raise RuntimeError for negative overlap_sentences
    with pytest.raises(RuntimeError, match="overlap_sentences must be non-negative"):
        DataLoaderConfig(data_path=valid_path, overlap_sentences=-1)


def test_data_loader_loads_valid_json(tmp_path: Path) -> None:
    """DataLoader should load valid JSON with TechQA structure."""
    # Create valid TechQA-style JSON
    data = {
        "doc1": {
            "id": "doc1",
            "text": "Artificial intelligence is demonstrated by machines. AI applications include web search.",
            "title": "Introduction to AI",
            "metadata": {"category": "technology", "source": "test"}
        },
        "doc2": {
            "id": "doc2",
            "text": "Machine learning enables computers to learn from data. Deep learning uses neural networks.",
            "title": "Machine Learning Basics",
            "metadata": {"category": "technology", "source": "test"}
        }
    }

    json_path = tmp_path / "test.json"
    json_path.write_text(json.dumps(data, indent=2))

    config = DataLoaderConfig(data_path=json_path)
    loader = DataLoader(config)
    paragraphs = loader.load_paragraphs()

    # Should load paragraphs from both documents
    assert len(paragraphs) >= 2  # Could be more if documents split
    assert all(p.text for p in paragraphs)
    assert all(p.id for p in paragraphs)

    # Check metadata includes doc_id and paragraph_index
    for paragraph in paragraphs:
        assert "doc_id" in paragraph.metadata
        assert "paragraph_index" in paragraph.metadata
        assert paragraph.metadata.get("category") == "technology"


def test_data_loader_handles_invalid_json(tmp_path: Path) -> None:
    """DataLoader should raise RuntimeError for invalid JSON."""
    json_path = tmp_path / "invalid.json"
    json_path.write_text("not valid json")

    config = DataLoaderConfig(data_path=json_path)
    loader = DataLoader(config)

    with pytest.raises(RuntimeError, match="Failed to parse JSON file"):
        loader.load_paragraphs()


def test_data_loader_handles_missing_text(tmp_path: Path) -> None:
    """DataLoader should skip documents without text fields."""
    data = {
        "doc1": {
            "id": "doc1",
            "text": "Valid document with text.",
            "title": "Valid Doc",
            "metadata": {}
        },
        "doc2": {
            "id": "doc2",
            # Missing text field
            "title": "Invalid Doc",
            "metadata": {}
        },
        "doc3": {
            "id": "doc3",
            "text": "",  # Empty text
            "title": "Empty Text Doc",
            "metadata": {}
        },
        "doc4": {
            "id": "doc4",
            "text": "   ",  # Whitespace only
            "title": "Whitespace Doc",
            "metadata": {}
        }
    }

    json_path = tmp_path / "test.json"
    json_path.write_text(json.dumps(data, indent=2))

    config = DataLoaderConfig(data_path=json_path)
    loader = DataLoader(config)
    paragraphs = loader.load_paragraphs()

    # Should only load paragraphs from doc1
    assert len(paragraphs) >= 1
    assert all("doc1" in p.metadata.get("doc_id", "") for p in paragraphs)


def test_data_loader_splits_long_documents(tmp_path: Path) -> None:
    """DataLoader should split long documents into multiple paragraphs."""
    # Create a document with many sentences that will exceed max_paragraph_length
    sentences = [
        f"Sentence {i}: This is a test sentence to create a long document. "
        for i in range(20)
    ]
    long_text = "".join(sentences)

    data = {
        "doc1": {
            "id": "doc1",
            "text": long_text,
            "title": "Long Document",
            "metadata": {"category": "test"}
        }
    }

    json_path = tmp_path / "test.json"
    json_path.write_text(json.dumps(data, indent=2))

    # Use small max_paragraph_length to force splitting
    config = DataLoaderConfig(data_path=json_path, max_paragraph_length=100)
    loader = DataLoader(config)
    paragraphs = loader.load_paragraphs()

    # Should create multiple paragraphs
    assert len(paragraphs) > 1

    # Each paragraph should not exceed max_paragraph_length (plus some slack for spaces)
    for paragraph in paragraphs:
        assert len(paragraph.text) <= config.max_paragraph_length + 50

    # Check paragraph indices are sequential
    para_indices = [p.metadata["paragraph_index"] for p in paragraphs]
    assert para_indices == list(range(len(paragraphs)))


def test_data_fitting_helper_load_paragraphs(tmp_path: Path) -> None:
    """DataFittingHelper.load_paragraphs_from_json should load paragraphs."""
    data = {
        "doc1": {
            "id": "doc1",
            "text": "Test document text for paragraph loading.",
            "title": "Test Doc",
            "metadata": {"source": "test"}
        }
    }

    json_path = tmp_path / "test.json"
    json_path.write_text(json.dumps(data, indent=2))

    # Test with path as string
    paragraphs = DataFittingHelper.load_paragraphs_from_json(str(json_path))
    assert len(paragraphs) >= 1
    assert paragraphs[0].text == "Test document text for paragraph loading."

    # Test with path as Path object
    paragraphs = DataFittingHelper.load_paragraphs_from_json(json_path)
    assert len(paragraphs) >= 1

    # Test with additional parameters
    paragraphs = DataFittingHelper.load_paragraphs_from_json(
        json_path,
        max_paragraph_length=200,
        overlap_sentences=0
    )
    assert len(paragraphs) >= 1


def test_data_fitting_helper_fit_pipeline(tmp_path: Path) -> None:
    """DataFittingHelper.fit_pipeline_from_json should load and fit pipeline."""
    data = {
        "doc1": {
            "id": "doc1",
            "text": "First test document for pipeline fitting by developers.",
            "title": "Doc 1",
            "metadata": {}
        },
        "doc2": {
            "id": "doc2",
            "text": "Second test document for pipeline fitting with language models.",
            "title": "Doc 2",
            "metadata": {}
        }
    }

    json_path = tmp_path / "test.json"
    json_path.write_text(json.dumps(data, indent=2))

    # Create pipeline with stub embedding provider and heuristic extractor
    config = RAGPipelineConfig()
    config.labelgen.extractor_mode = "heuristic"
    config.labelgen.use_graph_community_detection = False
    pipeline = RAGPipeline(config, embedding_provider=StubEmbeddingProvider())

    # Fit pipeline using DataFittingHelper
    fitted_pipeline = DataFittingHelper.fit_pipeline_from_json(
        pipeline=pipeline,
        data_path=json_path,
        max_paragraph_length=500,
        overlap_sentences=1
    )

    # Pipeline should be fitted (able to build context)
    result = fitted_pipeline.build_context("test query")
    assert result.question == "test query"
    assert len(result.retrieved_paragraphs) >= 0


def test_data_fitting_helper_with_loader(tmp_path: Path) -> None:
    """DataFittingHelper.fit_pipeline_with_loader should fit pipeline using DataLoader."""
    data = {
        "doc1": {
            "id": "doc1",
            "text": "Document for loader-based fitting with language models.",
            "title": "Loader Doc",
            "metadata": {"test": True}
        }
    }

    json_path = tmp_path / "test.json"
    json_path.write_text(json.dumps(data, indent=2))

    # Create DataLoader
    config = DataLoaderConfig(data_path=json_path)
    loader = DataLoader(config)

    # Create pipeline with heuristic extractor
    pipeline_config = RAGPipelineConfig()
    pipeline_config.labelgen.extractor_mode = "heuristic"
    pipeline_config.labelgen.use_graph_community_detection = False
    pipeline = RAGPipeline(pipeline_config, embedding_provider=StubEmbeddingProvider())

    # Fit pipeline using loader
    fitted_pipeline = DataFittingHelper.fit_pipeline_with_loader(pipeline, loader)

    # Pipeline should be fitted
    result = fitted_pipeline.build_context("test")
    assert result.question == "test"


def test_data_loader_handles_non_dict_top_level(tmp_path: Path) -> None:
    """DataLoader should raise RuntimeError for non-dict top-level JSON."""
    # JSON array instead of object
    json_path = tmp_path / "array.json"
    json_path.write_text('[{"id": "doc1", "text": "test"}]')

    config = DataLoaderConfig(data_path=json_path)
    loader = DataLoader(config)

    with pytest.raises(RuntimeError, match="Expected JSON object"):
        loader.load_paragraphs()


def test_data_loader_empty_document_returns_no_paragraphs(tmp_path: Path) -> None:
    """DataLoader should raise RuntimeError if no valid paragraphs found."""
    data = {
        "doc1": {
            "id": "doc1",
            # No text field
            "title": "No Text"
        },
        "doc2": {
            "id": "doc2",
            "text": "",  # Empty text
            "title": "Empty"
        }
    }

    json_path = tmp_path / "empty.json"
    json_path.write_text(json.dumps(data, indent=2))

    config = DataLoaderConfig(data_path=json_path)
    loader = DataLoader(config)

    with pytest.raises(RuntimeError, match="No valid paragraphs found"):
        loader.load_paragraphs()