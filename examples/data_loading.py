"""Data loading example for labelrag.

This example demonstrates loading data from JSON file and fitting a RAGPipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add src directory to path to ensure labelrag.data can be imported
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from labelrag import RAGPipeline, RAGPipelineConfig
from labelrag.data import DataFittingHelper


def main() -> None:
    """Load data from JSON file and fit RAG pipeline."""
    # Load data from JSON file
    data_file = Path(__file__).parent / "techqa_test.json"

    # Create pipeline configuration
    config = RAGPipelineConfig()
    config.labelgen.extractor_mode = "heuristic"
    config.labelgen.use_graph_community_detection = False

    # Create pipeline instance
    pipeline = RAGPipeline(config)

    # Fit pipeline using DataFittingHelper
    fitted_pipeline = DataFittingHelper.fit_pipeline_from_json(
        pipeline=pipeline,
        data_path=data_file,
        max_paragraph_length=500,
        overlap_sentences=1,
    )

    # Demonstrate retrieval
    sample_question = "How to fix installation error?"
    result = fitted_pipeline.build_context(sample_question)

    print(f"Query: '{sample_question}'")
    print(f"Retrieved {len(result.retrieved_paragraphs)} paragraphs")
    for i, paragraph in enumerate(result.retrieved_paragraphs[:3], 1):
        doc_id = paragraph.metadata.get("doc_id", "unknown")
        preview = paragraph.text[:80].replace("\n", " ")
        print(f"{i}. [{doc_id}] {preview}...")

    print(f"\nPrompt context length: {len(result.prompt_context)} characters")


if __name__ == "__main__":
    main()