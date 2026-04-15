"""Data loading example for labelrag.

This example demonstrates how to load embedded test data and fit a RAGPipeline
using the `DataFittingHelper` utility class.
"""

import json
import tempfile
from pathlib import Path

from labelrag import RAGPipeline, RAGPipelineConfig
from labelrag.data import DataFittingHelper


def main() -> None:
    """Load embedded test data and fit RAG pipeline."""
    # Embedded test data in TechQA format (3 documents)
    test_data = {
        "doc1": {
            "id": "doc1",
            "text": "Artificial intelligence (AI) is intelligence demonstrated by machines. "
                   "AI applications include advanced web search engines, recommendation systems, "
                   "understanding human speech, self-driving cars, and automated decision-making.",
            "title": "Introduction to AI",
            "metadata": {"category": "technology", "source": "example"}
        },
        "doc2": {
            "id": "doc2",
            "text": "Machine learning is a subset of AI that enables computers to learn "
                   "from data without being explicitly programmed. Deep learning uses "
                   "neural networks with multiple layers to analyze various factors of data.",
            "title": "Machine Learning Basics",
            "metadata": {"category": "technology", "source": "example"}
        },
        "doc3": {
            "id": "doc3",
            "text": "Natural language processing allows computers to understand, "
                   "interpret, and manipulate human language. NLP techniques are used "
                   "in chatbots, translation services, and sentiment analysis tools.",
            "title": "Natural Language Processing",
            "metadata": {"category": "technology", "source": "example"}
        }
    }

    # Create temporary file for DataLoader
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(test_data, f, indent=2)
        temp_file = Path(f.name)

    try:
        print(f"Using embedded test data (3 documents) at: {temp_file}")
        print()

        # Create pipeline configuration
        config = RAGPipelineConfig()
        config.labelgen.extractor_mode = "heuristic"
        config.labelgen.use_graph_community_detection = False

        # Create pipeline instance
        pipeline = RAGPipeline(config)

        print("Fitting pipeline with embedded test data...")

        # Fit pipeline using DataFittingHelper
        try:
            fitted_pipeline = DataFittingHelper.fit_pipeline_from_json(
                pipeline=pipeline,
                data_path=temp_file,
                max_paragraph_length=500,
                overlap_sentences=1,
            )
            print("Pipeline fitted successfully!")
        except Exception as e:
            print(f"Failed to fit pipeline: {e}")
            return

        # Demonstrate retrieval
        sample_question = "What is machine learning?"
        print(f"\nTesting retrieval with question: '{sample_question}'")

        try:
            result = fitted_pipeline.build_context(sample_question)

            print(f"\nRetrieved {len(result.retrieved_paragraphs)} paragraphs:")
            for i, paragraph in enumerate(result.retrieved_paragraphs[:3], 1):
                print(f"{i}. {paragraph.paragraph_id}: {paragraph.text[:100]}...")

            print(f"\nPrompt context length: {len(result.prompt_context)} characters")

        except Exception as e:
            print(f"Retrieval failed: {e}")

        print("\nExample completed successfully!")

    finally:
        # Clean up temporary file
        temp_file.unlink(missing_ok=True)


if __name__ == "__main__":
    main()