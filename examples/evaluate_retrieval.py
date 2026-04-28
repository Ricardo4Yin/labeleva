"""Retrieval evaluation example for labelrag using the TechQA test set."""

from __future__ import annotations

import json
from pathlib import Path

from _demo_embedding import DemoEmbeddingProvider

from labelrag import RAGPipeline, RAGPipelineConfig
from labelrag.evaluation import DefaultRetrievalEvaluator, format_report


def main() -> None:
    """Fit a pipeline on TechQA documents and evaluate retrieval effectiveness."""
    data_path = Path(__file__).parent / "techqa_test.json"
    with open(data_path, encoding="utf-8") as f:
        documents = json.load(f)

    paragraphs = [doc["text"] for doc in documents.values()]

    config = RAGPipelineConfig()
    config.labelgen.extractor_mode = "heuristic"
    config.labelgen.use_graph_community_detection = False
    config.embedding.normalize = False

    pipeline = RAGPipeline(
        config,
        embedding_provider=DemoEmbeddingProvider(),
    )
    pipeline.fit(paragraphs)

    queries = [
        "How to fix installation error?",
        "What is WebSphere MQ?",
    ]

    evaluator = DefaultRetrievalEvaluator()
    metrics = evaluator.evaluate(pipeline, queries)

    print(format_report(metrics))


if __name__ == "__main__":
    main()
