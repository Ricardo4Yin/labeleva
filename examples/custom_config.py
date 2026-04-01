"""Custom retrieval configuration example for labelrag."""

from labelrag import RAGPipeline, RAGPipelineConfig


def main() -> None:
    """Run retrieval with explicit deterministic configuration changes."""

    config = RAGPipelineConfig()
    config.labelgen.extractor_mode = "heuristic"
    config.labelgen.use_graph_community_detection = False
    config.retrieval.max_paragraphs = 2
    config.retrieval.require_full_label_coverage = True
    config.prompt.include_label_annotations = True

    paragraphs = [
        "OpenAI builds language models for developers.",
        "Developers use language models in production systems.",
        "Production systems rely on monitoring and deployment tooling.",
    ]

    pipeline = RAGPipeline(config)
    pipeline.fit(paragraphs)

    complete = pipeline.build_context("How do developers use language models?")
    missing = pipeline.build_context("How do developers use language models and monitoring?")

    print("Complete query:")
    print(f"- prompt_context_present={bool(complete.prompt_context)}")
    print(f"- metadata={complete.metadata}")
    print("\nPartially coverable query with require_full_label_coverage=True:")
    print(f"- prompt_context_present={bool(missing.prompt_context)}")
    print(f"- metadata={missing.metadata}")


if __name__ == "__main__":
    main()
