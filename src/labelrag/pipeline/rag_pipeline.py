"""Main RAG pipeline entrypoint."""

from __future__ import annotations

from pathlib import Path

from labelgen import Paragraph

from labelrag.config import RAGPipelineConfig
from labelrag.generation.generator import AnswerGenerator
from labelrag.types import RAGAnswerResult, RetrievalResult


class RAGPipeline:
    """Public pipeline entrypoint for `labelrag`."""

    def __init__(
        self,
        config: RAGPipelineConfig | None = None,
        generator: AnswerGenerator | None = None,
    ) -> None:
        self.config = config or RAGPipelineConfig()
        self.generator = generator

    def fit(self, paragraphs: list[str] | list[Paragraph]) -> RAGPipeline:
        """Fit the pipeline on a paragraph corpus."""

        raise NotImplementedError("RAGPipeline.fit() is not implemented yet.")

    def build_context(self, question: str) -> RetrievalResult:
        """Build retrieval context for a question."""

        raise NotImplementedError("RAGPipeline.build_context() is not implemented yet.")

    def answer(self, question: str) -> RAGAnswerResult:
        """Answer a question using retrieval and an optional generator."""

        raise NotImplementedError("RAGPipeline.answer() is not implemented yet.")

    def answer_with_generator(
        self,
        question: str,
        generator: AnswerGenerator,
    ) -> RAGAnswerResult:
        """Answer a question with a per-call generator override."""

        raise NotImplementedError(
            "RAGPipeline.answer_with_generator() is not implemented yet."
        )

    def save(self, path: str | Path) -> None:
        """Persist the pipeline state to disk."""

        raise NotImplementedError("RAGPipeline.save() is not implemented yet.")

    @classmethod
    def load(cls, path: str | Path) -> RAGPipeline:
        """Load a pipeline from disk."""

        raise NotImplementedError("RAGPipeline.load() is not implemented yet.")
