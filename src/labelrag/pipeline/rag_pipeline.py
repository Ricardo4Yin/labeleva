"""Main RAG pipeline entrypoint."""

from __future__ import annotations

from pathlib import Path

from labelgen import LabelGenerationResult, LabelGenerator, Paragraph

from labelrag.config import RAGPipelineConfig
from labelrag.generation.generator import AnswerGenerator
from labelrag.indexing.corpus_index import CorpusIndex, build_corpus_index
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
        self._label_generator = LabelGenerator(self.config.labelgen)
        self._fit_result: LabelGenerationResult | None = None
        self._corpus_index: CorpusIndex | None = None

    @property
    def fit_result(self) -> LabelGenerationResult | None:
        """Return the most recent fit result when available."""

        return self._fit_result

    @property
    def corpus_index(self) -> CorpusIndex | None:
        """Return the paragraph-side corpus index when available."""

        return self._corpus_index

    def fit(self, paragraphs: list[str] | list[Paragraph]) -> RAGPipeline:
        """Fit the pipeline on a paragraph corpus."""

        result = self._label_generator.fit_transform(paragraphs)
        self._fit_result = result
        self._corpus_index = build_corpus_index(result)
        return self

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
