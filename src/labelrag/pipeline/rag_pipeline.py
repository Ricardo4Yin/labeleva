"""Main RAG pipeline entrypoint."""

from __future__ import annotations

from pathlib import Path

from labelgen import LabelGenerationResult, LabelGenerator, Paragraph

from labelrag.config import RAGPipelineConfig
from labelrag.generation.generator import AnswerGenerator
from labelrag.indexing.corpus_index import CorpusIndex, build_corpus_index
from labelrag.retrieval.selector import select_greedy_paragraphs
from labelrag.types import (
    QueryAnalysis,
    RAGAnswerResult,
    RetrievalResult,
    RetrievedParagraph,
)


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

    def analyze_query(self, question: str) -> QueryAnalysis:
        """Analyze a query against the fitted label space."""

        self._require_fitted()

        result = self._label_generator.transform([question])
        concept_ids = sorted({mention.concept_id for mention in result.mentions})
        concepts_by_id = {concept.id: concept.normalized for concept in result.concepts}
        label_display_names_by_id = {
            community.id: community.display_name for community in result.communities
        }
        label_ids = list(result.paragraph_labels[0].label_ids) if result.paragraph_labels else []

        return QueryAnalysis(
            query_text=question,
            concepts=[
                concepts_by_id[concept_id]
                for concept_id in concept_ids
                if concept_id in concepts_by_id
            ],
            concept_ids=concept_ids,
            label_ids=label_ids,
            label_display_names=[
                label_display_names_by_id[label_id]
                for label_id in label_ids
                if label_id in label_display_names_by_id
            ],
        )

    def _retrieve_paragraphs(self, query_analysis: QueryAnalysis) -> list[RetrievedParagraph]:
        """Retrieve paragraphs for a analyzed query using greedy label coverage."""

        self._require_fitted()
        assert self._corpus_index is not None

        return select_greedy_paragraphs(
            query_analysis,
            self._corpus_index,
            max_paragraphs=self.config.retrieval.max_paragraphs,
        )

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

    def _require_fitted(self) -> None:
        """Validate that the pipeline has already been fitted."""

        if self._fit_result is None or self._corpus_index is None:
            raise RuntimeError("RAGPipeline requires fit() before query-time operations.")
