"""Tests for greedy label-based retrieval."""

from labelrag.indexing.corpus_index import CorpusIndex
from labelrag.retrieval.selector import select_greedy_paragraphs
from labelrag.types import IndexedParagraph, QueryAnalysis


def test_select_greedy_paragraphs_covers_query_labels_greedily() -> None:
    """The selector should choose paragraphs that greedily cover remaining labels."""

    corpus_index = CorpusIndex(
        paragraphs_by_id={
            "p1": IndexedParagraph(
                paragraph_id="p1",
                text="Paragraph 1",
                metadata=None,
                concept_ids=["c1"],
                concept_texts=["developers"],
                label_ids=["l1"],
                label_display_names=["developers"],
            ),
            "p2": IndexedParagraph(
                paragraph_id="p2",
                text="Paragraph 2",
                metadata=None,
                concept_ids=["c2"],
                concept_texts=["systems"],
                label_ids=["l2"],
                label_display_names=["systems"],
            ),
            "p3": IndexedParagraph(
                paragraph_id="p3",
                text="Paragraph 3",
                metadata=None,
                concept_ids=["c1", "c2"],
                concept_texts=["developers", "systems"],
                label_ids=["l1", "l2"],
                label_display_names=["developers", "systems"],
            ),
        }
    )
    query_analysis = QueryAnalysis(
        query_text="How do developers use systems?",
        concepts=["developers", "systems"],
        concept_ids=["c1", "c2"],
        label_ids=["l1", "l2"],
        label_display_names=["developers", "systems"],
    )

    selected = select_greedy_paragraphs(query_analysis, corpus_index, max_paragraphs=3)

    assert [paragraph.paragraph_id for paragraph in selected] == ["p3"]
    assert selected[0].marginal_gain == 2


def test_select_greedy_paragraphs_uses_concept_overlap_as_tiebreak() -> None:
    """Concept overlap should break ties when label gain is the same."""

    corpus_index = CorpusIndex(
        paragraphs_by_id={
            "p1": IndexedParagraph(
                paragraph_id="p1",
                text="Paragraph 1",
                metadata=None,
                concept_ids=["c1", "c2"],
                concept_texts=["developers", "language models"],
                label_ids=["l1"],
                label_display_names=["developers"],
            ),
            "p2": IndexedParagraph(
                paragraph_id="p2",
                text="Paragraph 2",
                metadata=None,
                concept_ids=["c1"],
                concept_texts=["developers"],
                label_ids=["l1"],
                label_display_names=["developers"],
            ),
        }
    )
    query_analysis = QueryAnalysis(
        query_text="How do developers use language models?",
        concepts=["developers", "language models"],
        concept_ids=["c1", "c2"],
        label_ids=["l1"],
        label_display_names=["developers"],
    )

    selected = select_greedy_paragraphs(query_analysis, corpus_index, max_paragraphs=2)

    assert [paragraph.paragraph_id for paragraph in selected] == ["p1"]


def test_select_greedy_paragraphs_uses_paragraph_id_as_final_tiebreak() -> None:
    """Lexicographically smaller paragraph IDs should win final ties."""

    corpus_index = CorpusIndex(
        paragraphs_by_id={
            "p1": IndexedParagraph(
                paragraph_id="p1",
                text="Paragraph 1",
                metadata=None,
                concept_ids=["c1"],
                concept_texts=["developers"],
                label_ids=["l1"],
                label_display_names=["developers"],
            ),
            "p2": IndexedParagraph(
                paragraph_id="p2",
                text="Paragraph 2",
                metadata=None,
                concept_ids=["c1"],
                concept_texts=["developers"],
                label_ids=["l1"],
                label_display_names=["developers"],
            ),
        }
    )
    query_analysis = QueryAnalysis(
        query_text="How do developers work?",
        concepts=["developers"],
        concept_ids=["c1"],
        label_ids=["l1"],
        label_display_names=["developers"],
    )

    selected = select_greedy_paragraphs(query_analysis, corpus_index, max_paragraphs=2)

    assert [paragraph.paragraph_id for paragraph in selected] == ["p1"]


def test_select_greedy_paragraphs_respects_max_paragraphs() -> None:
    """Selection should stop when the configured paragraph limit is reached."""

    corpus_index = CorpusIndex(
        paragraphs_by_id={
            "p1": IndexedParagraph(
                paragraph_id="p1",
                text="Paragraph 1",
                metadata=None,
                concept_ids=["c1"],
                concept_texts=["developers"],
                label_ids=["l1"],
                label_display_names=["developers"],
            ),
            "p2": IndexedParagraph(
                paragraph_id="p2",
                text="Paragraph 2",
                metadata=None,
                concept_ids=["c2"],
                concept_texts=["systems"],
                label_ids=["l2"],
                label_display_names=["systems"],
            ),
        }
    )
    query_analysis = QueryAnalysis(
        query_text="How do developers use systems?",
        concepts=["developers", "systems"],
        concept_ids=["c1", "c2"],
        label_ids=["l1", "l2"],
        label_display_names=["developers", "systems"],
    )

    selected = select_greedy_paragraphs(query_analysis, corpus_index, max_paragraphs=1)

    assert len(selected) == 1
