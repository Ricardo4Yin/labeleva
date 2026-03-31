"""Selection helpers for greedy retrieval ranking."""

from labelrag.indexing.corpus_index import CorpusIndex
from labelrag.retrieval.coverage import uncovered_overlap_size
from labelrag.types import QueryAnalysis, RetrievedParagraph


def rank_retrieved_paragraphs(
    paragraphs: list[RetrievedParagraph],
) -> list[RetrievedParagraph]:
    """Return a stable descending retrieval order for traced paragraphs."""

    return sorted(
        paragraphs,
        key=lambda item: (
            -item.marginal_gain,
            -len(item.matched_concept_ids),
            -len(item.paragraph_label_ids),
            item.paragraph_id,
        ),
    )


def select_greedy_paragraphs(
    query_analysis: QueryAnalysis,
    corpus_index: CorpusIndex,
    *,
    max_paragraphs: int,
) -> list[RetrievedParagraph]:
    """Select paragraphs by greedy coverage over query label IDs."""

    remaining_label_ids = set(query_analysis.label_ids)
    query_label_ids = set(query_analysis.label_ids)
    query_concept_ids = set(query_analysis.concept_ids)
    selected_paragraph_ids: set[str] = set()
    selected: list[RetrievedParagraph] = []

    while remaining_label_ids and len(selected) < max_paragraphs:
        best_candidate: RetrievedParagraph | None = None
        best_sort_key: tuple[int, int, int, str] | None = None

        for paragraph_id, paragraph in corpus_index.paragraphs_by_id.items():
            if paragraph_id in selected_paragraph_ids:
                continue

            paragraph_label_ids = set(paragraph.label_ids)
            gain = uncovered_overlap_size(paragraph_label_ids, remaining_label_ids)
            if gain == 0:
                continue

            matched_concept_ids = sorted(set(paragraph.concept_ids) & query_concept_ids)
            candidate = RetrievedParagraph(
                paragraph_id=paragraph.paragraph_id,
                text=paragraph.text,
                metadata=paragraph.metadata,
                matched_label_ids=sorted(paragraph_label_ids & query_label_ids),
                matched_concept_ids=matched_concept_ids,
                paragraph_label_ids=list(paragraph.label_ids),
                paragraph_concept_ids=list(paragraph.concept_ids),
                marginal_gain=gain,
                retrieval_score=float(gain),
            )
            sort_key = (
                gain,
                len(matched_concept_ids),
                len(paragraph.label_ids),
                _reverse_lexicographic_key(paragraph.paragraph_id),
            )

            if best_candidate is None or best_sort_key is None or sort_key > best_sort_key:
                best_candidate = candidate
                best_sort_key = sort_key

        if best_candidate is None:
            break

        selected.append(best_candidate)
        selected_paragraph_ids.add(best_candidate.paragraph_id)
        remaining_label_ids -= set(best_candidate.matched_label_ids)

    return selected


def _reverse_lexicographic_key(value: str) -> str:
    """Invert lexicographic order so smaller IDs sort earlier in max comparisons."""

    return "".join(chr(0x10FFFF - ord(character)) for character in value)
