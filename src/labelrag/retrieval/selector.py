"""Selection helpers for retrieval ranking."""

from labelrag.types import RetrievedParagraph


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

