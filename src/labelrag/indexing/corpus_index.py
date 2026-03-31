"""Corpus index models and helpers."""

from dataclasses import dataclass, field

from labelrag.types import IndexedParagraph


@dataclass(slots=True)
class CorpusIndex:
    """Stored paragraph-side retrieval artifacts."""

    paragraphs_by_id: dict[str, IndexedParagraph] = field(default_factory=lambda: {})
    paragraph_ids_by_label: dict[str, list[str]] = field(default_factory=lambda: {})
    label_ids_by_paragraph: dict[str, list[str]] = field(default_factory=lambda: {})
    concept_ids_by_paragraph: dict[str, list[str]] = field(default_factory=lambda: {})
    label_display_names_by_id: dict[str, str] = field(default_factory=lambda: {})

