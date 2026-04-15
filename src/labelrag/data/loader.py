"""Data loader for TechQA-style JSON datasets.

This module provides a DataLoader class specifically designed for loading
JSON files with TechQA dataset structure. The TechQA format consists of a
JSON object mapping document IDs to document objects, where each document
has 'id', 'text', 'title', and 'metadata' fields.

Note: This implementation is specifically designed for TechQA dataset format.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from labelgen import Paragraph


@dataclass(slots=True)
class DataLoaderConfig:
    """Configuration for loading JSON files with TechQA structure.

    Args:
        data_path: Path to JSON file with TechQA structure (dict mapping
            document_id to document object). Accepts str or Path.
        max_paragraph_length: Maximum characters per paragraph (default: 500).
        overlap_sentences: Number of overlapping sentences between paragraphs
            (default: 1).
    """

    data_path: Path
    max_paragraph_length: int = 500
    overlap_sentences: int = 1

    def __post_init__(self) -> None:
        """Convert string path to Path and validate."""
        if isinstance(self.data_path, str):
            self.data_path = Path(self.data_path)

        if not self.data_path.exists():
            raise RuntimeError(f"Data file not found: {self.data_path}")

        if self.max_paragraph_length <= 0:
            raise RuntimeError(
                f"max_paragraph_length must be positive, got {self.max_paragraph_length}"
            )

        if self.overlap_sentences < 0:
            raise RuntimeError(
                f"overlap_sentences must be non-negative, got {self.overlap_sentences}"
            )


class DataLoader:
    """Loader for JSON files with TechQA dataset structure.

    This loader reads TechQA-style JSON files, splits documents into paragraphs
    using sentence boundaries, and returns Paragraph objects for use with the
    RAGPipeline.fit() method.
    """

    def __init__(self, config: DataLoaderConfig) -> None:
        self.config = config
        self._sentence_pattern = re.compile(r'(?<=[.!?])\s+')

    def load_paragraphs(self) -> list[Paragraph]:
        """Load and split documents from the configured JSON file.

        Returns:
            List of Paragraph objects with text, IDs, and metadata.

        Raises:
            RuntimeError: If JSON file cannot be parsed or has invalid structure.
        """
        # Load JSON file
        try:
            with open(self.config.data_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise RuntimeError(
                f"Failed to parse JSON file {self.config.data_path}: {e}"
            ) from e

        # Validate basic structure (should be a dict mapping IDs to documents)
        if not isinstance(data, dict):
            raise RuntimeError(
                f"Expected JSON object (dict) at top level, got {type(data).__name__}"
            )

        paragraphs: list[Paragraph] = []

        for doc_id, doc_data in data.items(): # type: ignore
            if not isinstance(doc_data, dict):
                continue

            # Extract text from TechQA document structure
            # TechQA format: {"id": "...", "text": "...", "title": "...", "metadata": {...}}
            text = doc_data.get("text")
            if not isinstance(text, str) or not text.strip():
                continue  # Skip documents without text

            # Split document into paragraphs
            doc_paragraphs = self._split_document_into_paragraphs(
                text=text,
                doc_id=str(doc_id),
                metadata=doc_data.get("metadata", {}),
            )
            paragraphs.extend(doc_paragraphs)

        if not paragraphs:
            raise RuntimeError(
                f"No valid paragraphs found in {self.config.data_path}. "
                "Check that documents have non-empty 'text' fields."
            )

        return paragraphs

    def _split_document_into_paragraphs(
        self,
        text: str,
        doc_id: str,
        metadata: dict[str, Any],
    ) -> list[Paragraph]:
        """Split a document into paragraphs using sentence boundaries.

        Args:
            text: Document text to split.
            doc_id: Original document ID.
            metadata: Document metadata to include in paragraph metadata.

        Returns:
            List of Paragraph objects.
        """
        # Split text into sentences
        sentences = self._sentence_pattern.split(text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return []

        paragraphs: list[Paragraph] = []
        current_paragraph: list[str] = []
        current_length = 0
        sentence_index = 0

        while sentence_index < len(sentences):
            sentence = sentences[sentence_index]
            sentence_len = len(sentence)

            # If adding this sentence would exceed max length (and we have some content)
            if current_paragraph and current_length + sentence_len > self.config.max_paragraph_length:
                # Create paragraph from accumulated sentences
                paragraph_text = " ".join(current_paragraph)
                paragraph_id = f"{doc_id}#p{len(paragraphs)}"

                paragraphs.append(
                    Paragraph(
                        id=paragraph_id,
                        text=paragraph_text,
                        metadata={
                            **metadata,
                            "doc_id": doc_id,
                            "paragraph_index": len(paragraphs),
                            "sentence_start": sentence_index - len(current_paragraph),
                            "sentence_end": sentence_index - 1,
                        },
                    )
                )

                # Keep overlap_sentences for context
                overlap = min(self.config.overlap_sentences, len(current_paragraph))
                current_paragraph = current_paragraph[-overlap:] if overlap > 0 else []
                current_length = sum(len(s) for s in current_paragraph)

            # Add current sentence
            current_paragraph.append(sentence)
            current_length += sentence_len
            sentence_index += 1

        # Add final paragraph if any sentences remain
        if current_paragraph:
            paragraph_text = " ".join(current_paragraph)
            paragraph_id = f"{doc_id}#p{len(paragraphs)}"

            paragraphs.append(
                Paragraph(
                    id=paragraph_id,
                    text=paragraph_text,
                    metadata={
                        **metadata,
                        "doc_id": doc_id,
                        "paragraph_index": len(paragraphs),
                        "sentence_start": sentence_index - len(current_paragraph),
                        "sentence_end": sentence_index - 1,
                    },
                )
            )

        return paragraphs