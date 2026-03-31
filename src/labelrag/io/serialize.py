"""Generic JSON helpers for `labelrag`."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from labelgen.io.serialize import config_from_dict, config_to_dict

from labelrag.config import PromptConfig, RAGPipelineConfig, RetrievalConfig
from labelrag.indexing.corpus_index import CorpusIndex
from labelrag.types import IndexedParagraph


def dump_json(data: dict[str, Any], path: str | Path) -> None:
    """Serialize a JSON object to disk."""

    destination = Path(path)
    destination.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def load_json(path: str | Path) -> dict[str, Any]:
    """Load a JSON object from disk."""

    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("Serialized object must be a JSON object.")
    return _as_string_key_dict(cast(object, data))


def _as_string_key_dict(value: object) -> dict[str, Any]:
    """Normalize a mapping into a string-key dictionary."""

    if not isinstance(value, dict):
        raise TypeError("Expected a JSON object.")

    normalized: dict[str, Any] = {}
    for key, item in cast(dict[object, object], value).items():
        if not isinstance(key, str):
            raise TypeError("JSON object keys must be strings.")
        normalized[key] = item
    return normalized


def pipeline_config_to_dict(config: RAGPipelineConfig) -> dict[str, Any]:
    """Convert a pipeline config into a JSON-serializable dictionary."""

    return {
        "labelgen": config_to_dict(config.labelgen),
        "retrieval": asdict(config.retrieval),
        "prompt": asdict(config.prompt),
    }


def pipeline_config_from_dict(data: dict[str, Any]) -> RAGPipelineConfig:
    """Reconstruct a pipeline config from serialized data."""

    retrieval_data = _as_string_key_dict(data.get("retrieval"))
    prompt_data = _as_string_key_dict(data.get("prompt"))
    return RAGPipelineConfig(
        labelgen=config_from_dict(_as_string_key_dict(data.get("labelgen"))),
        retrieval=RetrievalConfig(**retrieval_data),
        prompt=PromptConfig(**prompt_data),
    )


def corpus_index_to_dict(index: CorpusIndex) -> dict[str, Any]:
    """Convert a corpus index into a JSON-serializable dictionary."""

    return {
        "paragraphs_by_id": {
            paragraph_id: asdict(paragraph)
            for paragraph_id, paragraph in index.paragraphs_by_id.items()
        },
        "paragraph_ids_by_label": index.paragraph_ids_by_label,
        "label_ids_by_paragraph": index.label_ids_by_paragraph,
        "concept_ids_by_paragraph": index.concept_ids_by_paragraph,
        "label_display_names_by_id": index.label_display_names_by_id,
    }


def corpus_index_from_dict(data: dict[str, Any]) -> CorpusIndex:
    """Reconstruct a corpus index from serialized data."""

    paragraphs_by_id_data = _as_string_key_dict(data.get("paragraphs_by_id"))
    return CorpusIndex(
        paragraphs_by_id={
            paragraph_id: IndexedParagraph(**_as_string_key_dict(paragraph_data))
            for paragraph_id, paragraph_data in paragraphs_by_id_data.items()
        },
        paragraph_ids_by_label={
            label_id: _as_string_list(value)
            for label_id, value in _as_string_key_dict(data.get("paragraph_ids_by_label")).items()
        },
        label_ids_by_paragraph={
            paragraph_id: _as_string_list(value)
            for paragraph_id, value in _as_string_key_dict(
                data.get("label_ids_by_paragraph")
            ).items()
        },
        concept_ids_by_paragraph={
            paragraph_id: _as_string_list(value)
            for paragraph_id, value in _as_string_key_dict(
                data.get("concept_ids_by_paragraph")
            ).items()
        },
        label_display_names_by_id={
            label_id: _as_string(value)
            for label_id, value in _as_string_key_dict(
                data.get("label_display_names_by_id")
            ).items()
        },
    )


def _as_string_list(value: object) -> list[str]:
    """Normalize a list of strings from serialized data."""

    if not isinstance(value, list):
        raise TypeError("Expected a list of strings.")

    normalized: list[str] = []
    for item in cast(list[object], value):
        normalized.append(_as_string(item))
    return normalized


def _as_string(value: object) -> str:
    """Normalize a string value from serialized data."""

    if not isinstance(value, str):
        raise TypeError("Expected a string value.")
    return value
