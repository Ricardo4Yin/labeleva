"""Generic JSON helpers for `labelrag`."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast


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

