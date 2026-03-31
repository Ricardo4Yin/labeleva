"""Generator protocol definitions."""

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class GeneratedAnswer:
    """Model-agnostic generated answer payload."""

    text: str
    metadata: dict[str, Any] = field(default_factory=lambda: {})


class AnswerGenerator(Protocol):
    """Protocol for injected synchronous answer generation."""

    def generate(self, question: str, context: str) -> GeneratedAnswer:
        """Generate an answer from the provided question and context."""

        ...
