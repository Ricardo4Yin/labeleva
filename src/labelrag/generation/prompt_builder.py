"""Prompt-building helpers for `labelrag`."""

from labelrag.config import PromptConfig
from labelrag.types import RetrievedParagraph


def build_prompt_context(
    paragraphs: list[RetrievedParagraph],
    config: PromptConfig,
) -> str:
    """Render a plain-text context block from retrieved paragraphs."""

    blocks: list[str] = []
    for index, paragraph in enumerate(paragraphs, start=1):
        if config.include_paragraph_ids:
            header = f"[Paragraph {index} | id={paragraph.paragraph_id}]"
        else:
            header = f"[Paragraph {index}]"
        blocks.append(f"{header}\n{paragraph.text}")

    context = "\n\n".join(blocks)
    if config.max_context_characters is not None:
        return context[: config.max_context_characters]
    return context

