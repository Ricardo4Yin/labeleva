# labelrag

`labelrag` is a label-driven RAG pipeline built on top of `paralabelgen`.

Current status:

- `0.0.0` draft implementation is available
- the target dependency is `paralabelgen==0.2.0`
- the default extraction path uses spaCy

## Install

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install -e .
python -m spacy download en_core_web_sm
```

## Quick Start

```python
from labelrag import (
    OpenAICompatibleAnswerGenerator,
    OpenAICompatibleConfig,
    RAGPipeline,
    RAGPipelineConfig,
)

paragraphs = [
    "OpenAI builds language models for developers.",
    "Developers use language models in production systems.",
    "Production systems need monitoring and evaluation tooling.",
]

pipeline = RAGPipeline(RAGPipelineConfig())
pipeline.fit(paragraphs)

retrieval = pipeline.build_context("How do developers use language models?")
print(retrieval.prompt_context)
print(retrieval.metadata)

answer = pipeline.answer("How do developers use language models?")
print(answer.answer_text)  # empty when no generator is configured

generator = OpenAICompatibleAnswerGenerator(
    OpenAICompatibleConfig(
        model="mistral-small-latest",
        api_key_env_var="MISTRAL_API_KEY",
        base_url="https://api.mistral.ai/v1",
    )
)
real_answer = pipeline.answer_with_generator(
    "How do developers use language models?",
    generator,
)
print(real_answer.answer_text)
```

## Current `0.0.0` Draft Scope

The current package supports:

- corpus fitting on top of `labelgen.LabelGenerator`
- query analysis against the fitted label space
- deterministic greedy retrieval over query label IDs
- deterministic concept-overlap fallback for label-free queries when enabled
- optional suppression of partial retrieval when full label coverage is required
- prompt context assembly
- optional paragraph label annotations in rendered context
- optional answer generation through an injected synchronous generator
- OpenAI-compatible provider integration for real answers
- pipeline save/load with JSON artifacts

## Config Behavior

The retrieval config currently behaves as follows:

- `max_paragraphs`: hard retrieval limit
- `require_full_label_coverage`: when `True`, partial retrieval is suppressed
- `allow_label_free_fallback`: when `True`, label-free queries fall back to concept overlap

The prompt config currently behaves as follows:

- `RetrievalConfig.max_paragraphs`
- `PromptConfig.include_paragraph_ids`
- `PromptConfig.include_label_annotations`
- `PromptConfig.max_context_characters`

## Provider Notes

The built-in provider adapter targets OpenAI-compatible chat-completions APIs.

It currently supports:

- standard base URLs such as `https://api.openai.com/v1`
- full endpoint URLs such as `https://api.mistral.ai/v1/chat/completions`
- API key injection through config or environment variables

This is intentionally a minimal non-streaming text-generation path for
`0.0.0`.

## Development Checks

```bash
.venv/bin/ruff check . --fix
.venv/bin/pyright
.venv/bin/pytest
```
