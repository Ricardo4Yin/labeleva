# labelrag

`labelrag` is a label-driven RAG pipeline built on top of `paralabelgen`.

Current status:

- MVP implementation is available
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
from labelrag import RAGPipeline, RAGPipelineConfig

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
```

## Current MVP Scope

The current package supports:

- corpus fitting on top of `labelgen.LabelGenerator`
- query analysis against the fitted label space
- deterministic greedy retrieval over query label IDs
- prompt context assembly
- optional answer generation through an injected synchronous generator
- pipeline save/load with JSON artifacts

## Current Deferred Config Fields

The following public config fields exist today but are still deferred/no-op in
the current MVP implementation:

- `RetrievalConfig.require_full_label_coverage`
- `RetrievalConfig.allow_label_free_fallback`
- `PromptConfig.include_label_annotations`

The current pipeline only actively uses:

- `RetrievalConfig.max_paragraphs`
- `PromptConfig.include_paragraph_ids`
- `PromptConfig.max_context_characters`

## Development Checks

```bash
.venv/bin/ruff check . --fix
.venv/bin/pyright
.venv/bin/pytest
```
