# Public API

This document defines the intended public API for `labelrag`.

The goal is to expose a small, stable surface for label-driven RAG evaluation
workflows while keeping `labelgen` as the only dependency responsible for
concept extraction, community detection, and label assignment.

Current dependency target:

- `paralabelgen==0.2.0`

## Package Entry Point

Recommended import style:

```python
from labelrag import (
    GeneratedAnswer,
    IndexedParagraph,
    PromptConfig,
    QueryAnalysis,
    RAGAnswerResult,
    RAGPipeline,
    RAGPipelineConfig,
    RetrievalConfig,
    RetrievalResult,
    RetrievedParagraph,
)
```

`labelrag` should also re-export `Paragraph` from `labelgen` for convenience:

```python
from labelrag import Paragraph
```

## Main Pipeline

### `RAGPipeline`

Primary public entrypoint for corpus fitting, query analysis, retrieval, and
optional answer generation.

Constructor:

```python
RAGPipeline(
    config: RAGPipelineConfig | None = None,
    generator: AnswerGenerator | None = None,
)
```

Behavior:

- `config` defaults to `RAGPipelineConfig()`
- `generator` is optional
- when no generator is configured, `answer(...)` may return an empty
  `answer_text`

### Methods

#### `fit`

```python
fit(paragraphs: list[str] | list[Paragraph]) -> RAGPipeline
```

Fits the underlying `labelgen.LabelGenerator`, builds paragraph-side retrieval
artifacts, and stores enough state for later `build_context(...)`, `answer(...)`,
and `save(...)`.

Accepted inputs:

- `list[str]`
- `list[labelgen.Paragraph]`

#### `build_context`

```python
build_context(question: str) -> RetrievalResult
```

Runs query-side analysis and retrieval only. Does not call the answer
generator.

#### `answer`

```python
answer(question: str) -> RAGAnswerResult
```

Runs query analysis, retrieval, prompt context construction, and optional answer
generation.

#### `answer_with_generator`

```python
answer_with_generator(
    question: str,
    generator: AnswerGenerator,
) -> RAGAnswerResult
```

Runs the full answer flow with a per-call generator override.

#### `save`

```python
save(path: str | Path) -> None
```

Persists pipeline configuration, fitted `LabelGenerator` state, and retrieval
artifacts.

#### `load`

```python
RAGPipeline.load(path: str | Path) -> RAGPipeline
```

Loads a previously saved pipeline.

### Error Conditions

The following conditions should raise explicit runtime errors:

- `build_context(...)` before `fit(...)`
- `answer(...)` before `fit(...)`
- `save(...)` before the pipeline has enough state to serialize

## Configuration Models

All public config models should use `@dataclass(slots=True)`.

### `RetrievalConfig`

```python
@dataclass(slots=True)
class RetrievalConfig:
    max_paragraphs: int = 8
    require_full_label_coverage: bool = False
    allow_label_free_fallback: bool = True
```

Field meaning:

- `max_paragraphs`: maximum retrieved paragraphs returned in one result
- `require_full_label_coverage`: when `True`, the pipeline may signal incomplete
  retrieval more strictly if not all query labels are covered
- `allow_label_free_fallback`: whether a query with no assigned labels may still
  return a fallback retrieval result

### `PromptConfig`

```python
@dataclass(slots=True)
class PromptConfig:
    include_paragraph_ids: bool = True
    include_label_annotations: bool = False
    max_context_characters: int | None = None
```

Field meaning:

- `include_paragraph_ids`: include stable paragraph IDs in rendered context
- `include_label_annotations`: optionally include label metadata in rendered
  prompt context
- `max_context_characters`: optional hard limit for prompt context length

### `RAGPipelineConfig`

```python
@dataclass(slots=True)
class RAGPipelineConfig:
    labelgen: LabelGeneratorConfig = field(default_factory=LabelGeneratorConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    prompt: PromptConfig = field(default_factory=PromptConfig)
```

Boundary rule:

- `labelrag` may configure `LabelGeneratorConfig`
- `labelrag` should not patch `labelgen` internals directly

## Data Models

All public result models should use `@dataclass(slots=True)`.

### `IndexedParagraph`

```python
@dataclass(slots=True)
class IndexedParagraph:
    paragraph_id: str
    text: str
    metadata: dict[str, Any] | None
    concept_ids: list[str]
    concept_texts: list[str]
    label_ids: list[str]
    label_display_names: list[str]
```

Meaning:

- normalized paragraph plus the label-side metadata needed for retrieval and
  trace inspection

### `QueryAnalysis`

```python
@dataclass(slots=True)
class QueryAnalysis:
    query_text: str
    concepts: list[str]
    concept_ids: list[str]
    label_ids: list[str]
    label_display_names: list[str]
```

Meaning:

- structured query-side output derived from
  `labelgen.LabelGenerator.transform([question])`

### `RetrievedParagraph`

```python
@dataclass(slots=True)
class RetrievedParagraph:
    paragraph_id: str
    text: str
    metadata: dict[str, Any] | None
    newly_covered_label_ids: list[str]
    already_covered_label_ids: list[str]
    matched_label_ids: list[str]
    matched_concept_ids: list[str]
    paragraph_label_ids: list[str]
    paragraph_concept_ids: list[str]
    concept_overlap_count: int
    marginal_gain: int
    retrieval_score: float
```

Meaning:

- one retrieval step record with enough data to audit why the paragraph was
  selected
- `newly_covered_label_ids` records the labels this retrieval step contributes
  toward final coverage
- `already_covered_label_ids` records labels that matched but were already
  covered by earlier retrieved paragraphs
- `concept_overlap_count` records the number of overlapping query concept IDs
  used in tie-break decisions

### `RetrievalResult`

```python
@dataclass(slots=True)
class RetrievalResult:
    question: str
    query_analysis: QueryAnalysis
    retrieved_paragraphs: list[RetrievedParagraph]
    prompt_context: str
    metadata: dict[str, Any]
```

Minimum required metadata keys:

- `covered_label_ids`
- `uncovered_label_ids`
- `attempted_covered_label_ids`
- `attempted_uncovered_label_ids`
- `retrieval_strategy`
- `used_label_free_fallback`
- `require_full_label_coverage`
- `full_label_coverage_met`

Recommended additional metadata keys:

- `query_label_ids`
- `retrieval_limit`

### `GeneratedAnswer`

```python
@dataclass(slots=True)
class GeneratedAnswer:
    text: str
    metadata: dict[str, Any]
```

Meaning:

- generator output normalized into a simple, model-agnostic structure

### `RAGAnswerResult`

```python
@dataclass(slots=True)
class RAGAnswerResult:
    question: str
    answer_text: str
    query_analysis: QueryAnalysis
    retrieved_paragraphs: list[RetrievedParagraph]
    prompt_context: str
    metadata: dict[str, Any]
```

Minimum required metadata keys:

- `covered_label_ids`
- `uncovered_label_ids`
- `attempted_covered_label_ids`
- `attempted_uncovered_label_ids`
- `retrieval_strategy`
- `used_label_free_fallback`
- `require_full_label_coverage`
- `full_label_coverage_met`
- `generator_name`
- `generation_model`

Recommended additional metadata keys:

- `query_label_ids`
- `retrieval_limit`
- `generation_metadata`

## Generator Protocol

`labelrag` should define an answer-generation abstraction rather than bind
directly to one provider SDK.

```python
class AnswerGenerator(Protocol):
    def generate(self, question: str, context: str) -> GeneratedAnswer: ...
```

Behavior:

- the protocol is synchronous for the first milestone
- generator implementation is injected from outside the retrieval core
- generator metadata should be folded into `RAGAnswerResult.metadata`

## Retrieval Semantics

Baseline retrieval strategy:

- greedy coverage over query label IDs

Tie-break order:

1. larger overlap with remaining query labels
2. larger overlap on query concept IDs when available
3. larger total paragraph label count
4. lexicographically smaller `paragraph_id`

Termination conditions:

- no remaining query labels
- `max_paragraphs` reached
- no paragraph yields positive gain

These retrieval rules are observable behavior and should be tested as part of
the public contract.

## Prompt Rendering

Default prompt context format:

```text
[Paragraph 1 | id=<paragraph_id>]
<paragraph text>

[Paragraph 2 | id=<paragraph_id>]
<paragraph text>
```

If `PromptConfig.include_label_annotations` is enabled, label annotations may be
added, but paragraph text should remain the primary content.

## Persistence Contract

`save(path)` should produce a human-inspectable directory layout similar to:

```text
pipeline_dir/
  config.json
  label_generator.json
  corpus_index.json
  fit_result.json
```

Persistence responsibilities:

- `label_generator.json` stores the fitted `labelgen.LabelGenerator`
- `config.json` stores `RAGPipelineConfig`
- `corpus_index.json` stores `labelrag` retrieval artifacts
- `fit_result.json` stores the fitted `LabelGenerationResult` snapshot exposed
  through `RAGPipeline.fit_result`

Public guarantee:

- a saved and reloaded pipeline should preserve retrieval behavior for the same
  fitted state, question, and config

## Convenience Re-Exports

For user ergonomics, `labelrag` may re-export the following `labelgen` public
types:

- `Paragraph`
- `LabelGeneratorConfig`

This is optional, but if re-exported it should be documented clearly and remain
stable once published.
