# Architecture

## Overview

The repository currently supports two architectural layers:

- a **legacy linear runtime path** used by the serving app and CLI
- a **modular research architecture** used for ablation experiments and future pipeline composition

Both layers are currently kept to preserve compatibility.

## Old Linear Architecture

The older runtime path is centered around:

- `rag/retrieval/`
- `rag/pipelines/`
- `rag/generation/`

Typical flow:

`User Question -> Query Rewriting -> FAISS Retrieval -> Filter Active Docs -> Prompt Builder -> LLM Answer`

This path is still used by:

- `app/api.py`
- `scripts/chat_cli.py`
- `rag/pipelines/chat_pipeline.py`

## New Modular Architecture

The new research-oriented architecture is centered around `rag/modules/`.

Typical modular flow:

`User Question -> History Selection -> Query Rewriting -> Retrieval -> Reranking -> Context Selection -> Answer Generation`

Not every ablation model uses all modules.

## Module Responsibilities

### `rag/modules/history_selection/`

Responsible only for selecting useful conversation history.

Examples:

- no history
- recency history
- semantic / hybrid history

### `rag/modules/query_rewriting/`

Responsible only for rewriting the current question using selected history.

Examples:

- no rewrite
- LLM rewrite
- cached rewrite

### `rag/modules/retrieval/`

Responsible only for document retrieval.

Examples:

- FAISS dense retrieval
- BM25 sparse retrieval
- hybrid retrieval
- retrieval fusion

### `rag/modules/reranking/`

Responsible only for reranking retrieved candidates.

Examples:

- no rerank
- cross-encoder reranker
- context selector

### `rag/modules/generation/`

Intended for modular answer generation and formatting. At the moment, this package exists but is not yet the main serving path.

## Pipelines

`rag/pipelines/` should compose modules into end-to-end pipelines.

It should not own low-level retrieval, rewriting, or reranking logic directly.

At the moment:

- `rag/pipelines/chat_pipeline.py` is still a legacy runtime composition layer
- future modular pipelines should be added alongside it, not by breaking it immediately

## Scripts

`scripts/` should contain:

- ablation evaluators
- ingestion / indexing tools
- dataset preparation tools
- one-off utilities

Recommended evaluator direction:

- `scripts/evaluate_model_1_baseline.py`
- `scripts/evaluate_model_2_rewrite_dense.py`
- future model-specific ablation scripts

Older general evaluators can remain, but should be documented as legacy/general-purpose.

## Where To Add New Components

### Add a new history selector

Place it under:

- `rag/modules/history_selection/`

### Add a new query rewriting strategy

Place it under:

- `rag/modules/query_rewriting/`

### Add a new retriever or fusion method

Place it under:

- `rag/modules/retrieval/`

### Add a new reranker

Place it under:

- `rag/modules/reranking/`

### Add a new ablation evaluator

Place it under:

- `scripts/`

Prefer naming like:

- `evaluate_model_<n>_<short_name>.py`

## Legacy Compatibility

Some older files under `rag/retrieval/` and `rag/pipelines/` remain intentionally.

These should be treated as compatibility layers until:

- app/API/CLI are migrated
- tests are updated
- old evaluation entrypoints are either wrapped or replaced

New development should prefer `rag/modules/*`.
