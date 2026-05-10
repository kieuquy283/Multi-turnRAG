from __future__ import annotations

import re
from typing import List

from .schemas import RetrievalResult


# =========================================================
# Text Processing
# =========================================================

STOPWORDS = {

    "và",
    "là",
    "của",
    "cho",
    "với",
    "the",
    "is",
    "are",
    "of",
    "to",
    "a",
    "an",
}


def normalize_text(
    text: str
) -> str:
    """
    Normalize text.
    """

    text = str(text).lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def tokenize_for_bm25(
    text: str
) -> List[str]:
    """
    Lightweight BM25 tokenizer.
    """

    text = normalize_text(
        text
    )

    tokens = re.findall(
        r"\w+",
        text,
    )

    tokens = [

        token

        for token in tokens

        if token not in STOPWORDS
        and len(token) > 1
    ]

    return tokens


# =========================================================
# Score Normalization
# =========================================================

def min_max_normalize(
    scores: List[float]
) -> List[float]:
    """
    Min-max normalization.
    """

    if not scores:
        return scores

    min_score = min(scores)

    max_score = max(scores)

    if max_score == min_score:
        return [1.0] * len(scores)

    return [

        (
            score - min_score
        )
        /
        (
            max_score - min_score
        )

        for score
        in scores
    ]


def normalize_dense_scores(
    results: List[RetrievalResult]
) -> List[RetrievalResult]:
    """
    Normalize dense retrieval scores.

    FAISS distance:
        smaller = better

    Convert to:
        larger = better
    """

    if not results:
        return results

    raw_scores = [
        r.score
        for r in results
    ]

    max_score = max(raw_scores)

    converted = [

        max_score - score

        for score
        in raw_scores
    ]

    normalized = (
        min_max_normalize(
            converted
        )
    )

    for result, score in zip(
        results,
        normalized,
    ):

        result.score = score

    return results


def normalize_sparse_scores(
    results: List[RetrievalResult]
) -> List[RetrievalResult]:
    """
    Normalize BM25 scores.
    """

    if not results:
        return results

    raw_scores = [
        r.score
        for r in results
    ]

    normalized = (
        min_max_normalize(
            raw_scores
        )
    )

    for result, score in zip(
        results,
        normalized,
    ):

        result.score = score

    return results


# =========================================================
# Deduplication
# =========================================================

def deduplicate_results(
    results: List[RetrievalResult]
) -> List[RetrievalResult]:
    """
    Remove duplicated chunks.

    Keep highest score.
    """

    unique_results = {}

    for result in results:

        chunk_id = result.chunk_id

        existing = unique_results.get(
            chunk_id
        )

        if existing is None:

            unique_results[
                chunk_id
            ] = result

            continue

        if (
            result.score
            > existing.score
        ):

            unique_results[
                chunk_id
            ] = result

    return sorted(

        unique_results.values(),

        key=lambda x: x.score,

        reverse=True,
    )


# =========================================================
# Filtering
# =========================================================

def filter_low_score_results(
    results: List[RetrievalResult],
    threshold: float = 0.1,
) -> List[RetrievalResult]:
    """
    Filter low-score results.
    """

    return [

        result

        for result
        in results

        if result.score >= threshold
    ]


# =========================================================
# Retrieval Diagnostics
# =========================================================

def summarize_results(
    results: List[RetrievalResult]
) -> List[dict]:
    """
    Lightweight debug summary.
    """

    summary = []

    for result in results:

        summary.append({

            "chunk_id": result.chunk_id,

            "score": round(
                result.score,
                4,
            ),

            "source": result.source,

            "preview": (
                result.text[:120]
            ),
        })

    return summary