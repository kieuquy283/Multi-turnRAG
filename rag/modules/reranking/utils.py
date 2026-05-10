from __future__ import annotations

import math
from typing import List

from .schemas import RerankResult


# =========================================================
# Score Utilities
# =========================================================

def sigmoid(
    x: float
) -> float:
    """
    Stable sigmoid normalization.
    """

    return 1 / (
        1 + math.exp(-x)
    )


def normalize_rerank_scores(
    scores: List[float]
) -> List[float]:
    """
    Normalize rerank logits
    into [0, 1].
    """

    if not scores:
        return []

    normalized = []

    for score in scores:

        try:
            normalized.append(
                sigmoid(score)
            )

        except OverflowError:

            normalized.append(
                1.0 if score > 0
                else 0.0
            )

    return normalized


# =========================================================
# Final Score Fusion
# =========================================================

def combine_scores(
    retrieval_score: float,
    rerank_score: float,
    alpha: float = 0.2,
) -> float:
    """
    Combine retrieval score
    and rerank score.

    rerank score dominates.
    """

    return (

        alpha
        * retrieval_score

        +

        (1 - alpha)
        * rerank_score
    )


# =========================================================
# Adaptive Threshold
# =========================================================

def adaptive_threshold(
    rerank_scores: List[float],
    base_threshold: float = 0.5,
) -> float:
    """
    Dynamic threshold based on
    score distribution.
    """

    if not rerank_scores:
        return base_threshold

    top_score = max(
        rerank_scores
    )

    if top_score >= 0.9:
        return 0.75

    if top_score >= 0.8:
        return 0.65

    if top_score >= 0.7:
        return 0.55

    return base_threshold


# =========================================================
# Context Selection
# =========================================================

def select_top_contexts(
    reranked_results: List[
        RerankResult
    ],
    min_contexts: int = 2,
    max_contexts: int = 8,
    relative_threshold: float = 0.8,
) -> List[RerankResult]:
    """
    Adaptive context selection.

    Rules:
        - if only few high-score chunks:
            select only those

        - if many high-score chunks:
            select all strong chunks
    """

    if not reranked_results:
        return []

    rerank_scores = [

        result.rerank_score

        for result
        in reranked_results
    ]

    threshold = adaptive_threshold(
        rerank_scores
    )

    top_score = (
        reranked_results[0]
        .rerank_score
    )

    selected = []

    for result in reranked_results:

        # ============================================
        # Absolute threshold
        # ============================================

        if (
            result.rerank_score
            < threshold
        ):
            continue

        # ============================================
        # Relative threshold
        # ============================================

        relative_score = (
            result.rerank_score
            /
            max(top_score, 1e-8)
        )

        if (
            relative_score
            < relative_threshold
        ):
            continue

        selected.append(result)

        if (
            len(selected)
            >= max_contexts
        ):
            break

    # ================================================
    # Ensure minimum contexts
    # ================================================

    if (
        len(selected)
        < min_contexts
    ):

        return reranked_results[
            : min_contexts
        ]

    return selected


# =========================================================
# Deduplication
# =========================================================

def deduplicate_reranked_results(
    results: List[RerankResult]
) -> List[RerankResult]:
    """
    Remove duplicated chunks.

    Keep highest rerank score.
    """

    unique = {}

    for result in results:

        chunk_id = (
            result.chunk_id
        )

        existing = unique.get(
            chunk_id
        )

        if existing is None:

            unique[
                chunk_id
            ] = result

            continue

        if (
            result.rerank_score
            > existing.rerank_score
        ):

            unique[
                chunk_id
            ] = result

    return sorted(

        unique.values(),

        key=lambda x: (
            x.final_score
        ),

        reverse=True,
    )


# =========================================================
# Diagnostics
# =========================================================

def summarize_rerank_results(
    results: List[RerankResult]
) -> List[dict]:
    """
    Lightweight debugging summary.
    """

    summary = []

    for result in results:

        summary.append({

            "chunk_id": (
                result.chunk_id
            ),

            "retrieval_score": round(
                result.retrieval_score,
                4,
            ),

            "rerank_score": round(
                result.rerank_score,
                4,
            ),

            "final_score": round(
                result.final_score,
                4,
            ),

            "preview": (
                result.text[:120]
            ),
        })

    return summary