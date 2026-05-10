from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from .schemas import RetrievalResult


# =========================================================
# Score Normalization
# =========================================================

def min_max_normalize(
    scores: List[float]
) -> List[float]:
    """
    Min-max normalize scores.
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


# =========================================================
# Weighted Score Fusion
# =========================================================

def weighted_fusion(
    dense_results: List[RetrievalResult],
    sparse_results: List[RetrievalResult],
    alpha: float = 0.5,
) -> List[RetrievalResult]:
    """
    Weighted hybrid fusion.

    final_score =
        alpha * dense_score
        +
        (1 - alpha) * sparse_score
    """

    merged: Dict[
        str,
        RetrievalResult
    ] = {}

    # =====================================================
    # Dense
    # =====================================================

    for result in dense_results:

        chunk_id = result.chunk_id

        fused_score = (
            alpha * result.score
        )

        copied = RetrievalResult(
            chunk_id=result.chunk_id,
            text=result.text,
            score=result.score,
            source=result.source,
            metadata=result.metadata,
            retrieval_rank=(
                result.retrieval_rank
            ),
        )

        copied.final_score = (
            fused_score
        )

        merged[chunk_id] = copied

    # =====================================================
    # Sparse
    # =====================================================

    for result in sparse_results:

        chunk_id = result.chunk_id

        sparse_score = (
            (1 - alpha)
            * result.score
        )

        if chunk_id in merged:

            merged[
                chunk_id
            ].final_score += (
                sparse_score
            )

        else:

            copied = RetrievalResult(
                chunk_id=result.chunk_id,
                text=result.text,
                score=result.score,
                source=result.source,
                metadata=result.metadata,
                retrieval_rank=(
                    result.retrieval_rank
                ),
            )

            copied.final_score = (
                sparse_score
            )

            merged[
                chunk_id
            ] = copied

    # =====================================================
    # Ranking
    # =====================================================

    final_results = sorted(

        merged.values(),

        key=lambda x: (
            x.final_score or 0.0
        ),

        reverse=True,
    )

    # =====================================================
    # Update rank
    # =====================================================

    for rank, result in enumerate(
        final_results
    ):

        result.retrieval_rank = (
            rank + 1
        )

    return final_results


# =========================================================
# Reciprocal Rank Fusion (RRF)
# =========================================================

def reciprocal_rank_fusion(
    dense_results: List[RetrievalResult],
    sparse_results: List[RetrievalResult],
    k: int = 60,
) -> List[RetrievalResult]:
    """
    Reciprocal Rank Fusion (RRF).

    Stronger and more stable
    than weighted fusion.
    """

    scores = defaultdict(float)

    merged_objects = {}

    # =====================================================
    # Dense
    # =====================================================

    for rank, result in enumerate(
        dense_results
    ):

        rrf_score = 1 / (
            k + rank + 1
        )

        scores[
            result.chunk_id
        ] += rrf_score

        merged_objects[
            result.chunk_id
        ] = result

    # =====================================================
    # Sparse
    # =====================================================

    for rank, result in enumerate(
        sparse_results
    ):

        rrf_score = 1 / (
            k + rank + 1
        )

        scores[
            result.chunk_id
        ] += rrf_score

        if (
            result.chunk_id
            not in merged_objects
        ):

            merged_objects[
                result.chunk_id
            ] = result

    # =====================================================
    # Final ranking
    # =====================================================

    final_results = []

    for chunk_id, score in (
        scores.items()
    ):

        item = merged_objects[
            chunk_id
        ]

        item.final_score = score

        final_results.append(item)

    final_results = sorted(

        final_results,

        key=lambda x: (
            x.final_score or 0.0
        ),

        reverse=True,
    )

    # =====================================================
    # Update rank
    # =====================================================

    for rank, result in enumerate(
        final_results
    ):

        result.retrieval_rank = (
            rank + 1
        )

    return final_results