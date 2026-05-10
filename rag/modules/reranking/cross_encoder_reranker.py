from __future__ import annotations

from typing import List

from sentence_transformers import (
    CrossEncoder,
)

from .base import BaseReranker
from .schemas import RerankResult
from .utils import (
    combine_scores,
    deduplicate_reranked_results,
    normalize_rerank_scores,
)


class CrossEncoderReranker(
    BaseReranker
):
    """
    Cross-attention reranker.

    Pipeline:
        query + chunk
            ↓
        cross-encoder
            ↓
        rerank score
            ↓
        final ranking
    """

    def __init__(
        self,
        model_name: str = (
            "BAAI/bge-reranker-base"
        ),
        batch_size: int = 8,
        max_length: int = 512,
        score_alpha: float = 0.2,
    ) -> None:

        self.model = CrossEncoder(

            model_name,

            max_length=max_length,
        )

        self.batch_size = (
            batch_size
        )

        self.score_alpha = (
            score_alpha
        )

    # =====================================================
    # Main API
    # =====================================================

    def rerank(
        self,
        query: str,
        retrieval_results,
    ) -> List[RerankResult]:

        if not retrieval_results:
            return []

        # ================================================
        # Build query-document pairs
        # ================================================

        pairs = [

            (
                query,
                result.text,
            )

            for result
            in retrieval_results
        ]

        # ================================================
        # Predict logits
        # ================================================

        raw_scores = (
            self.model.predict(

                pairs,

                batch_size=(
                    self.batch_size
                ),

                show_progress_bar=False,
            )
        )

        # ================================================
        # Normalize rerank scores
        # ================================================

        rerank_scores = (
            normalize_rerank_scores(
                raw_scores.tolist()
            )
        )

        reranked_results = []

        # ================================================
        # Build rerank outputs
        # ================================================

        for rank, (
            retrieval_result,
            rerank_score,
        ) in enumerate(

            zip(
                retrieval_results,
                rerank_scores,
            )
        ):

            retrieval_score = float(

                getattr(
                    retrieval_result,
                    "final_score",
                    None,
                )

                or

                getattr(
                    retrieval_result,
                    "score",
                    0.0,
                )
            )

            final_score = (
                combine_scores(

                    retrieval_score=(
                        retrieval_score
                    ),

                    rerank_score=(
                        rerank_score
                    ),

                    alpha=(
                        self.score_alpha
                    ),
                )
            )

            reranked_results.append(

                RerankResult(

                    chunk_id=(
                        retrieval_result
                        .chunk_id
                    ),

                    text=(
                        retrieval_result
                        .text
                    ),

                    retrieval_score=(
                        retrieval_score
                    ),

                    rerank_score=(
                        rerank_score
                    ),

                    final_score=(
                        final_score
                    ),

                    metadata=(
                        retrieval_result
                        .metadata
                    ),

                    retrieval_rank=(
                        getattr(
                            retrieval_result,
                            "retrieval_rank",
                            rank + 1,
                        )
                    ),
                )
            )

        # ================================================
        # Sort by final score
        # ================================================

        reranked_results = sorted(

            reranked_results,

            key=lambda x: (
                x.final_score
            ),

            reverse=True,
        )

        # ================================================
        # Update rerank rank
        # ================================================

        for rank, result in enumerate(
            reranked_results
        ):

            result.rerank_rank = (
                rank + 1
            )

        # ================================================
        # Deduplicate
        # ================================================

        reranked_results = (
            deduplicate_reranked_results(
                reranked_results
            )
        )

        return reranked_results