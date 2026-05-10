from __future__ import annotations

from typing import List

from .base import BaseRetriever
from .fusion import (
    reciprocal_rank_fusion,
    weighted_fusion,
)
from .schemas import RetrievalResult
from .utils import (
    deduplicate_results,
    filter_low_score_results,
)


class HybridRetriever(
    BaseRetriever
):
    """
    Hybrid retriever combining:
        - Dense retrieval (FAISS)
        - Sparse retrieval (BM25)

    Supported fusion:
        - weighted
        - rrf
    """

    def __init__(
        self,
        dense_retriever,
        sparse_retriever,
        fusion_type: str = "rrf",
        alpha: float = 0.5,
        filter_threshold: float = 0.05,
    ) -> None:

        self.dense_retriever = (
            dense_retriever
        )

        self.sparse_retriever = (
            sparse_retriever
        )

        self.fusion_type = (
            fusion_type
        )

        self.alpha = alpha

        self.filter_threshold = (
            filter_threshold
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[RetrievalResult]:

        if not query.strip():
            return []

        # =================================================
        # Retrieve extra candidates
        # =================================================

        candidate_k = max(
            top_k * 3,
            top_k + 10,
        )

        # =================================================
        # Dense retrieval
        # =================================================

        dense_results = (
            self.dense_retriever.retrieve(
                query=query,
                top_k=candidate_k,
            )
        )

        # =================================================
        # Sparse retrieval
        # =================================================

        sparse_results = (
            self.sparse_retriever.retrieve(
                query=query,
                top_k=candidate_k,
            )
        )

        # =================================================
        # Fusion
        # =================================================

        if (
            self.fusion_type
            == "weighted"
        ):

            fused_results = (
                weighted_fusion(
                    dense_results=(
                        dense_results
                    ),
                    sparse_results=(
                        sparse_results
                    ),
                    alpha=self.alpha,
                )
            )

        elif (
            self.fusion_type
            == "rrf"
        ):

            fused_results = (
                reciprocal_rank_fusion(
                    dense_results=(
                        dense_results
                    ),
                    sparse_results=(
                        sparse_results
                    ),
                )
            )

        else:

            raise ValueError(
                f"Unsupported fusion_type: "
                f"{self.fusion_type}"
            )

        # =================================================
        # Deduplicate
        # =================================================

        fused_results = (
            deduplicate_results(
                fused_results
            )
        )

        # =================================================
        # Filter low-score
        # =================================================

        fused_results = (
            filter_low_score_results(
                fused_results,
                threshold=(
                    self.filter_threshold
                ),
            )
        )

        # =================================================
        # Final top-k
        # =================================================

        final_results = (
            fused_results[:top_k]
        )

        # =================================================
        # Final ranking
        # =================================================

        for rank, result in enumerate(
            final_results
        ):

            result.retrieval_rank = (
                rank + 1
            )

        return final_results