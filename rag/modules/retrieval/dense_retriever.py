from __future__ import annotations

from typing import List

from .base import BaseRetriever
from .schemas import RetrievalResult
from .utils import (
    deduplicate_results,
    normalize_dense_scores,
)


class DenseRetriever(
    BaseRetriever
):
    """
    Dense retriever using FAISS
    similarity search.
    """

    def __init__(
        self,
        vectorstore,
    ) -> None:

        self.vectorstore = (
            vectorstore
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[RetrievalResult]:

        if not query.strip():
            return []

        # ============================================
        # Retrieve extra candidates
        # ============================================

        search_k = max(
            top_k * 2,
            top_k + 5,
        )

        docs_scores = (
            self.vectorstore.similarity_search_with_score(
                query=query,
                k=search_k,
            )
        )

        results = []

        for rank, (
            doc,
            raw_score,
        ) in enumerate(docs_scores):

            metadata = (
                doc.metadata or {}
            )

            chunk_id = str(
                metadata.get(
                    "chunk_id",
                    rank,
                )
            )

            text = (
                doc.page_content
                or ""
            ).strip()

            if not text:
                continue

            results.append(

                RetrievalResult(

                    chunk_id=chunk_id,

                    text=text,

                    score=float(raw_score),

                    source="dense",

                    metadata=metadata,
                )
            )

        # ============================================
        # Normalize
        # ============================================

        results = normalize_dense_scores(
            results
        )

        # ============================================
        # Deduplicate
        # ============================================

        results = deduplicate_results(
            results
        )

        # ============================================
        # Final top-k
        # ============================================

        return results[:top_k]