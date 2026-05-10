from __future__ import annotations

from typing import List

from rank_bm25 import BM25Okapi

from .base import BaseRetriever
from .schemas import RetrievalResult
from .utils import (
    deduplicate_results,
    normalize_sparse_scores,
    tokenize_for_bm25,
)


class SparseRetriever(
    BaseRetriever
):
    """
    Sparse retriever using BM25.
    """

    def __init__(
        self,
        documents,
    ) -> None:

        self.documents = documents

        self.tokenized_corpus = [

            tokenize_for_bm25(
                doc.page_content
            )

            for doc
            in documents
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_corpus
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[RetrievalResult]:

        if not query.strip():
            return []

        tokenized_query = (
            tokenize_for_bm25(query)
        )

        if not tokenized_query:
            return []

        # ============================================
        # BM25 scoring
        # ============================================

        scores = self.bm25.get_scores(
            tokenized_query
        )

        ranked_indices = sorted(

            range(len(scores)),

            key=lambda i: scores[i],

            reverse=True,
        )[: max(top_k * 2, top_k + 5)]

        results = []

        for idx in ranked_indices:

            doc = self.documents[idx]

            metadata = (
                doc.metadata or {}
            )

            chunk_id = str(
                metadata.get(
                    "chunk_id",
                    idx,
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

                    score=float(scores[idx]),

                    source="sparse",

                    metadata=metadata,
                )
            )

        # ============================================
        # Normalize
        # ============================================

        results = normalize_sparse_scores(
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