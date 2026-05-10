from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class RerankResult:
    """
    Unified reranker output schema.
    """

    chunk_id: str

    text: str

    retrieval_score: float

    rerank_score: float

    final_score: float

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    retrieval_rank: int = -1

    rerank_rank: int = -1

    source: str = "reranker"

    def to_dict(
        self,
    ) -> Dict[str, Any]:

        return {

            "chunk_id": self.chunk_id,

            "text": self.text,

            "retrieval_score": (
                self.retrieval_score
            ),

            "rerank_score": (
                self.rerank_score
            ),

            "final_score": (
                self.final_score
            ),

            "metadata": self.metadata,

            "retrieval_rank": (
                self.retrieval_rank
            ),

            "rerank_rank": (
                self.rerank_rank
            ),

            "source": self.source,
        }