from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class RetrievalResult:
    """
    Unified retrieval result schema.

    Used for:
        - Dense retrieval
        - Sparse retrieval
        - Hybrid retrieval
        - Reranking
    """

    chunk_id: str

    text: str

    score: float

    source: str

    metadata: Dict[str, Any] = field(
        default_factory=dict
    )

    retrieval_rank: int = -1

    rerank_score: float | None = None

    final_score: float | None = None

    def to_dict(
        self,
    ) -> Dict[str, Any]:

        return {

            "chunk_id": self.chunk_id,

            "text": self.text,

            "score": self.score,

            "source": self.source,

            "metadata": self.metadata,

            "retrieval_rank": (
                self.retrieval_rank
            ),

            "rerank_score": (
                self.rerank_score
            ),

            "final_score": (
                self.final_score
            ),
        }