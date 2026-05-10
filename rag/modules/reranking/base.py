from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from .schemas import RerankResult


class BaseReranker(ABC):
    """
    Abstract reranker interface.

    All rerankers must:
        - rerank retrieved chunks
        - return sorted RerankResult
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        retrieval_results,
    ) -> List[RerankResult]:
        """
        Rerank retrieved candidates.
        """

        raise NotImplementedError