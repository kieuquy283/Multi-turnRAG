from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from .schemas import RetrievalResult


class BaseRetriever(ABC):
    """
    Abstract retrieval interface.

    All retrievers must return:
        List[RetrievalResult]   
    """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[RetrievalResult]:
        """
        Retrieve top-k relevant chunks.
        """

        raise NotImplementedError