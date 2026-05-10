from typing import Dict, Any, List

from .base import BaseHistorySelector

from .utils import (
    filter_meaningful_history,
    compute_recency_score,
    cosine_similarity,
    get_turn_content,
)


class HybridHistorySelector(
    BaseHistorySelector
):
    """
    Hybrid history selector.

    Strategy:
        1. Remove meaningless turns
        2. Compute semantic similarity
        3. Compute recency score
        4. Combine scores
        5. Select top-k relevant turns
    """

    def __init__(
        self,
        embedding_model,
        top_k: int = 3,
        alpha: float = 0.8,
        beta: float = 0.2,
        recent_window: int = 20
    ):

        super().__init__(top_k)

        self.embedding_model = (
            embedding_model
        )

        self.alpha = alpha
        self.beta = beta

        self.recent_window = (
            recent_window
        )

    def rank_history(
        self,
        query: str,
        history: List[Dict[str, Any]]
    ):
        """
        Rank history using:
            semantic similarity + recency.

        Returns:
            List[(turn, score)]
        """

        meaningful_history = (
            filter_meaningful_history(
                history
            )
        )

        meaningful_history = (
            meaningful_history[
                -self.recent_window :
            ]
        )

        if not meaningful_history:
            return []

        query_embedding = (
            self.embedding_model
            .embed_query(query)
        )

        scored_history = []

        total_turns = len(
            meaningful_history
        )

        for idx, turn in enumerate(
            meaningful_history
        ):

            content = (
                get_turn_content(turn)
            )

            turn_embedding = (
                self.embedding_model
                .embed_query(content)
            )

            semantic_score = (
                cosine_similarity(
                    query_embedding,
                    turn_embedding
                )
            )

            recency_score = (
                compute_recency_score(
                    idx,
                    total_turns
                )
            )

            final_score = (
                self.alpha
                * semantic_score
                +
                self.beta
                * recency_score
            )

            scored_history.append(
                (
                    turn,
                    final_score
                )
            )

        scored_history.sort(
            key=lambda x: x[1],
            reverse=True
        )

        return scored_history

    def select(
        self,
        query: str,
        history: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Select top-k ranked history.
        """

        ranked_history = (
            self.rank_history(
                query,
                history
            )
        )

        selected = [

            turn

            for turn, score
            in ranked_history[
                : self.top_k * 2
            ]
        ]

        return selected

    def run(
        self,
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run hybrid history selection.
        """

        query = state.get(
            "query",
            ""
        )

        history = state.get(
            "history",
            []
        )

        selected_history = (
            self.select(
                query,
                history
            )
        )

        state[
            "selected_history"
        ] = selected_history

        return state

    def __repr__(self):

        return (
            f"{self.__class__.__name__}("
            f"top_k={self.top_k}, "
            f"alpha={self.alpha}, "
            f"beta={self.beta}"
            f")"
        )