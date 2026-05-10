from typing import Dict, Any, List

from .utils import (
    filter_meaningful_history,
    rank_by_recency,
)


class BaseHistorySelector:
    """
    Baseline history selector.

    Strategy:
        1. Remove meaningless turns
        2. Rank by recency
        3. Select top-k recent turns
    """

    def __init__(
        self,
        top_k: int = 3
    ):

        self.top_k = top_k

    def select(
        self,
        history: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Select meaningful recent history.

        Args:
            history:
                Full conversation history.

        Returns:
            Selected history turns.
        """

        meaningful_history = (
            filter_meaningful_history(
                history
            )
        )

        ranked_history = (
            rank_by_recency(
                meaningful_history
            )
        )

        return ranked_history[
            : self.top_k * 2
        ]

    def run(
        self,
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run history selection module.

        Args:
            state:
                Pipeline state.

        Returns:
            Updated state.
        """

        history = state.get(
            "history",
            []
        )

        selected_history = (
            self.select(history)
        )

        state[
            "selected_history"
        ] = selected_history

        return state

    def __repr__(self):

        return (
            f"{self.__class__.__name__}"
            f"(top_k={self.top_k})"
        )