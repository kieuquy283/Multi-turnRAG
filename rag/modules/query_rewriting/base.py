from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseQueryRewriter(ABC):
    """
    Base class for query rewriting modules.

    Input state:
        {
            "query": str,
            "selected_history": List[dict]
        }

    Output state:
        {
            "rewritten_query": str
        }
    """

    @abstractmethod
    def rewrite(
        self,
        query: str,
        history_text: str
    ) -> str:
        """
        Rewrite query using formatted history.

        Args:
            query:
                Current user query.

            history_text:
                Formatted selected history.

        Returns:
            Rewritten standalone query.
        """
        pass

    def run(
        self,
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute query rewriting module.

        Args:
            state:
                Pipeline state.

        Returns:
            Updated pipeline state.
        """

        query = state.get(
            "query",
            ""
        )

        history_text = state.get(
            "formatted_history",
            ""
        )

        rewritten_query = self.rewrite(
            query=query,
            history_text=history_text
        )

        if not rewritten_query:
            rewritten_query = query

        state[
            "rewritten_query"
        ] = rewritten_query

        return state

    def __repr__(self):

        return (
            f"{self.__class__.__name__}()"
        )