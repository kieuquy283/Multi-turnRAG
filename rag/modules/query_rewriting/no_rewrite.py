from .base import BaseQueryRewriter


class NoRewrite(
    BaseQueryRewriter
):
    """
    Optional bypass rewriting module.

    Simply returns the original query.
    """

    def rewrite(
        self,
        query: str,
        history_text: str
    ) -> str:

        return query