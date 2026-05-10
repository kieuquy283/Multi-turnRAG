from __future__ import annotations

from typing import Dict, Any
import re

from rag.config.llm import (
    REWRITE_MODEL
)

from rag.generation.llm_client import (
    get_llm
)

from .base import (
    BaseQueryRewriter
)

from .prompts import (
    REWRITE_PROMPT
)

from .utils import (
    format_history_for_rewrite,
    should_skip_rewrite,
    clean_rewritten_query,
    is_likely_follow_up,
)


class LLMQueryRewrite(
    BaseQueryRewriter
):
    """
    LLM-based query rewriting module.

    Strategy:
        1. Detect whether rewrite is necessary
        2. Format selected history
        3. Rewrite into standalone query
        4. Clean and validate output
    """

    def __init__(
        self,
        model_name: str = REWRITE_MODEL,
        temperature: float = 0.0,
        max_rewrite_ratio: float = 2.0,
        max_tokens_multiplier: int = 20,
    ):

        self.model_name = model_name

        self.temperature = temperature

        self.max_rewrite_ratio = (
            max_rewrite_ratio
        )

        self.max_tokens_multiplier = (
            max_tokens_multiplier
        )

        self.llm = get_llm(
            model_name=self.model_name,
            temperature=self.temperature,
        )

    def should_rewrite(
        self,
        query: str,
        selected_history,
    ) -> bool:
        """
        Determine whether rewrite is needed.
        """

        if not query.strip():
            return False

        if should_skip_rewrite(
            selected_history
        ):
            return False

        if not is_likely_follow_up(
            query
        ):
            return False

        return True


    def validate_rewrite(
        self,
        original_query: str,
        rewritten_query: str,
    ) -> bool:
        """
        Validate rewritten query quality.

        Validation Strategy:
            1. Non-empty check
            2. Length explosion detection
            3. Answer-style detection
            4. Hallucinated formatting detection
            5. Semantic preservation heuristic
            6. Keyword preservation
            7. Repetition detection
        """

        # =====================================================
        # Normalize
        # =====================================================

        original_query = (
            original_query.strip()
        )

        rewritten_query = (
            rewritten_query.strip()
        )

        # =====================================================
        # Empty check
        # =====================================================

        if not rewritten_query:
            return False

        # =====================================================
        # Exact same query is valid
        # =====================================================

        if (
            rewritten_query.lower()
            ==
            original_query.lower()
        ):
            return True

        # =====================================================
        # Token length validation
        # =====================================================

        original_tokens = (
            original_query.split()
        )

        rewritten_tokens = (
            rewritten_query.split()
        )

        original_len = len(
            original_tokens
        )

        rewritten_len = len(
            rewritten_tokens
        )

        max_allowed = max(
            self.max_tokens_multiplier,
            int(
                original_len
                * self.max_rewrite_ratio
            ),
        )

        if rewritten_len > max_allowed:
            return False

        # =====================================================
        # Extremely short rewritten query
        # =====================================================

        if rewritten_len <= 1:
            return False

        # =====================================================
        # Detect answer-style outputs
        # =====================================================

        answer_patterns = [

            r"là\s",
            r"bao gồm",
            r"được hiểu là",
            r"là một",
            r"refer to",
            r"is a",
            r"means",
            r"can be",
        ]

        lowered = rewritten_query.lower()

        for pattern in answer_patterns:

            if re.search(pattern, lowered):

                # Allow if query itself
                # already contains similar phrasing

                if pattern not in (
                    original_query.lower()
                ):
                    return False

        # =====================================================
        # Detect hallucinated formatting
        # =====================================================

        invalid_patterns = [

            r"^answer:",
            r"^response:",
            r"^giải thích",
            r"^rewrite:",
            r"^rewritten query:",
            r"^standalone query:",
        ]

        for pattern in invalid_patterns:

            if re.search(
                pattern,
                lowered
            ):
                return False

        # =====================================================
        # Detect excessive punctuation
        # =====================================================

        if rewritten_query.count("?") > 2:
            return False

        # =====================================================
        # Detect repetition
        # =====================================================

        repeated = re.search(
            r"\b(\w+)\s+\1\b",
            lowered
        )

        if repeated:
            return False

        # =====================================================
        # Keyword preservation
        # =====================================================

        original_keywords = {

            token.lower()

            for token in original_tokens

            if len(token) >= 4
        }

        rewritten_keywords = {

            token.lower()

            for token in rewritten_tokens

            if len(token) >= 4
        }

        # Avoid over-strict filtering
        # for very short queries

        if len(original_keywords) >= 2:

            overlap = (
                len(
                    original_keywords
                    &
                    rewritten_keywords
                )
                /
                len(original_keywords)
            )

            if overlap < 0.3:
                return False

        # =====================================================
        # Preserve numbers
        # =====================================================

        original_numbers = re.findall(
            r"\d+",
            original_query
        )

        rewritten_numbers = re.findall(
            r"\d+",
            rewritten_query
        )

        for number in original_numbers:

            if number not in rewritten_numbers:
                return False

        # =====================================================
        # Preserve uppercase entities
        # =====================================================

        original_entities = [

            token

            for token
            in original_query.split()

            if len(token) > 1
            and token[0].isupper()
        ]

        rewritten_text_lower = (
            rewritten_query.lower()
        )

        for entity in original_entities:

            if (
                entity.lower()
                not in rewritten_text_lower
            ):
                return False

        return True
    
    def rewrite(
        self,
        query: str,
        history_text: str,
    ) -> str:
        """
        Rewrite query using LLM.
        """

        prompt = REWRITE_PROMPT.format(
            history=history_text,
            query=query,
        )

        response = self.llm.invoke(
            prompt
        )

        rewritten_query = (
            clean_rewritten_query(
                response.content
            )
        )

        return rewritten_query

    def run(
        self,
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute query rewriting module.
        """

        query = state.get(
            "query",
            ""
        ).strip()

        selected_history = state.get(
            "selected_history",
            []
        )

        history_text = (
            format_history_for_rewrite(
                selected_history
            )
        )

        state[
            "formatted_history"
        ] = history_text

        if not self.should_rewrite(
            query,
            selected_history,
        ):

            state[
                "rewritten_query"
            ] = query

            state[
                "rewrite_applied"
            ] = False

            return state

        rewritten_query = self.rewrite(
            query=query,
            history_text=history_text,
        )

        if not self.validate_rewrite(
            query,
            rewritten_query,
        ):

            rewritten_query = query

        state[
            "rewritten_query"
        ] = rewritten_query

        state[
            "rewrite_applied"
        ] = (
            rewritten_query != query
        )

        return state

    def __repr__(self):

        return (
            f"{self.__class__.__name__}("
            f"model_name={self.model_name}, "
            f"temperature={self.temperature}"
            f")"
        )