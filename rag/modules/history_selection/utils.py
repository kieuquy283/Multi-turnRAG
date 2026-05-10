from typing import List, Dict, Any

import numpy as np


MEANINGLESS_PATTERNS = {
    "hi",
    "hello",
    "hey",
    "xin chào",
    "chào",
    "ok",
    "oke",
    "okay",
    "thanks",
    "thank you",
    "cảm ơn",
    "bye",
    "goodbye",
}


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.

    Args:
        text:
            Input text.

    Returns:
        Normalized text.
    """

    return text.strip().lower()


def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray
) -> float:
    """
    Compute cosine similarity.

    Args:
        a:
            First vector.

        b:
            Second vector.

    Returns:
        Cosine similarity score.
    """

    denominator = (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


def get_turn_content(
    turn: Dict[str, Any]
) -> str:
    """
    Safely extract turn content.

    Args:
        turn:
            Conversation turn.

    Returns:
        Content string.
    """

    return str(
        turn.get("content", "")
    ).strip()


def is_meaningful_turn(
    turn: Dict[str, Any],
    min_words: int = 3
) -> bool:
    """
    Determine whether a conversation turn
    is meaningful for query rewriting.

    Examples of meaningless turns:
        - hi
        - hello
        - ok
        - thank you

    Args:
        turn:
            Conversation turn.

        min_words:
            Minimum number of words required.

    Returns:
        True if meaningful.
    """

    content = normalize_text(
        get_turn_content(turn)
    )

    if not content:
        return False

    if content in MEANINGLESS_PATTERNS:
        return False

    if len(content.split()) < min_words:
        return False

    return True


def filter_meaningful_history(
    history: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Remove meaningless conversation turns.

    Args:
        history:
            Full conversation history.

    Returns:
        Filtered history.
    """

    return [
        turn
        for turn in history
        if is_meaningful_turn(turn)
    ]


def compute_recency_score(
    index: int,
    total_turns: int
) -> float:
    """
    Compute normalized recency score.

    More recent turns receive higher scores.

    Args:
        index:
            Turn index.

        total_turns:
            Total number of turns.

    Returns:
        Recency score in range [0, 1].
    """

    if total_turns <= 0:
        return 0.0

    return (index + 1) / total_turns


def rank_by_recency(
    history: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Rank history by recency.

    Most recent turns come first.

    Args:
        history:
            Conversation history.

    Returns:
        Ranked history.
    """

    return history[::-1]


def format_history(
    history: List[Dict[str, Any]]
) -> str:
    """
    Format history for prompts.

    Args:
        history:
            Selected history turns.

    Returns:
        Formatted string.
    """

    if not history:
        return "No previous conversation."

    lines = []

    for turn in history:

        role = str(
            turn.get("role", "user")
        ).capitalize()

        content = get_turn_content(turn)

        lines.append(
            f"{role}: {content}"
        )

    return "\n".join(lines)