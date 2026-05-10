from __future__ import annotations

from typing import List, Dict, Any


DEFAULT_EMPTY_HISTORY = (
    "No previous conversation."
)


def normalize_whitespace(
    text: str
) -> str:
    """
    Normalize whitespace.

    Args:
        text:
            Input text.

    Returns:
        Cleaned text.
    """

    return " ".join(
        str(text).strip().split()
    )


def safe_role(
    turn: Dict[str, Any]
) -> str:
    """
    Extract and normalize role.

    Args:
        turn:
            Conversation turn.

    Returns:
        Normalized role.
    """

    role = str(
        turn.get("role", "user")
    ).strip().lower()

    role_mapping = {
        "human": "User",
        "user": "User",
        "assistant": "Assistant",
        "ai": "Assistant",
        "system": "System",
    }

    return role_mapping.get(
        role,
        role.capitalize()
    )


def safe_content(
    turn: Dict[str, Any]
) -> str:
    """
    Extract and normalize content.

    Args:
        turn:
            Conversation turn.

    Returns:
        Cleaned content.
    """

    content = turn.get(
        "content",
        ""
    )

    return normalize_whitespace(
        content
    )


def validate_turn(
    turn: Dict[str, Any]
) -> bool:
    """
    Validate conversation turn.

    Args:
        turn:
            Conversation turn.

    Returns:
        True if valid.
    """

    if not isinstance(turn, dict):
        return False

    content = safe_content(turn)

    if not content:
        return False

    return True


def format_turn(
    turn: Dict[str, Any]
) -> str:
    """
    Format single conversation turn.

    Example:
        User: What is RAG?

    Args:
        turn:
            Conversation turn.

    Returns:
        Formatted string.
    """

    role = safe_role(turn)

    content = safe_content(turn)

    return f"{role}: {content}"


def truncate_history(
    history: List[Dict[str, Any]],
    max_messages: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Truncate history.

    Args:
        history:
            Conversation history.

        max_messages:
            Maximum number of messages.

    Returns:
        Truncated history.
    """

    if (
        max_messages is None
        or max_messages <= 0
    ):
        return history

    return history[-max_messages:]


def format_history_for_rewrite(
    history: List[Dict[str, Any]],
    max_messages: int | None = None,
) -> str:
    """
    Format selected history
    for query rewriting prompts.

    Example:
        User: What is RAG?
        Assistant: Retrieval-Augmented Generation...

    Args:
        history:
            Selected history.

        max_messages:
            Maximum number of messages.

    Returns:
        Formatted history string.
    """

    if not history:
        return DEFAULT_EMPTY_HISTORY

    history = truncate_history(
        history,
        max_messages=max_messages,
    )

    formatted_lines = []

    for turn in history:

        if not validate_turn(turn):
            continue

        formatted_lines.append(
            format_turn(turn)
        )

    if not formatted_lines:
        return DEFAULT_EMPTY_HISTORY

    return "\n".join(
        formatted_lines
    )