from __future__ import annotations

import re
from typing import Any, Dict, List


# =========================================================
# Constants
# =========================================================

DEFAULT_EMPTY_HISTORY = (
    "No previous conversation."
)


# =========================================================
# Explicit Follow-up Patterns
# =========================================================

FOLLOW_UP_PATTERNS = [

    # continuation
    r"^vậy",
    r"^thế",
    r"^còn",
    r"^thế còn",
    r"^nếu",
    r"^như vậy",
    r"^trường hợp",
    r"^với trường hợp",

    # references
    r"^cái đó",
    r"^điều đó",
    r"^việc đó",
    r"^nội dung đó",
    r"^quy định đó",
    r"^thông tin đó",

    # pronouns
    r"^nó",
    r"^họ",
    r"^ông ấy",
    r"^bà ấy",

    # incomplete dependent questions
    r"^loại nào",
    r"^mục nào",
    r"^đối tượng nào",
    r"^trường hợp nào",

    # ambiguous short questions
    r"^bao nhiêu",
    r"^bao lâu",
    r"^khi nào",
    r"^ở đâu",
    r"^bao giờ",
    r"^ai",

    # yes/no dependent
    r"^được không",
    r"^có được không",
]


# =========================================================
# Pronoun / Reference Keywords
# =========================================================

PRONOUN_KEYWORDS = [

    "nó",
    "đó",
    "điều đó",
    "việc đó",
    "cái đó",

    "họ",
    "ông ấy",
    "bà ấy",

    "loại nào",
    "mục nào",
    "đối tượng nào",
    "trường hợp nào",
    "cái nào",
]


# =========================================================
# Implicit Dependency Keywords
# =========================================================

IMPLICIT_DEPENDENCY_KEYWORDS = [

    # legal / policy continuation
    "mức phạt",
    "xử phạt",
    "quy định này",
    "quy định đó",
    "trường hợp này",
    "trường hợp trên",

    # continuation
    "cái này",
    "loại này",
    "vấn đề này",
    "nội dung này",

    # comparison
    "khác gì",
    "khác nhau",
    "giống nhau",

    # applicability
    "áp dụng",
    "có áp dụng",
    "được áp dụng",
]


# =========================================================
# Standalone Query Indicators
# =========================================================

STANDALONE_PATTERNS = [

    # english
    r"^what is",
    r"^how to",
    r"^define",
    r"^explain",

    # vietnamese
    r"^là gì",
    r"^giải thích",
    r"^định nghĩa",
    r"^hướng dẫn",
]


# =========================================================
# Generic Text Utilities
# =========================================================

def normalize_text(
    text: str
) -> str:
    """
    Normalize whitespace and lowercase text.
    """

    return " ".join(
        str(text).strip().lower().split()
    )


def safe_strip(
    value: Any
) -> str:
    """
    Safely convert to string and strip.
    """

    return str(value).strip()


# =========================================================
# Conversation Turn Utilities
# =========================================================

def get_turn_role(
    turn: Dict[str, Any]
) -> str:
    """
    Extract normalized role.
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


def get_turn_content(
    turn: Dict[str, Any]
) -> str:
    """
    Extract normalized content.
    """

    content = turn.get(
        "content",
        ""
    )

    return " ".join(
        str(content).strip().split()
    )


def validate_turn(
    turn: Dict[str, Any]
) -> bool:
    """
    Validate conversation turn.
    """

    if not isinstance(turn, dict):
        return False

    content = get_turn_content(
        turn
    )

    if not content:
        return False

    return True


# =========================================================
# History Formatting
# =========================================================

def truncate_history(
    history: List[Dict[str, Any]],
    max_turns: int | None = None,
) -> List[Dict[str, Any]]:
    """
    Truncate conversation history.
    """

    if (
        max_turns is None
        or max_turns <= 0
    ):
        return history

    return history[-max_turns * 2 :]


def format_turn(
    turn: Dict[str, Any]
) -> str:
    """
    Format single turn.
    """

    role = get_turn_role(
        turn
    )

    content = get_turn_content(
        turn
    )

    return f"{role}: {content}"


def format_history_for_rewrite(
    history: List[Dict[str, Any]],
    max_turns: int = 3,
) -> str:
    """
    Format selected history
    for rewrite prompts.
    """

    if not history:
        return DEFAULT_EMPTY_HISTORY

    selected = truncate_history(
        history,
        max_turns=max_turns,
    )

    lines = []

    for turn in selected:

        if not validate_turn(turn):
            continue

        lines.append(
            format_turn(turn)
        )

    if not lines:
        return DEFAULT_EMPTY_HISTORY

    return "\n".join(lines)


# =========================================================
# Regex / Keyword Utilities
# =========================================================

def contains_pattern(
    text: str,
    patterns: List[str]
) -> bool:
    """
    Check regex patterns.
    """

    for pattern in patterns:

        if re.search(pattern, text):
            return True

    return False


def contains_keyword(
    text: str,
    keywords: List[str]
) -> bool:
    """
    Check keyword existence.
    """

    for keyword in keywords:

        if keyword in text:
            return True

    return False


# =========================================================
# Query Heuristics
# =========================================================

def is_short_query(
    query: str,
    threshold: int = 6,
) -> bool:
    """
    Short queries are more likely
    to depend on history.
    """

    return (
        len(query.split())
        <= threshold
    )


def is_long_query(
    query: str,
    threshold: int = 18,
) -> bool:
    """
    Long queries are usually
    standalone enough.
    """

    return (
        len(query.split())
        >= threshold
    )


def has_named_entity(
    query: str
) -> bool:
    """
    Simple heuristic for detecting
    capitalized entities.
    """

    words = query.split()

    capitalized = [

        word

        for word in words

        if len(word) > 1
        and word[0].isupper()
    ]

    return len(capitalized) > 0


# =========================================================
# Follow-up Detection
# =========================================================

def is_likely_follow_up(
    query: str
) -> bool:
    """
    Determine whether query
    likely depends on conversation history.

    Strategy:
        1. Explicit patterns
        2. Pronoun/reference detection
        3. Implicit dependency
        4. Query length heuristics
        5. Standalone detection
    """

    query = normalize_text(
        query
    )

    if not query:
        return False

    # =====================================================
    # Explicit dependency
    # =====================================================

    if contains_pattern(
        query,
        FOLLOW_UP_PATTERNS,
    ):
        return True

    # =====================================================
    # Pronoun/reference dependency
    # =====================================================

    if contains_keyword(
        query,
        PRONOUN_KEYWORDS,
    ):
        return True

    # =====================================================
    # Implicit dependency
    # =====================================================

    if contains_keyword(
        query,
        IMPLICIT_DEPENDENCY_KEYWORDS,
    ):
        return True

    # =====================================================
    # Short query heuristic
    # =====================================================

    if is_short_query(query):

        if not has_named_entity(query):
            return True

    # =====================================================
    # Long standalone query
    # =====================================================

    if is_long_query(query):
        return False

    # =====================================================
    # Standalone patterns
    # =====================================================

    if contains_pattern(
        query,
        STANDALONE_PATTERNS,
    ):
        return False

    return False


# =========================================================
# Rewrite Output Cleaning
# =========================================================

def clean_rewritten_query(
    text: str
) -> str:
    """
    Clean rewritten query output.
    """

    if not text:
        return ""

    cleaned = safe_strip(text)

    cleaned = (
        cleaned
        .replace("```", "")
        .strip()
    )

    cleaned = (
        cleaned
        .splitlines()[0]
        .strip()
    )

    prefixes = [

        "rewritten standalone query:",
        "standalone query:",
        "rewritten query:",
        "query:",
    ]

    lower_cleaned = (
        cleaned.lower()
    )

    for prefix in prefixes:

        if lower_cleaned.startswith(
            prefix
        ):

            cleaned = cleaned[
                len(prefix):
            ].strip()

            break

    cleaned = (
        cleaned
        .strip("\"'“”‘’")
        .strip()
    )

    return cleaned


# =========================================================
# Rewrite Decision Helpers
# =========================================================

def should_skip_rewrite(
    history: List[Dict[str, Any]]
) -> bool:
    """
    Determine whether rewriting
    should be skipped.
    """

    if not history:
        return True

    return False