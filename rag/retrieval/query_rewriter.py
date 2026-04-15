from __future__ import annotations

import re
from typing import Any, Dict, List

from rag.config.llm import REWRITE_MODEL
from rag.config.retrieval import HISTORY_TURNS
from rag.generation.llm_client import get_llm


FOLLOW_UP_PATTERNS = [
    r"^vậy",
    r"^thế",
    r"^còn",
    r"^nếu",
    r"^như vậy",
    r"^trường hợp",
    r"^loại nào",
    r"^khi nào",
    r"^ở đâu",
    r"^bao giờ",
    r"^bao nhiêu",
    r"^ai",
    r"^đối tượng nào",
    r"^cái đó",
    r"^điều đó",
    r"^việc đó",
    r"^nó",
    r"^họ",
    r"^được không",
    r"^có được không",
]

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
    "trường hợp nào",
    "đối tượng nào",
    "cái nào",
    "mục nào",
]


def format_history_for_rewrite(
    history: List[Dict[str, Any]],
    max_turns: int = HISTORY_TURNS,
) -> str:
    if not history:
        return "No previous conversation."

    selected = history[-max_turns * 2 :]
    lines = []

    for msg in selected:
        role = str(msg.get("role", "user")).capitalize()
        content = str(msg.get("content", "")).strip()
        lines.append(f"{role}: {content}")

    return "\n".join(lines)


def is_likely_follow_up(question: str) -> bool:
    q = question.strip().lower()
    if not q:
        return False

    for pattern in FOLLOW_UP_PATTERNS:
        if re.search(pattern, q):
            return True

    for keyword in PRONOUN_KEYWORDS:
        if keyword in q:
            return True

    return False


def clean_rewritten_query(text: str) -> str:
    if not text:
        return ""

    cleaned = text.strip()
    cleaned = cleaned.replace("```", "").strip()
    cleaned = cleaned.splitlines()[0].strip()

    prefixes = [
        "rewritten standalone query:",
        "standalone query:",
        "rewritten query:",
        "query:",
    ]

    lower_cleaned = cleaned.lower()
    for prefix in prefixes:
        if lower_cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
            break

    cleaned = cleaned.strip("\"'“”‘’").strip()
    return cleaned


def rewrite_query(current_question: str, history: List[Dict[str, Any]]) -> str:
    question = current_question.strip()
    if not question:
        raise ValueError("current_question rỗng, không thể rewrite.")

    if not history:
        return question

    if not is_likely_follow_up(question):
        return question

    history_text = format_history_for_rewrite(history, max_turns=HISTORY_TURNS)

    prompt = f"""
You are a query rewriter for a multi-turn RAG system.

Your job is to rewrite the user's latest question into ONE short standalone search query.

Important rules:
- Preserve the meaning exactly.
- Only add missing context from conversation history when necessary.
- Do not answer the question.
- Do not explain anything.
- Do not add extra details not present in the conversation.
- Keep the rewritten query natural, concise, and retrieval-friendly.
- Output only the rewritten query, with no label or commentary.
- If the current question is already standalone enough for retrieval, return it unchanged.

Conversation history:
{history_text}

Current user question:
{question}

Standalone search query:
""".strip()

    llm = get_llm(model_name=REWRITE_MODEL, temperature=0.0)
    response = llm.invoke(prompt)
    rewritten_query = clean_rewritten_query(response.content)

    if not rewritten_query:
        return question

    if len(rewritten_query.split()) > max(20, len(question.split()) * 2):
        return question

    return rewritten_query
