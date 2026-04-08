from typing import List, Dict, Any
import re

from code.llm import get_llm
from code.config import HISTORY_TURNS, REWRITE_MODEL


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
    "nó", "đó", "điều đó", "việc đó", "cái đó", "họ", "ông ấy", "bà ấy",
    "loại nào", "trường hợp nào", "đối tượng nào", "cái nào", "mục nào"
]


def format_history_for_rewrite(history: List[Dict[str, Any]], max_turns: int = 3) -> str:
    """
    Format lịch sử hội thoại thành text ngắn gọn để phục vụ rewrite.
    """
    if not history:
        return "No previous conversation."

    selected = history[-max_turns * 2:]
    lines = []

    for msg in selected:
        role = str(msg.get("role", "user")).capitalize()
        content = str(msg.get("content", "")).strip()
        lines.append(f"{role}: {content}")

    return "\n".join(lines)


def is_likely_follow_up(question: str) -> bool:
    """
    Heuristic: xác định xem câu hỏi hiện tại có khả năng là follow-up không.
    Nếu không phải follow-up thì nên giữ nguyên để tránh rewrite làm hỏng retrieval.
    """
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
    """
    Làm sạch output của model để tránh các tiền tố/thừa dòng làm hỏng retrieval.
    """
    if not text:
        return ""

    cleaned = text.strip()

    # Bỏ markdown/code fence nếu có
    cleaned = cleaned.replace("```", "").strip()

    # Lấy dòng đầu tiên nếu model trả nhiều dòng
    cleaned = cleaned.splitlines()[0].strip()

    # Bỏ các tiền tố thường gặp
    prefixes = [
        "rewritten standalone query:",
        "standalone query:",
        "rewritten query:",
        "query:",
    ]

    lower_cleaned = cleaned.lower()
    for prefix in prefixes:
        if lower_cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            break

    # Bỏ dấu ngoặc kép bao ngoài
    cleaned = cleaned.strip("\"'“”‘’").strip()

    return cleaned


def rewrite_query(current_question: str, history: List[Dict[str, Any]]) -> str:
    """
    Rewrite câu hỏi hiện tại thành standalone query nếu thật sự cần.
    
    Logic an toàn:
    - Nếu không có history -> giữ nguyên
    - Nếu câu hỏi có vẻ đã standalone -> giữ nguyên
    - Chỉ rewrite khi có dấu hiệu là follow-up
    - Nếu output model bất thường -> fallback về câu gốc
    """
    question = current_question.strip()
    if not question:
        raise ValueError("current_question rỗng, không thể rewrite.")

    # Không có history thì không cần rewrite
    if not history:
        return question

    # Nếu câu hỏi có vẻ đã standalone thì giữ nguyên
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

    # fallback an toàn
    if not rewritten_query:
        return question

    # Nếu model trả quá dài, khả năng cao là rewrite không tốt
    if len(rewritten_query.split()) > max(20, len(question.split()) * 2):
        return question

    return rewritten_query