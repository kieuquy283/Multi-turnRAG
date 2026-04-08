from typing import List, Dict, Any
from langchain_core.documents import Document


def format_docs(docs: List[Document]) -> str:
    """
    Format retrieved docs thành context text.

    Args:
        docs: danh sách retrieved docs

    Returns:
        str: context đã format
    """
    if not docs:
        return "Không tìm thấy tài liệu liên quan."

    formatted_parts = []

    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source_file", "unknown")
        page = doc.metadata.get("page", "unknown")
        chunk_id = doc.metadata.get("chunk_id", "unknown")
        content = doc.page_content.strip()

        block = (
            f"[Document {i} | Source: {source} | Page: {page} | Chunk: {chunk_id}]\n"
            f"{content}"
        )
        formatted_parts.append(block)

    return "\n\n".join(formatted_parts)


def format_recent_history(history: List[Dict[str, Any]], max_turns: int = 3) -> str:
    """
    Format vài lượt hội thoại gần nhất để đưa vào prompt trả lời.

    Args:
        history: danh sách message
        max_turns: số turns gần nhất

    Returns:
        str
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


def build_answer_prompt(
    current_question: str,
    rewritten_query: str,
    docs: List[Document],
    history: List[Dict[str, Any]],
    max_turns: int = 3
) -> str:
    """
    Xây prompt cuối cho answer generation.

    Args:
        current_question: câu hỏi người dùng vừa nhập
        rewritten_query: câu truy vấn độc lập đã rewrite
        docs: tài liệu retrieve được
        history: lịch sử hội thoại
        max_turns: số turns gần nhất đưa vào prompt

    Returns:
        str
    """
    recent_history = format_recent_history(history, max_turns=max_turns)
    context = format_docs(docs)

    prompt = f"""
Bạn là trợ lý AI trả lời câu hỏi dựa trên tài liệu được truy xuất.

Nguyên tắc bắt buộc:
1. Chỉ sử dụng thông tin có trong phần Context.
2. Nếu Context không đủ thông tin để trả lời chính xác, phải nói rõ là không tìm thấy đủ thông tin trong tài liệu.
3. Không bịa, không suy đoán vượt quá dữ liệu.
4. Trả lời rõ ràng, đúng trọng tâm, dễ hiểu.

Lịch sử hội thoại gần đây:
{recent_history}

Câu hỏi hiện tại của người dùng:
{current_question}

Truy vấn độc lập dùng để tìm kiếm:
{rewritten_query}

Context:
{context}

Hãy trả lời bằng tiếng Việt.
""".strip()

    return prompt