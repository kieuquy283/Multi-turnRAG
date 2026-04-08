# formatter.py

from langchain_core.documents import Document
from typing import List


def strip_passage_prefix(text: str) -> str:
    if text.startswith("passage: "):
        return text[len("passage: "):]
    return text


def format_context(docs: List[Document], max_chunks=5) -> str:
    if not docs:
        return "Không có ngữ cảnh phù hợp."

    blocks = []

    for i, d in enumerate(docs[:max_chunks], 1):
        source_file = d.metadata.get("source_file", "unknown")
        article = d.metadata.get("article", "N/A")
        content = strip_passage_prefix(d.page_content)

        block = (
            f"[Nguồn {i}]\n"
            f"Tệp: {source_file}\n"
            f"Điều: {article}\n"
            f"Nội dung:\n{content}"
        )

        blocks.append(block)

    return "\n\n".join(blocks)