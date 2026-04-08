import re
from typing import List, Dict
from langchain_core.documents import Document


def clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)

    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]

    return "\n".join(lines).strip()


def split_legal_text_by_articles(text: str) -> List[Dict]:
    text = clean_text(text)

    article_pattern = re.compile(
        r"(^|\n)(Điều\s+\d+[\.\:]?\s+.*)",
        flags=re.MULTILINE
    )

    matches = list(article_pattern.finditer(text))

    if not matches:
        return [{"article": "N/A", "clause": None, "content": text}]

    chunks = []

    for i, match in enumerate(matches):
        start = match.start(2)
        end = matches[i + 1].start(2) if i + 1 < len(matches) else len(text)

        chunk = text[start:end].strip()
        first_line = chunk.split("\n", 1)[0].strip()

        chunks.append({
            "article": first_line,
            "clause": None,
            "content": chunk
        })

    return chunks


def build_legal_chunks(page_docs: List[Document]) -> List[Document]:
    docs_by_file: Dict[str, List[Document]] = {}

    for doc in page_docs:
        source_file = doc.metadata.get("source_file", "unknown.pdf")
        docs_by_file.setdefault(source_file, []).append(doc)

    final_chunks: List[Document] = []

    for source_file, docs in docs_by_file.items():
        docs = sorted(docs, key=lambda x: x.metadata.get("page", 0))
        full_text = "\n".join(clean_text(d.page_content) for d in docs)
        split_chunks = split_legal_text_by_articles(full_text)

        for idx, item in enumerate(split_chunks):
            content = item["content"].strip()
            if not content:
                continue

            passage_text = f"passage: {content}"

            final_chunks.append(
                Document(
                    page_content=passage_text,
                    metadata={
                        "source_file": source_file,
                        "article": item["article"],
                        "chunk_id": idx
                    }
                )
            )

    return final_chunks