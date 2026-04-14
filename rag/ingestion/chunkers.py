from __future__ import annotations

import re
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag.config.retrieval import CHUNK_OVERLAP, CHUNK_SIZE
from rag.ingestion.readers import read_document


def split_legal_articles(text: str) -> List[str]:
    normalized = re.sub(r"\r\n?", "\n", text).strip()
    if not normalized:
        return []

    pattern = re.compile(
        r"(?im)^\s*(Điều\s+\d+\s*[\.:]?\s*.*?)(?=^\s*Điều\s+\d+\s*[\.:]?\s*|\Z)",
        re.DOTALL,
    )
    matches = pattern.findall(normalized)
    return [m.strip() for m in matches if m and m.strip()]


def build_text_splitter(
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def fallback_text_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    splitter = build_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    return [c.strip() for c in chunks if c and c.strip()]


def chunk_legal_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    article_chunks = split_legal_articles(text)

    if len(article_chunks) >= 2:
        final_chunks: List[str] = []
        for chunk in article_chunks:
            if len(chunk) <= chunk_size:
                final_chunks.append(chunk)
            else:
                final_chunks.extend(
                    fallback_text_chunks(
                        chunk,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                )
        return [c for c in final_chunks if c.strip()]

    return fallback_text_chunks(
        text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


def chunk_document(
    file_path: str | Path,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    text = read_document(file_path)
    return chunk_legal_text(
        text=text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
