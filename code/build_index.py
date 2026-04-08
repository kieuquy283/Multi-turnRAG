from __future__ import annotations

import json
import hashlib
import os
import re
import uuid
from dotenv import load_dotenv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import docx2txt
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


# =========================
# Config
# =========================
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_CHUNK_SIZE = 1200
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_INDEX_DIRNAME = "vector_store"
MANIFEST_FILENAME = "manifest.json"
CHUNKS_METADATA_FILENAME = "chunks_metadata.json"
load_dotenv()


# =========================
# Helpers
# =========================
def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def file_sha256(file_path: str | Path) -> str:
    sha = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_json(file_path: str | Path, default: Any) -> Any:
    path = Path(file_path)
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(file_path: str | Path, data: Any) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def scan_document_files(data_dir: str | Path) -> List[str]:
    data_dir = Path(data_dir)
    files: List[str] = []
    for root, _, filenames in os.walk(data_dir):
        for filename in filenames:
            ext = Path(filename).suffix.lower()
            if ext in SUPPORTED_EXTENSIONS:
                files.append(str(Path(root) / filename))
    return sorted(files)


def index_exists(index_dir: str | Path) -> bool:
    index_dir = Path(index_dir)
    return (
        (index_dir / "index.faiss").exists()
        and (index_dir / "index.pkl").exists()
        and (index_dir / MANIFEST_FILENAME).exists()
        and (index_dir / CHUNKS_METADATA_FILENAME).exists()
    )


# =========================
# Reading documents
# =========================
# def read_pdf(file_path: str | Path) -> str:
#     loader = PyPDFLoader(str(file_path))
#     docs = loader.load()
#     return "\n\n".join(doc.page_content for doc in docs if doc.page_content)

def read_pdf(file_path: str | Path) -> str:
    # Cách 1: PyPDFLoader
    try:
        loader = PyPDFLoader(str(file_path))
        docs = loader.load()
        text = "\n\n".join(doc.page_content for doc in docs if doc.page_content)
        if text and text.strip():
            print(f"[DEBUG] PyPDFLoader extracted {len(text)} chars")
            return text
    except Exception as e:
        print(f"[DEBUG] PyPDFLoader failed: {e}")

    # Cách 2: pdfplumber fallback
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(str(file_path)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    pages.append(page_text)
        text = "\n\n".join(pages)
        if text and text.strip():
            print(f"[DEBUG] pdfplumber extracted {len(text)} chars")
            return text
    except Exception as e:
        print(f"[DEBUG] pdfplumber failed: {e}")

    # Cách 3: PyMuPDF fallback
    try:
        import fitz  # pymupdf
        doc = fitz.open(str(file_path))
        pages = []
        for page in doc:
            page_text = page.get_text("text") or ""
            if page_text.strip():
                pages.append(page_text)
        text = "\n\n".join(pages)
        if text and text.strip():
            print(f"[DEBUG] PyMuPDF extracted {len(text)} chars")
            return text
    except Exception as e:
        print(f"[DEBUG] PyMuPDF failed: {e}")

    print(f"[WARNING] Could not extract text from PDF: {file_path}")
    return ""


def read_docx(file_path: str | Path) -> str:
    return docx2txt.process(str(file_path)) or ""


def read_text(file_path: str | Path) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def read_document(file_path: str | Path) -> str:
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return read_pdf(file_path)
    if ext == ".docx":
        return read_docx(file_path)
    if ext in {".txt", ".md"}:
        return read_text(file_path)
    raise ValueError(f"Unsupported file type: {ext}")


# =========================
# Chunking
# =========================
def split_legal_articles(text: str) -> List[str]:
    """
    Ưu tiên tách theo cấu trúc pháp luật kiểu:
    Điều 1.
    Điều 2.

    Nếu không match đủ tốt thì fallback sang splitter thường.
    """
    normalized = re.sub(r"\r\n?", "\n", text).strip()
    if not normalized:
        return []

    # Match "Điều 1.", "Điều 12:", kể cả đầu dòng có khoảng trắng.
    pattern = re.compile(r"(?im)^\s*(Điều\s+\d+[\.:].*?)(?=^\s*Điều\s+\d+[\.:]|\Z)", re.DOTALL)
    matches = pattern.findall(normalized)

    cleaned = [m.strip() for m in matches if m and m.strip()]
    return cleaned


def fallback_text_chunks(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    return [c.strip() for c in chunks if c and c.strip()]


def chunk_document(
    file_path: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    text = read_document(file_path)
    article_chunks = split_legal_articles(text)

    # Nếu tài liệu pháp luật tách được các Điều tương đối ổn thì dùng luôn.
    if len(article_chunks) >= 2:
        final_chunks: List[str] = []
        for chunk in article_chunks:
            if len(chunk) <= chunk_size:
                final_chunks.append(chunk)
            else:
                final_chunks.extend(fallback_text_chunks(chunk, chunk_size, chunk_overlap))
        return [c for c in final_chunks if c.strip()]

    return fallback_text_chunks(text, chunk_size, chunk_overlap)


# =========================
# Metadata
# =========================
def make_chunk_metadata(
    chunk_id: str,
    source_file: str,
    file_hash: str,
    chunk_text: str,
    chunk_index: int,
) -> Dict[str, Any]:
    return {
        "chunk_id": chunk_id,
        "source_file": source_file,
        "file_hash": file_hash,
        "content_hash": text_sha256(chunk_text),
        "chunk_index": chunk_index,
        "is_active": True,
        "created_at": utc_now_iso(),
        "text": chunk_text,
    }


def deactivate_chunks_for_file(metadata: List[Dict[str, Any]], source_file: str) -> int:
    changed = 0
    for item in metadata:
        if item.get("source_file") == source_file and item.get("is_active", True):
            item["is_active"] = False
            item["deactivated_at"] = utc_now_iso()
            changed += 1
    return changed


def build_chunks_and_metadata_for_file(
    file_path: str,
    chunk_size: int,
    chunk_overlap: int,
) -> Tuple[List[str], List[Dict[str, Any]], str]:
    f_hash = file_sha256(file_path)
    chunks = chunk_document(file_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    metadatas: List[Dict[str, Any]] = []
    for i, chunk_text in enumerate(chunks):
        chunk_id = str(uuid.uuid4())
        metadatas.append(
            make_chunk_metadata(
                chunk_id=chunk_id,
                source_file=file_path,
                file_hash=f_hash,
                chunk_text=chunk_text,
                chunk_index=i,
            )
        )

    return chunks, metadatas, f_hash


# =========================
# Embeddings
# =========================
def get_embeddings(model: str = DEFAULT_EMBEDDING_MODEL) -> OpenAIEmbeddings:
    # Cần biến môi trường OPENAI_API_KEY
    return OpenAIEmbeddings(model=model)


# =========================
# Main functions
# =========================
def build_index_from_documents(
    data_dir: str,
    index_dir: str = DEFAULT_INDEX_DIRNAME,
    embedding_model: Optional[OpenAIEmbeddings] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> None:
    """
    Build full index từ đầu.

    Dùng khi:
    - Chưa có index
    - Hoặc muốn rebuild sạch toàn bộ
    """
    embedding_model = embedding_model or get_embeddings()
    ensure_dir(index_dir)

    manifest_path = Path(index_dir) / MANIFEST_FILENAME
    metadata_path = Path(index_dir) / CHUNKS_METADATA_FILENAME

    files = scan_document_files(data_dir)
    if not files:
        raise ValueError(f"No supported documents found in: {data_dir}")

    all_texts: List[str] = []
    all_metadatas: List[Dict[str, Any]] = []
    manifest: Dict[str, Any] = {}

    for file_path in files:
        texts, metadatas, f_hash = build_chunks_and_metadata_for_file(
            file_path=file_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        all_texts.extend(texts)
        all_metadatas.extend(metadatas)
        manifest[file_path] = {
            "file_hash": f_hash,
            "last_indexed_at": utc_now_iso(),
        }
        print(f"[BUILD] {file_path} -> {len(texts)} chunks")

    if not all_texts:
        raise ValueError("No chunks generated from the provided documents.")

    vectorstore = FAISS.from_texts(
        texts=all_texts,
        embedding=embedding_model,
        metadatas=all_metadatas,
    )
    vectorstore.save_local(index_dir)

    save_json(manifest_path, manifest)
    save_json(metadata_path, all_metadatas)

    print(f"[BUILD DONE] files={len(files)}, chunks={len(all_texts)}, index_dir={index_dir}")



def update_index_from_documents(
    data_dir: str,
    index_dir: str = DEFAULT_INDEX_DIRNAME,
    embedding_model: Optional[OpenAIEmbeddings] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> None:
    """
    Update incremental.

    Xử lý:
    - file mới   -> add mới
    - file sửa   -> deactivate chunk cũ + add chunk mới
    - file xóa   -> deactivate chunk cũ
    """
    embedding_model = embedding_model or get_embeddings()

    if not index_exists(index_dir):
        raise ValueError(
            f"Index not found in '{index_dir}'. Run build_index_from_documents first."
        )

    manifest_path = Path(index_dir) / MANIFEST_FILENAME
    metadata_path = Path(index_dir) / CHUNKS_METADATA_FILENAME

    vectorstore = FAISS.load_local(
        index_dir,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )

    manifest: Dict[str, Any] = load_json(manifest_path, {})
    metadata: List[Dict[str, Any]] = load_json(metadata_path, [])

    current_files = scan_document_files(data_dir)
    current_file_set = set(current_files)
    old_file_set = set(manifest.keys())

    new_texts: List[str] = []
    new_metadatas: List[Dict[str, Any]] = []

    added_files = 0
    updated_files = 0
    deleted_files = 0
    deactivated_chunks = 0

    for file_path in current_files:
        current_hash = file_sha256(file_path)

        # File mới
        if file_path not in manifest:
            texts, metadatas, f_hash = build_chunks_and_metadata_for_file(
                file_path=file_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            new_texts.extend(texts)
            new_metadatas.extend(metadatas)
            manifest[file_path] = {
                "file_hash": f_hash,
                "last_indexed_at": utc_now_iso(),
            }
            added_files += 1
            print(f"[UPDATE][NEW] {file_path} -> {len(texts)} chunks")
            continue

        # File bị sửa
        if manifest[file_path].get("file_hash") != current_hash:
            deactivated_chunks += deactivate_chunks_for_file(metadata, file_path)

            texts, metadatas, f_hash = build_chunks_and_metadata_for_file(
                file_path=file_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            new_texts.extend(texts)
            new_metadatas.extend(metadatas)
            manifest[file_path] = {
                "file_hash": f_hash,
                "last_indexed_at": utc_now_iso(),
            }
            updated_files += 1
            print(f"[UPDATE][MODIFIED] {file_path} -> {len(texts)} new chunks")

    # File bị xóa khỏi kho dữ liệu
    removed_files = old_file_set - current_file_set
    for file_path in sorted(removed_files):
        deactivated_chunks += deactivate_chunks_for_file(metadata, file_path)
        manifest.pop(file_path, None)
        deleted_files += 1
        print(f"[UPDATE][DELETED] {file_path}")

    # Add vector mới vào index
    if new_texts:
        vectorstore.add_texts(texts=new_texts, metadatas=new_metadatas)
        metadata.extend(new_metadatas)

    vectorstore.save_local(index_dir)
    save_json(manifest_path, manifest)
    save_json(metadata_path, metadata)

    print(
        "[UPDATE DONE] "
        f"added_files={added_files}, "
        f"updated_files={updated_files}, "
        f"deleted_files={deleted_files}, "
        f"deactivated_chunks={deactivated_chunks}, "
        f"new_chunks={len(new_texts)}"
    )


# =========================
# Optional helper for retrieval phase
# =========================
def filter_active_docs(docs: Sequence[Any], top_k: int = 5) -> List[Any]:
    filtered: List[Any] = []
    for doc in docs:
        if getattr(doc, "metadata", {}).get("is_active", True):
            filtered.append(doc)
        if len(filtered) >= top_k:
            break
    return filtered


# =========================
# CLI
# =========================
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build or update FAISS index for RAG documents.")
    parser.add_argument("mode", choices=["build", "update"], help="Run full build or incremental update.")
    parser.add_argument("--data-dir", default="data", help="Directory containing source documents.")
    parser.add_argument("--index-dir", default=DEFAULT_INDEX_DIRNAME, help="Directory to store FAISS index and metadata.")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, help="Embedding model name.")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Chunk size for fallback splitter.")
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Chunk overlap for fallback splitter.")

    args = parser.parse_args()
    embeddings = get_embeddings(args.embedding_model)

    if args.mode == "build":
        build_index_from_documents(
            data_dir=args.data_dir,
            index_dir=args.index_dir,
            embedding_model=embeddings,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    else:
        update_index_from_documents(
            data_dir=args.data_dir,
            index_dir=args.index_dir,
            embedding_model=embeddings,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )


if __name__ == "__main__":
    main()
