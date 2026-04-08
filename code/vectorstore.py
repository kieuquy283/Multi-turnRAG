from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from code.config import OPENAI_API_KEY, EMBEDDING_MODEL, INDEX_DIR


def get_embeddings() -> OpenAIEmbeddings:
    """
    Khởi tạo embedding model.
    """
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY
    )


def ensure_index_dir() -> None:
    """
    Tạo thư mục index nếu chưa có.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)


def build_and_save_vectorstore(documents: List[Document]) -> FAISS:
    """
    Build FAISS vectorstore từ documents và lưu xuống disk.

    Args:
        documents: danh sách chunks

    Returns:
        FAISS: vectorstore đã build
    """
    if not documents:
        raise ValueError("Không có documents để build vectorstore.")

    ensure_index_dir()
    embeddings = get_embeddings()

    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(str(INDEX_DIR))

    return vectorstore


def load_vectorstore() -> FAISS:
    """
    Load FAISS vectorstore từ disk.

    Returns:
        FAISS: vectorstore đã lưu
    """
    if not INDEX_DIR.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục index: {INDEX_DIR}\n"
            f"Hãy chạy: python -m code.build_index"
        )

    index_files = list(Path(INDEX_DIR).glob("*"))
    if not index_files:
        raise FileNotFoundError(
            f"Thư mục index rỗng: {INDEX_DIR}\n"
            f"Hãy chạy: python -m code.build_index"
        )

    embeddings = get_embeddings()

    vectorstore = FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore