from pathlib import Path
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader

from code.config import DATA_DIR, SUPPORTED_EXTENSIONS


def _get_pdf_files(data_dir: Path) -> List[Path]:
    """
    Lấy toàn bộ file PDF trong thư mục data.
    """
    files: List[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(data_dir.glob(f"*{ext}"))
    return sorted(files)


def load_documents() -> List[Document]:
    """
    Load tất cả PDF trong thư mục data/ thành danh sách Document.

    Returns:
        List[Document]: danh sách document đã load, có metadata nguồn file.
    """
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục data: {DATA_DIR}\n"
            f"Hãy tạo thư mục 'data' ở root project và thêm file PDF vào đó."
        )

    pdf_files = _get_pdf_files(DATA_DIR)
    if not pdf_files:
        raise FileNotFoundError(
            f"Không tìm thấy file PDF nào trong: {DATA_DIR}"
        )

    all_documents: List[Document] = []

    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()

        for doc in docs:
            # Chuẩn hóa metadata để các file khác dùng dễ hơn
            doc.metadata["source_file"] = pdf_file.name
            doc.metadata["source_path"] = str(pdf_file)

        all_documents.extend(docs)

    return all_documents