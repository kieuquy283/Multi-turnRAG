from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from code.config import CHUNK_SIZE, CHUNK_OVERLAP


def split_documents(documents: List[Document]) -> List[Document]:
    """
    Chia documents thành các chunks nhỏ để embedding và retrieval hiệu quả hơn.

    Args:
        documents: danh sách document gốc

    Returns:
        List[Document]: danh sách chunked documents
    """
    if not documents:
        raise ValueError("split_documents nhận documents rỗng.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    # Thêm metadata chunk_id nếu muốn debug tốt hơn
    for idx, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = idx

    return chunks