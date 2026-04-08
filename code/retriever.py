from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from code.config import TOP_K


def get_retriever(vectorstore: FAISS):
    """
    Tạo retriever từ FAISS vectorstore.
    """
    return vectorstore.as_retriever(
        search_kwargs={"k": TOP_K}
    )


def retrieve_documents(query: str, vectorstore: FAISS) -> List[Document]:
    """
    Retrieve top-k documents từ vectorstore theo query.

    Args:
        query: câu truy vấn
        vectorstore: FAISS vectorstore

    Returns:
        List[Document]: danh sách tài liệu liên quan
    """
    if not query or not query.strip():
        raise ValueError("Query rỗng, không thể retrieve.")

    retriever = get_retriever(vectorstore)
    docs = retriever.invoke(query)
    return docs