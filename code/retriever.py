from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from code.config import TOP_K


def retrieve_documents(
    query: str,
    vectorstore: FAISS,
    top_k: int | None = None,
) -> List[Document]:
    """
    Retrieve top-k documents từ vectorstore theo query, kèm raw_score trong metadata.

    Lưu ý:
    - Với FAISS similarity_search_with_score, score thường là distance.
    - Distance càng thấp thì càng gần.
    """
    if not query or not query.strip():
        raise ValueError("Query rỗng, không thể retrieve.")

    k = top_k or TOP_K
    docs_and_scores = vectorstore.similarity_search_with_score(query, k=k)

    results: List[Document] = []
    for doc, score in docs_and_scores:
        metadata = dict(doc.metadata or {})
        metadata["raw_score"] = float(score)

        # metadata từ index của bạn đang dùng source_file
        if "source_file" not in metadata:
            metadata["source_file"] = metadata.get("source", "unknown")

        # giữ lại object Document mới để metadata chắc chắn có raw_score
        results.append(
            Document(
                page_content=doc.page_content,
                metadata=metadata,
            )
        )

    # sort lại cho chắc: distance thấp hơn là tốt hơn
    results.sort(key=lambda d: float(d.metadata.get("raw_score", 1e9)))
    return results


def filter_active_docs(docs: List[Document], top_k: int | None = None) -> List[Document]:
    """
    Chỉ giữ các chunk đang active.
    """
    filtered = [doc for doc in docs if doc.metadata.get("is_active", True)]
    if top_k is not None:
        return filtered[:top_k]
    return filtered


def build_top_files(docs: List[Document], top_k_files: int = 3) -> List[Dict[str, Any]]:
    """
    Gom chunk theo file nguồn.
    Vì raw_score là distance nên:
    - best_score = min(raw_score)
    - avg_score = trung bình distance
    """
    grouped: Dict[str, List[float]] = defaultdict(list)

    for doc in docs:
        source_file = doc.metadata.get("source_file", "unknown")
        raw_score = float(doc.metadata.get("raw_score", 1e9))
        grouped[source_file].append(raw_score)

    results: List[Dict[str, Any]] = []
    for source_file, scores in grouped.items():
        results.append(
            {
                "source_file": source_file,
                "best_score": min(scores),
                "avg_score": sum(scores) / len(scores),
                "hits": len(scores),
            }
        )

    # Vì là distance nên score nhỏ hơn là tốt hơn
    results.sort(key=lambda x: (x["best_score"], -x["hits"]))
    return results[:top_k_files]