from code.loader import load_documents
from code.chunker import split_documents
from code.vectorstore import build_and_save_vectorstore


def main() -> None:
    print("=== BẮT ĐẦU BUILD INDEX ===")

    print("[1/3] Đang load tài liệu PDF...")
    documents = load_documents()
    print(f"Đã load {len(documents)} document(s).")

    print("[2/3] Đang chunk tài liệu...")
    chunks = split_documents(documents)
    print(f"Đã tạo {len(chunks)} chunk(s).")

    print("[3/3] Đang build và lưu FAISS index...")
    build_and_save_vectorstore(chunks)

    print("=== BUILD INDEX THÀNH CÔNG ===")


if __name__ == "__main__":
    main()