from code.config import *
from code.vectorstore import load_faiss
from code.retriever import retrieve_context
from code.llm import ask_llm

db = load_faiss(INDEX_DIR, EMBEDDING_MODEL)

print("Ready!")

while True:
    q = input("\n>> ")

    if q.lower() in ["exit", "quit"]:
        break

    docs = retrieve_context(db, q, TOP_K)

    ans = ask_llm(q, docs)

    print("\nANSWER:\n", ans)

    print("\nSOURCES:")
    for i, d in enumerate(docs, 1):
        print(f"{i}. {d.metadata.get('source_file')} | {d.metadata.get('article')}")

    print("=" * 80)