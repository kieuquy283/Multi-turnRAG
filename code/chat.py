from typing import List, Dict, Any

from code.vectorstore import load_vectorstore
from code.retriever import retrieve_documents
from code.query_rewriter import rewrite_query
from code.formatter import build_answer_prompt
from code.llm import generate_answer
from code.config import HISTORY_TURNS, SHOW_REWRITTEN_QUERY


def main() -> None:
    print("Đang load FAISS index...")
    vectorstore = load_vectorstore()

    history: List[Dict[str, Any]] = []

    print("=== MULTI-TURN RAG CHATBOT ===")
    print("Gõ 'exit' hoặc 'quit' để thoát.\n")

    while True:
        question = input("Bạn: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Thoát chương trình.")
            break

        if not question:
            print("Vui lòng nhập câu hỏi.")
            continue

        try:
            # 1) Rewrite query cho multi-turn retrieval
            rewritten_query = rewrite_query(
                current_question=question,
                history=history
            )
            # rewritten_query = question

            if SHOW_REWRITTEN_QUERY:
                print(f"\n[Rewritten Query]: {rewritten_query}\n")

            # 2) Retrieve bằng rewritten query
            docs = retrieve_documents(
                query=rewritten_query,
                vectorstore=vectorstore
            )

            # 3) Build answer prompt
            prompt = build_answer_prompt(
                current_question=question,
                rewritten_query=rewritten_query,
                docs=docs,
                history=history,
                max_turns=HISTORY_TURNS
            )

            # 4) Generate answer
            answer = generate_answer(prompt)

            print(f"Bot: {answer}\n")

            # 5) Update history
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})

        except Exception as e:
            print(f"\n[ERROR] {e}\n")


if __name__ == "__main__":
    main()