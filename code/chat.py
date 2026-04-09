from __future__ import annotations

from typing import List, Dict, Any

from code.vectorstore import load_vectorstore
from code.retriever import retrieve_documents, filter_active_docs, build_top_files
from code.query_rewriter import rewrite_query
from code.formatter import build_answer_prompt
from code.llm import generate_answer
from code.config import HISTORY_TURNS, SHOW_REWRITTEN_QUERY, TOP_K


INSUFFICIENT_SENTINEL = "__INSUFFICIENT_CONTEXT__"


def build_fallback_prompt(question: str, history: List[Dict[str, Any]]) -> str:
    history_text = ""
    if history:
        recent = history[-HISTORY_TURNS * 2:]
        history_text = "\n".join(
            f"{msg.get('role', 'user').capitalize()}: {msg.get('content', '')}"
            for msg in recent
        )

    return f"""
Bạn là trợ lý AI hữu ích.

Kho tri thức hiện tại không có đủ tài liệu liên quan hoặc không đủ ngữ cảnh để trả lời chắc chắn câu hỏi của người dùng.

Hãy trả lời câu hỏi bằng kiến thức nền của bạn một cách hữu ích, rõ ràng, trung thực.
Không được nói rằng bạn có tài liệu nếu thực tế không có.
Nếu có điểm chưa chắc chắn, hãy nói rõ.

Lịch sử hội thoại gần đây:
{history_text if history_text else "Không có."}

Câu hỏi hiện tại:
{question}
""".strip()


def build_grounded_prompt_with_guardrail(
    current_question: str,
    rewritten_query: str,
    docs: List[Any],
    history: List[Dict[str, Any]],
) -> str:
    base_prompt = build_answer_prompt(
        current_question=current_question,
        rewritten_query=rewritten_query,
        docs=docs,
        history=history,
        max_turns=HISTORY_TURNS
    )

    extra_rule = f"""

QUY TẮC BẮT BUỘC:
- Chỉ dùng thông tin từ phần tài liệu được cung cấp.
- Nếu tài liệu không đủ để trả lời chắc chắn, hãy chỉ in đúng duy nhất chuỗi sau:
{INSUFFICIENT_SENTINEL}
- Không giải thích thêm khi in chuỗi đó.
"""

    return base_prompt + "\n\n" + extra_rule


def print_top_files(top_files: List[Dict[str, Any]]) -> None:
    if not top_files:
        print("[Top Files]: Không có file liên quan.\n")
        return

    print("[Top Files]:")
    for i, item in enumerate(top_files, 1):
        print(
            f"  {i}. {item['source_file']} | "
            f"best_score={item['best_score']:.4f} | "
            f"hits={item['hits']}"
        )
    print()


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
            # 1) Turn đầu tiên: không rewrite
            if not history:
                rewritten_query = question
                used_rewrite = False
            else:
                rewritten_query = rewrite_query(
                    current_question=question,
                    history=history
                )
                used_rewrite = (rewritten_query.strip() != question.strip())

            if SHOW_REWRITTEN_QUERY:
                print(f"\n[Rewritten Query]: {rewritten_query}")
                print(f"[Used Rewrite]: {used_rewrite}\n")

            # 2) Retrieve docs + score
            docs = retrieve_documents(
                query=rewritten_query,
                vectorstore=vectorstore,
                top_k=TOP_K
            )

            # 3) Chỉ giữ active docs
            docs = filter_active_docs(docs, top_k=TOP_K)

            # 4) Build top-k files
            top_files = build_top_files(docs, top_k_files=3)
            print_top_files(top_files)

            # 5) Nếu không có docs -> fallback ngay
            if not docs:
                fallback_prompt = build_fallback_prompt(
                    question=question,
                    history=history
                )
                fallback_answer = generate_answer(fallback_prompt)

                warning = (
                    "⚠️ Cảnh báo: Hệ thống không tìm thấy tài liệu liên quan trong kho tri thức. "
                    "Câu trả lời dưới đây được sinh bởi LLM, không dựa trên tài liệu nội bộ."
                )

                print(f"{warning}\n")
                print(f"Bot: {fallback_answer}\n")

                history.append({"role": "user", "content": question})
                history.append({"role": "assistant", "content": fallback_answer})
                continue

            # 6) Thử trả lời grounded
            grounded_prompt = build_grounded_prompt_with_guardrail(
                current_question=question,
                rewritten_query=rewritten_query,
                docs=docs,
                history=history
            )

            grounded_answer = generate_answer(grounded_prompt).strip()

            # 7) Nếu tài liệu không đủ -> fallback có warning
            if grounded_answer == INSUFFICIENT_SENTINEL:
                fallback_prompt = build_fallback_prompt(
                    question=question,
                    history=history
                )
                fallback_answer = generate_answer(fallback_prompt)

                warning = (
                    "⚠️ Cảnh báo: Có tài liệu được truy xuất nhưng không đủ thông tin để trả lời chắc chắn. "
                    "Câu trả lời dưới đây được LLM suy luận thêm và có thể không bám hoàn toàn vào tài liệu trong hệ thống."
                )

                print(f"{warning}\n")
                print(f"Bot: {fallback_answer}\n")

                history.append({"role": "user", "content": question})
                history.append({"role": "assistant", "content": fallback_answer})
            else:
                print("[Mode]: grounded\n")
                print(f"Bot: {grounded_answer}\n")

                history.append({"role": "user", "content": question})
                history.append({"role": "assistant", "content": grounded_answer})

        except Exception as e:
            print(f"\n[ERROR] {e}\n")


if __name__ == "__main__":
    main()