from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

from code.chat import (
    INSUFFICIENT_SENTINEL,
    build_fallback_prompt,
    build_grounded_prompt_with_guardrail,
)
from code.config import HISTORY_TURNS, SHOW_REWRITTEN_QUERY, TOP_K
from code.llm import generate_answer
from code.query_rewriter import rewrite_query
from code.retriever import retrieve_documents, filter_active_docs, build_top_files
from code.vectorstore import load_vectorstore


def run_chat(
    question: str,
    history: List[Dict[str, Any]],
    vectorstore=None,
) -> Dict[str, Any]:
    if vectorstore is None:
        vectorstore = load_vectorstore()

    if not history:
        rewritten_query = question
        used_rewrite = False
    else:
        rewritten_query = rewrite_query(current_question=question, history=history)
        used_rewrite = rewritten_query.strip() != question.strip()

    docs = retrieve_documents(query=rewritten_query, vectorstore=vectorstore, top_k=TOP_K)
    docs = filter_active_docs(docs, top_k=TOP_K)
    top_files = build_top_files(docs, top_k_files=3)

    if not docs:
        fallback_prompt = build_fallback_prompt(question=question, history=history)
        fallback_answer = generate_answer(fallback_prompt)
        return {
            "answer": fallback_answer,
            "mode": "fallback",
            "top_files": top_files,
            "rewritten_query": rewritten_query,
            "used_rewrite": used_rewrite,
            "warning": "No documents found in knowledge base.",
        }

    grounded_prompt = build_grounded_prompt_with_guardrail(
        current_question=question,
        rewritten_query=rewritten_query,
        docs=docs,
        history=history,
    )
    grounded_answer = generate_answer(grounded_prompt).strip()

    if grounded_answer == INSUFFICIENT_SENTINEL:
        fallback_prompt = build_fallback_prompt(question=question, history=history)
        fallback_answer = generate_answer(fallback_prompt)
        return {
            "answer": fallback_answer,
            "mode": "fallback",
            "top_files": top_files,
            "rewritten_query": rewritten_query,
            "used_rewrite": used_rewrite,
            "warning": "Documents were found but not enough context to answer confidently.",
        }

    return {
        "answer": grounded_answer,
        "mode": "grounded",
        "top_files": top_files,
        "rewritten_query": rewritten_query,
        "used_rewrite": used_rewrite,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat API wrapper for Multi-turn RAG.")
    parser.add_argument("--question", required=True, help="User question.")
    parser.add_argument("--history-json", default="[]", help="Conversation history as JSON string.")
    args = parser.parse_args()

    try:
        history = json.loads(args.history_json)
    except json.JSONDecodeError:
        history = []

    try:
        response = run_chat(question=args.question, history=history)
        print(json.dumps(response, ensure_ascii=False))
    except Exception as exc:
        error_payload = {"error": str(exc)}
        print(json.dumps(error_payload, ensure_ascii=False))
        sys.exit(1)


if __name__ == "__main__":
    main()
