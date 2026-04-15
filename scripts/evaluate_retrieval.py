from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

from rag.retrieval.vectorstore import load_vectorstore
from rag.retrieval.retriever import retrieve_documents, extract_cids_from_docs
from rag.retrieval.ranking import filter_active_docs


def load_evaluation_data(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_metrics(retrieved_cids: List[Any], gt_cids: List[Any]):
    gt_set = set(gt_cids)

    hit = int(any(cid in gt_set for cid in retrieved_cids))

    recall = 0.0
    if gt_set:
        recall = len(set(retrieved_cids) & gt_set) / len(gt_set)

    mrr = 0.0
    for rank, cid in enumerate(retrieved_cids, start=1):
        if cid in gt_set:
            mrr = 1.0 / rank
            break

    return hit, recall, mrr


def evaluate_retrieval(
    eval_path: str = "data/evaluation.json",
    index_dir: str = "indexes/default",
    top_k: int = 10,
):
    print("[INFO] Loading evaluation dataset...")
    data = load_evaluation_data(eval_path)

    print("[INFO] Loading FAISS vectorstore...")
    vectorstore = load_vectorstore(index_dir=index_dir)

    total = len(data)
    if total == 0:
        print("[INFO] Evaluation dataset rỗng.")
        return

    total_hit = 0
    total_recall = 0.0
    total_mrr = 0.0

    print(f"[INFO] Running evaluation on {total} samples...\n")

    for i, sample in enumerate(data):
        question = str(sample.get("question", "")).strip()
        gt_cids = sample.get("ground_truth_cids", [])

        if not question:
            continue

        docs = retrieve_documents(
            query=question,
            vectorstore=vectorstore,
            top_k=top_k,
        )
        docs = filter_active_docs(docs, top_k=top_k)

        retrieved_cids = extract_cids_from_docs(docs)

        hit, recall, mrr = compute_metrics(retrieved_cids, gt_cids)

        total_hit += hit
        total_recall += recall
        total_mrr += mrr

        if i < 5:
            print("=" * 60)
            print(f"Q{i+1}: {question}")
            print("GT CIDs  :", gt_cids)
            print("Retrieved:", retrieved_cids)
            print(f"Hit={hit}, Recall={recall:.2f}, MRR={mrr:.2f}")

    print("\n===== FINAL METRICS =====")
    print(f"Samples: {total}")
    print(f"Hit@{top_k}: {total_hit / total:.4f}")
    print(f"Recall@{top_k}: {total_recall / total:.4f}")
    print(f"MRR: {total_mrr / total:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate single-turn retrieval.")
    parser.add_argument("--eval-path", default="data/evaluation.json")
    parser.add_argument("--index-dir", default="indexes/default")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    evaluate_retrieval(
        eval_path=args.eval_path,
        index_dir=args.index_dir,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()