from __future__ import annotations

import argparse
from typing import Any, Dict, List

from rag.retrieval.query_rewriter import rewrite_query
from rag.retrieval.ranking import filter_active_docs
from rag.retrieval.retriever import extract_cids_from_docs, retrieve_documents
from rag.retrieval.vectorstore import load_vectorstore
from rag.utils.io import load_json


def compute_metrics(retrieved_cids: List[Any], gt_cids: List[Any]):
    gt_set = set(gt_cids)

    hit = int(any(cid in gt_set for cid in retrieved_cids))

    retrieved_set = set(retrieved_cids)
    intersection = retrieved_set & gt_set

    precision = 0.0
    if retrieved_cids:
        precision = len(intersection) / len(retrieved_cids)

    recall = 0.0
    if gt_set:
        recall = len(intersection) / len(gt_set)

    f1 = 0.0
    if precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)

    mrr = 0.0
    for rank, cid in enumerate(retrieved_cids, start=1):
        if cid in gt_set:
            mrr = 1.0 / rank
            break

    return hit, precision, recall, f1, mrr


def evaluate_multiturn(
    eval_path: str = "data/multiturn_evaluation_filled.json",
    index_dir: str = "indexes/default",
    top_k: int = 5,
    use_rewrite: bool = True,
) -> None:
    data = load_json(eval_path, [])
    vectorstore = load_vectorstore(index_dir)

    total = 0
    total_hit = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_mrr = 0.0

    for i, sample in enumerate(data):
        question = sample["question"]
        history = sample.get("history", [])
        gt_cids = sample.get("ground_truth_cids", [])

        if not gt_cids:
            continue

        query = rewrite_query(question, history) if use_rewrite else question

        docs = retrieve_documents(query=query, vectorstore=vectorstore, top_k=top_k)
        docs = filter_active_docs(docs, top_k=top_k)
        retrieved_cids = extract_cids_from_docs(docs)

        hit, precision, recall, f1, mrr = compute_metrics(retrieved_cids, gt_cids)

        total += 1
        total_hit += hit
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        total_mrr += mrr

        print("=" * 60)
        print(f"Sample {i+1}")
        print("Original :", question)
        print("Rewritten:", query)
        print("GT CIDs  :", gt_cids)
        print("Retrieved:", retrieved_cids)
        print(f"Hit={hit}, Precision={precision:.2f}, Recall={recall:.2f}, F1={f1:.2f}, MRR={mrr:.2f}")

    if total == 0:
        print("Không có sample nào có ground_truth_cids để đánh giá.")
        return

    print("\n===== FINAL METRICS =====")
    mode_name = "WITH_REWRITE" if use_rewrite else "NO_REWRITE"
    print(f"Mode: {mode_name}")
    print(f"Samples: {total}")
    print(f"Hit@{top_k}: {total_hit / total:.4f}")
    print(f"Precision@{top_k}: {total_precision / total:.4f}")
    print(f"Recall@{top_k}: {total_recall / total:.4f}")
    print(f"F1: {total_f1 / total:.4f}")
    print(f"MRR: {total_mrr / total:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-path", default="data/multiturn_evaluation_filled.json")
    parser.add_argument("--index-dir", default="indexes/default")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--use-rewrite", action="store_true")
    args = parser.parse_args()

    evaluate_multiturn(
        eval_path=args.eval_path,
        index_dir=args.index_dir,
        top_k=args.top_k,
        use_rewrite=args.use_rewrite,
    )
