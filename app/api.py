from __future__ import annotations

from pathlib import Path
from typing import Any, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag.config.retrieval import DEFAULT_INDEX_DIR
from rag.ingestion.indexing import index_exists
from rag.pipelines.chat_pipeline import ChatPipeline
from rag.retrieval.query_rewriter import rewrite_query
from rag.retrieval.vectorstore import load_vectorstore
from rag.retrieval.retriever import extract_cids_from_docs, retrieve_documents
from rag.retrieval.ranking import filter_active_docs
from rag.utils.io import load_json, save_json


class ChatMessage(BaseModel):
    role: str
    content: str
    time: str | None = None
    metadata: dict[str, Any] | None = None


class ChatSession(BaseModel):
    id: str
    name: str
    createdAt: str
    messages: list[ChatMessage]


class ChatRequest(BaseModel):
    question: str
    history: list[dict] = []


class ChatResponse(BaseModel):
    answer: str
    rewritten_query: str
    used_rewrite: bool
    show_rewritten_query: bool
    grounded: bool
    warning: str
    mode: str
    top_files: list[dict]
    history: list[dict]


class EvaluationStats(BaseModel):
    name: str
    top_k: int
    eval_path: str
    sample_count: int
    hit: float
    precision: float
    recall: float
    f1: float
    mrr: float


class EvaluationResponse(BaseModel):
    results: List[EvaluationStats]


app = FastAPI(title="Multi-turn RAG Chat API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

default_index_dir = Path(DEFAULT_INDEX_DIR)
faiss_index_dir = Path("faiss_index")
selected_index_dir = default_index_dir

if not index_exists(default_index_dir) and index_exists(faiss_index_dir):
    selected_index_dir = faiss_index_dir
elif not index_exists(default_index_dir) and not index_exists(faiss_index_dir):
    selected_index_dir = default_index_dir

pipeline = ChatPipeline(index_dir=str(selected_index_dir))

ROOT_DIR = Path(__file__).resolve().parents[1]
SESSION_FILE_PATH = ROOT_DIR / "data" / "chat_sessions.json"


def load_chat_sessions() -> list[Any]:
    sessions = load_json(SESSION_FILE_PATH, [])
    return sessions if isinstance(sessions, list) else []


def save_chat_sessions(sessions: list[Any]) -> None:
    save_json(SESSION_FILE_PATH, sessions)


@app.get("/sessions", response_model=list[ChatSession])
def get_sessions() -> list[ChatSession]:
    return load_chat_sessions()


@app.post("/sessions", response_model=list[ChatSession])
def save_sessions(sessions: list[ChatSession]) -> list[ChatSession]:
    save_chat_sessions([session.dict() for session in sessions])
    return sessions


def compute_metrics(retrieved_cids: list[Any], gt_cids: list[Any]) -> tuple[float, float, float, float, float]:
    gt_set = set(gt_cids)
    retrieved_set = set(retrieved_cids)
    intersection = retrieved_set & gt_set

    hit = float(int(any(cid in gt_set for cid in retrieved_cids)))
    precision = float(len(intersection) / len(retrieved_cids)) if retrieved_cids else 0.0
    recall = float(len(intersection) / len(gt_set)) if gt_set else 0.0
    f1 = 0.0
    if precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)

    mrr = 0.0
    for rank, cid in enumerate(retrieved_cids, start=1):
        if cid in gt_set:
            mrr = 1.0 / rank
            break
    return hit, precision, recall, f1, mrr


def evaluate_single_turn(eval_path: str, index_dir: str, top_k: int = 10) -> EvaluationStats:
    data = load_json(eval_path, [])
    if not isinstance(data, list) or not data:
        raise ValueError(f"Evaluation data is missing or invalid: {eval_path}")

    vectorstore = load_vectorstore(index_dir=index_dir)
    total = 0
    total_hit = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_mrr = 0.0

    for sample in data:
        if not isinstance(sample, dict):
            continue

        question = str(sample.get("question", "")).strip()
        gt_cids = sample.get("ground_truth_cids", [])
        if not question or not gt_cids:
            continue

        docs = retrieve_documents(query=question, vectorstore=vectorstore, top_k=top_k)
        docs = filter_active_docs(docs, top_k=top_k)
        retrieved_cids = extract_cids_from_docs(docs)

        hit, precision, recall, f1, mrr = compute_metrics(retrieved_cids, gt_cids)
        total += 1
        total_hit += hit
        total_precision += precision
        total_recall += recall
        total_f1 += f1
        total_mrr += mrr

    if total == 0:
        raise ValueError(f"No valid evaluation samples found in: {eval_path}")

    return EvaluationStats(
        name="Single-turn Retrieval",
        top_k=top_k,
        eval_path=eval_path,
        sample_count=total,
        hit=total_hit / total,
        precision=total_precision / total,
        recall=total_recall / total,
        f1=total_f1 / total,
        mrr=total_mrr / total,
    )


def evaluate_multiturn(eval_path: str, index_dir: str, top_k: int = 10, use_rewrite: bool = True) -> EvaluationStats:
    data = load_json(eval_path, [])
    if not isinstance(data, list) or not data:
        raise ValueError(f"Evaluation data is missing or invalid: {eval_path}")

    vectorstore = load_vectorstore(index_dir=index_dir)
    total = 0
    total_hit = 0.0
    total_precision = 0.0
    total_recall = 0.0
    total_f1 = 0.0
    total_mrr = 0.0

    for sample in data:
        if not isinstance(sample, dict):
            continue

        question = str(sample.get("question", "")).strip()
        gt_cids = sample.get("ground_truth_cids", [])
        history = sample.get("history", [])
        if not question or not gt_cids:
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

    if total == 0:
        raise ValueError(f"No valid evaluation samples found in: {eval_path}")

    name = "Multi-turn Rewrite" if use_rewrite else "Multi-turn No Rewrite"
    return EvaluationStats(
        name=name,
        top_k=top_k,
        eval_path=eval_path,
        sample_count=total,
        hit=total_hit / total,
        precision=total_precision / total,
        recall=total_recall / total,
        f1=total_f1 / total,
        mrr=total_mrr / total,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> Any:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = pipeline.chat(question=request.question, history=request.history)
    return result


@app.get("/evaluation", response_model=EvaluationResponse)
def get_evaluation() -> EvaluationResponse:
    index_dir = str(selected_index_dir)
    try:
        single = evaluate_single_turn(
            eval_path="data/evaluation.json",
            index_dir=index_dir,
            top_k=10,
        )
        multiturn_no_rewrite = evaluate_multiturn(
            eval_path="data/multiturn_evaluation_filled.json",
            index_dir=index_dir,
            top_k=10,
            use_rewrite=False,
        )
        multiturn_rewrite = evaluate_multiturn(
            eval_path="data/multiturn_evaluation_filled.json",
            index_dir=index_dir,
            top_k=10,
            use_rewrite=True,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return EvaluationResponse(results=[single, multiturn_no_rewrite, multiturn_rewrite])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.api:app", host="127.0.0.1", port=8000, reload=True)
