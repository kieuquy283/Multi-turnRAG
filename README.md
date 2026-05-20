# 🧠 Multi-turn RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot system designed for conversational AI with multi-turn capability.

---
## Features
## Retrieval
- FAISS vector database
- Local embeddings (multilingual-e5-base)
- Legal-aware chunking (split theo Điều luật)
- Fallback chunking nếu không detect được cấu trúc
## Multi-turn Reasoning
- Query rewriting (context-aware)
- Heuristic + LLM rewrite (an toàn, không phá query)
- History-aware retrieval
## Answer Generation
- Grounded answer (dựa trên tài liệu)
- Fallback answer (LLM nếu thiếu context)
- Warning khi không đủ dữ liệu
## Research / Evaluation
- Build retrieval corpus từ dataset
- Evaluate single-turn retrieval
- Evaluate multi-turn retrieval (with rewrite)
---

## ⚙️ Setup

---
### 1. Clone repository

```bash
git clone https://github.com/your_username/Multi-turnRAG.git
cd Multi-turnRAG
```
---

---
### 2. Create virtual environment
```bash
python -m venv rag_env
```
---

---
### Activate:

```bash
rag_env\Scripts\activate
```
### Mac/Linux:

```bash
source rag_env/bin/activate
``` 
---

### 3. Install dependencies

```bash
pip install -r requirements.txt
``` 

---
### 4. Setup environment variables
```text
Truy cập link: [get dashscope_api_key](https://modelstudio.console.alibabacloud.com/) để tạo tk -> thêm phương thức thanh toán -> dùng trial
```
Create a .env file:
```bash
DASHSCOPE_API_KEY=your_api_key_here
QWEN_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
CHAT_MODEL=qwen-plus
REWRITE_MODEL=qwen-plus
EMBEDDING_BACKEND=local
LOCAL_EMBEDDING_MODEL=intfloat/multilingual-e5-base
INDEX_DIR=faiss_index
TOP_K=5
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
HISTORY_TURNS=3
SHOW_REWRITTEN_QUERY=true
```
---

---
## 5. Add Data

Create a data/ folder and add your PDF files:

```text
data/
 ├── document1.pdf
 ├── document2.pdf
```
---

---
## 6. Build vector database
Build (nếu chưa embedding, ở đây là đã embedding và lưu ở faiss_index)
```bash
python -m scripts.build_index --corpus-json data/retrieval_corpus.json --mode from_json --index-dir faiss_index
```

This step:
- Load documents (PDF/DOCX/TXT)
- Extract text (fallback nhiều tầng)
- Chunk theo Điều luật
- Tạo metadata (hash, chunk_id, source_file)
- Generate embeddings (local)
- Build FAISS index

Run Chatbot

```bash
python -m scripts.chat_cli --index-dir faiss_index
```

Run the modular chat pipeline directly:

```bash
python -m scripts.chat_cli_modular --index-dir faiss_index
```

Or run the new web chat UI:

```bash
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000 --reload
cd chatRAG
npm install
npm run dev
```

Open the browser at `http://localhost:5173` and chat through the web interface.

Use the left-side menu to switch between `Chat` và `Evaluation`. The Evaluation tab tổng hợp kết quả từ:
- Single-turn Retrieval
- Multi-turn (No rewrite)
- Multi-turn (Rewrite)

--- 

Then start asking questions in terminal or in the web UI.

---
## Architecture

The repository currently has two layers:

- Legacy linear runtime flow used by the app and CLI
- Modular research architecture used for ablation and future pipeline composition

Modular flow:

User Question  
→ History Selection  
→ Query Rewriting  
→ Retrieval  
→ Reranking  
→ Context Selection  
→ Answer Generation

Ablation models may use only a subset of these modules.

## Ablation Models

- Model 1: Baseline Dense FAISS
- Model 2: Query Rewriting + Dense FAISS
- Model 3: Hybrid Retrieval
- Model 4: Retrieval + Reranking
- Model 5: Hybrid History Selection
- Model 6: Multi-query Hybrid Retrieval
- Model 7: Full Best Modular Pipeline
- Model 8: HyDE, if implemented

## Legacy Compatibility

Some older files under `rag/retrieval/` and `rag/pipelines/` remain in the repository as compatibility layers for the current app, CLI, and older evaluation scripts.

New research development should prefer:

- `rag/modules/history_selection/`
- `rag/modules/query_rewriting/`
- `rag/modules/retrieval/`
- `rag/modules/reranking/`

Legacy scripts such as `scripts/evaluate_retrieval.py` and `scripts/evaluate_multiturn_retrieval.py` are still available, but model-specific ablation scripts are the recommended path for research experiments.

For the FastAPI app, you can switch the runtime chat pipeline with:

```bash
RAG_CHAT_PIPELINE=legacy
RAG_CHAT_PIPELINE=modular
```

If unset, the app keeps using the legacy pipeline by default.

---
## 7. Evaluation
7.1 Single-turn Retrieval
```bash
python -m scripts.evaluate_retrieval --eval-path data/evaluation.json --index-dir faiss_index --top-k 10
```
7.2. Multi-turn (No rewrite)
```bash
python -m scripts.evaluate_multiturn_retrieval --eval-path data/multiturn_evaluation_filled.json --index-dir faiss_index --top-k 10
```

7.3. Multi-turn (rewrite)
```bash
python -m scripts.evaluate_multiturn_retrieval --eval-path data/multiturn_evaluation_filled.json --index-dir faiss_index --top-k 10 --use-rewrite
```
### Model 1: Baseline Dense FAISS
Mô tả:
Model 1 giữ nguyên câu hỏi hiện tại làm truy vấn, không sử dụng lịch sử hội thoại, không rewrite, không hybrid retrieval và không reranking. Mô hình dùng FAISS dense retrieval để lấy top-k chunks và đánh giá bằng Hit@k, Recall@k, MRR.

```bash
python -m scripts.evaluate_model_1_baseline \
  --eval-path data/multiturn_evaluation_filled.json \
  --index-dir indexes/default \
  --top-k 10 \
  --output-path logs/eval_runs/model_1_baseline.json
```

Model 1 là baseline tuyệt đối cho các model sau, nên không thêm bất kỳ tối ưu nào ngoài dense FAISS retrieval hiện có.

### Model 2: Query Rewriting + Dense FAISS

Description:
Model 2 uses the conversation history to rewrite the current question into a standalone query. The rewritten query is then passed to the same FAISS dense retriever used in Model 1. This model does not use hybrid retrieval, multi-query expansion, reranking, or answer generation.

```bash
python -m scripts.evaluate_model_2_rewrite_dense \
  --eval-path data/multiturn_evaluation_filled.json \
  --index-dir indexes/default \
  --top-k 10 \
  --output-path logs/eval_runs/model_2_rewrite_dense.json
```

Important:
Model 2 must differ from Model 1 only by the query rewriting step. Keep all other components the same as Model 1 so the ablation result is meaningful.

### Model 3: Hybrid Retrieval

Model 3 keeps the original question unchanged and replaces dense-only FAISS retrieval with hybrid retrieval combining dense and sparse signals. It does not use query rewriting, history selection, multi-query, reranking, or answer generation.

```bash
python -m scripts.evaluate_model_3_hybrid \
  --eval-path data/multiturn_evaluation_legal.json \
  --index-dir indexes/legal \
  --corpus-path data/legal_corpus_chunks.json \
  --top-k 10 \
  --output-path logs/eval_runs/model_3_hybrid_legal.json
```

### Model 4: Hybrid Retrieval + Reranking

Model 4 uses the original question, retrieves candidate chunks using hybrid retrieval, then applies a reranker to reorder candidates before selecting final top-k. It does not use query rewriting, history selection, multi-query, or answer generation.

```bash
python -m scripts.evaluate_model_4_hybrid_rerank \
  --eval-path data/multiturn_evaluation_legal.json \
  --index-dir indexes/legal \
  --corpus-path data/legal_corpus_chunks.json \
  --top-k 10 \
  --candidate-k 30 \
  --output-path logs/eval_runs/model_4_hybrid_rerank_legal.json
```

### Model 5: Hybrid History Selection + Query Rewriting + Hybrid Retrieval

Model 5 uses hybrid history selection to choose useful conversation turns, rewrites the current question using the selected history, then retrieves evidence with hybrid retrieval. It does not use reranking, multi-query, or answer generation.

```bash
python -m scripts.evaluate_model_5_hybrid_history \
  --eval-path data/multiturn_evaluation_legal.json \
  --index-dir indexes/legal \
  --corpus-path data/legal_corpus_chunks.json \
  --top-k 10 \
  --history-top-k 4 \
  --output-path logs/eval_runs/model_5_hybrid_history_legal.json
```

### Model 6: Multi-Query Hybrid Retrieval

Model 6 extends Model 5 by generating multiple query variants from the rewritten query, retrieving evidence with hybrid retrieval for each query, and fusing the results. It does not use reranking or answer generation.

```bash
python -m scripts.evaluate_model_6_multi_query_hybrid \
  --eval-path data/multiturn_evaluation_legal.json \
  --index-dir indexes/legal \
  --corpus-path data/legal_corpus_chunks.json \
  --top-k 10 \
  --history-top-k 4 \
  --num-queries 4 \
  --output-path logs/eval_runs/model_6_multi_query_hybrid_legal.json
```

### Model 7: Full Modular Pipeline

Model 7 combines hybrid history selection, query rewriting, multi-query generation, hybrid retrieval, fusion, and reranking. It represents the full optimized retrieval pipeline and is compared against previous ablation models.

```bash
python -m scripts.evaluate_model_7_full_pipeline \
  --eval-path data/multiturn_evaluation_legal.json \
  --index-dir indexes/legal \
  --corpus-path data/legal_corpus_chunks.json \
  --top-k 10 \
  --candidate-k 40 \
  --history-top-k 4 \
  --num-queries 4 \
  --output-path logs/eval_runs/model_7_full_pipeline_legal.json
```

### Model 8: HyDE + Hybrid Retrieval + Reranking

Model 8 generates a hypothetical legal passage from the rewritten query, uses it as an expanded retrieval query together with the rewritten query, retrieves evidence with hybrid retrieval, fuses candidates, and reranks them before selecting final top-k. It is used to test whether HyDE improves the full modular retrieval pipeline.

```bash
python -m scripts.evaluate_model_8_hyde \
  --eval-path data/multiturn_evaluation_legal.json \
  --index-dir indexes/legal \
  --corpus-path data/legal_corpus_chunks.json \
  --top-k 5 \
  --candidate-k 40 \
  --history-top-k 4 \
  --output-path logs/eval_runs/model_8_hyde_legal_top5.json
```

### Preparing Legal_Dataset_V1

```bash
python -m scripts.prepare_legal_dataset \
  --input-path data/Legal_Dataset_V1.json \
  --corpus-output data/legal_corpus_chunks.json \
  --eval-output data/multiturn_evaluation_legal.json
```

```bash
python -m scripts.build_index \
  --mode from_json \
  --corpus-json data/legal_corpus_chunks.json \
  --index-dir indexes/legal
```

```bash
python -m scripts.evaluate_model_1_baseline \
  --eval-path data/multiturn_evaluation_legal.json \
  --index-dir indexes/legal \
  --top-k 10 \
  --output-path logs/eval_runs/model_1_baseline_legal.json
```

```bash
python -m scripts.evaluate_model_2_rewrite_dense \
  --eval-path data/multiturn_evaluation_legal.json \
  --index-dir indexes/legal \
  --top-k 10 \
  --output-path logs/eval_runs/model_2_rewrite_dense_legal.json
```

Important:
Model 1 and Model 2 must use the same FAISS index and same evaluation file.
The only difference between Model 1 and Model 2 should be:
- Model 1 uses original question directly.
- Model 2 rewrites current_question using conversation history before retrieval.

Evaluation Metrics
| Metric   | Ý nghĩa                             |
| -------- | ----------------------------------- |
| Hit@k    | Có tìm được tài liệu đúng không     |
| Recall@k | Tìm được bao nhiêu tài liệu đúng    |
| MRR      | Tài liệu đúng đứng vị trí bao nhiêu |

### 8.Pipeline Overview
```text
User Question
     ↓
Query Rewriting (multi-turn)
     ↓
Embedding (E5)
     ↓
FAISS Retrieval
     ↓
Filter Active Docs
     ↓
Top-k Documents
     ↓
Prompt Builder
     ↓
LLM (Qwen API)
     ↓
Answer (Grounded / Fallback)
```
