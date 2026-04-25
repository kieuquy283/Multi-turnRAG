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
Run from the project root `E:\Multi-turnRAG`.

### Build from JSON dataset
```bash
python -m scripts.build_index --corpus-json data/DataSetRAG.json --mode from_json --index-dir faiss_index
```

### Build from raw PDFs
```bash
python -m scripts.build_index --mode documents --data-dir data/PDF --index-dir faiss_index
```

This step will:
- Load source documents
- Extract text (PDF/DOCX/TXT)
- Chunk theo Điều luật hoặc fallback
- Tạo metadata (chunk_id, source_file, ...)
- Generate embeddings (local)
- Save FAISS index vào `faiss_index`

---
## 7. Run the chatbot

### Terminal CLI
```bash
python -m scripts.chat_cli --index-dir faiss_index
```

### Web UI
```bash
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000 --reload
```
In another terminal:
```bash
cd chatRAG
npm install
npm run dev
```
Open browser at `http://localhost:5173`.

---
## 8. Evaluation
Run evaluation from the project root.

### 8.1 Single-turn Retrieval
```bash
python -m scripts.evaluate_retrieval --eval-path data/evaluation.json --index-dir faiss_index --top-k 10
```

### 8.2 Multi-turn Retrieval (No rewrite)
```bash
python -m scripts.evaluate_multiturn_retrieval --eval-path data/multiturn_evaluation_filled.json --index-dir faiss_index --top-k 10
```

### 8.3 Multi-turn Retrieval (Rewrite)
```bash
python -m scripts.evaluate_multiturn_retrieval --eval-path data/multiturn_evaluation_filled.json --index-dir faiss_index --top-k 10 --use-rewrite
```

> Note: `7.2` vẫn dùng query gốc, nên đôi khi kết quả giống `7.1` nếu không có rewrite.

Evaluation Metrics
| Metric   | Ý nghĩa                             |
| -------- | ----------------------------------- |
| Hit@k    | Có tìm được tài liệu đúng không     |
| Precision@k | Trong kết quả trả về, bao nhiêu phần trăm là đúng |
| Recall@k | Tìm được bao nhiêu tài liệu đúng    |
| F1       | Trung bình điều hòa của Precision và Recall |
| MRR      | Tài liệu đúng đứng vị trí bao nhiêu |

> F1 = 2 * Precision * Recall / (Precision + Recall)

---
## 9. Pipeline Overview
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
