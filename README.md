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
<<<<<<< HEAD
=======
INDEX_DIR=faiss_index
>>>>>>> afc4b2326c014776df660479b984efbfe0a067c5
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
<<<<<<< HEAD
Build lần đầu
```bash
python -m scripts.build_index --data-dir data --index-dir faiss_index
```
Update (khi thêm/sửa file)
```bash
python -m scripts.update_index --data-dir data --index-dir faiss_index
=======
Build (nếu chưa embedding, ở đây là đã embedding và lưu ở faiss_index)
```bash
python -m scripts.build_index --corpus-json data/retrieval_corpus.json --mode from_json --index-dir faiss_index
>>>>>>> afc4b2326c014776df660479b984efbfe0a067c5
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
<<<<<<< HEAD
--- 

Then start asking questions in terminal.

### 7.Pipeline Overview
=======

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
Evaluation Metrics
| Metric   | Ý nghĩa                             |
| -------- | ----------------------------------- |
| Hit@k    | Có tìm được tài liệu đúng không     |
| Recall@k | Tìm được bao nhiêu tài liệu đúng    |
| MRR      | Tài liệu đúng đứng vị trí bao nhiêu |

### 8.Pipeline Overview
>>>>>>> afc4b2326c014776df660479b984efbfe0a067c5
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