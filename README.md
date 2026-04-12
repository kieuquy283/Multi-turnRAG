# 🧠 Multi-turn RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot system designed for conversational AI with multi-turn capability.

## 🚀 Features

- Load PDF documents (e.g., legal documents)
- Chunk text into manageable segments
- Build FAISS vector database
- Retrieve relevant context
- Generate answers using LLM (OpenAI GPT)
- Modular pipeline (loader, chunker, retriever, llm, etc.)


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
Truy cập link: [get dashscope_api_key](https://modelstudio.console.alibabacloud.com/) để tạo tk -> thêm phương thức thanh toán -> dùng trial
Create a .env file:
```bash
DASHSCOPE_API_KEY=your_api_key_here
QWEN_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
CHAT_MODEL=qwen-plus
REWRITE_MODEL=qwen-plus
EMBEDDING_BACKEND=local
LOCAL_EMBEDDING_MODEL=intfloat/multilingual-e5-base
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
Build lần đầu
```bash
python -m code.build_index build --data-dir data --index-dir faiss_index
```
Update (khi thêm/sửa file)
```bash
python -m code.build_index update --data-dir data --index-dir faiss_index
```

This step:
- Loads documents
- Extract text (PDF/DOCX)
- Chunk theo Điều luật hoặc fallback
- Generate embeddings (local)
- Stores vectors in FAISS

Run Chatbot

```bash
python -m code.chat
```
--- 

Then start asking questions in terminal.

### 7Pipeline Overview
```User Query
   ↓
Embedding
   ↓
FAISS Retrieval
   ↓
Top-k Documents
   ↓
Prompt Construction
   ↓
LLM (GPT)
   ↓
Final Answer
```