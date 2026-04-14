# 🧠 Multi-turn RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot system designed for conversational AI with multi-turn capability.

## 🚀 Features

- Load PDF documents (e.g., legal documents)
- Chunk text into manageable segments
- Build FAISS vector database
- Retrieve relevant context
- Generate answers using local LLM qua vLLM (Qwen)
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
### 2. Create conda environment
```bash
conda create -n vllm_env python=3.12 -y
conda activate vllm_env
```
---

---
### 3. Install dependencies

```bash
pip install vllm langchain-openai langchain-huggingface langchain-community sentence-transformers faiss-cpu
pip install -r requirements.txt
``` 

---
### 4. Setup environment variables

Create a .env file:
```bash
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_API_KEY=none
CHAT_MODEL=qwen-rag
REWRITE_MODEL=qwen-rag
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
TOP_K=5
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
HISTORY_TURNS=3
SHOW_REWRITTEN_QUERY=true
```
---

### 4.1 Start local vLLM server

```bash
VLLM_TARGET_DEVICE=cuda python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct-AWQ \
    --gpu-memory-utilization 0.7 \
    --max-model-len 4096 \
    --max-num-seqs 16 \
    --enforce-eager \
    --served-model-name qwen-rag \
    --host 0.0.0.0 \
    --port 8000 
```

Lưu ý: `--served-model-name` phải khớp với `CHAT_MODEL` và `REWRITE_MODEL` trong `.env`.

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
- Loads PDF documents
- Splits into chunks
- Embeds text
- Stores vectors in FAISS

Run Chatbot

```bash
python -m code.chat
```
---

Then start asking questions in terminal.

## 7. Web UI Chat Interface

Bạn có thể chạy giao diện web chat với session lưu trữ.

```bash
python -m code.chat_server
```

Mở trình duyệt và truy cập:

```text
http://localhost:8500
```

Tính năng:
- Giao diện web giống desktop, có sidebar session bên trái
- Tạo phiên mới, xóa phiên, chuyển đổi phiên nhanh
- Lưu lịch sử chat theo từng session trong `data/chat_sessions.json`
- Nếu chưa chọn session và gửi câu hỏi, hệ thống tự tạo session mới
- Sidebar cố định kích thước item, nhiều session sẽ scroll được

---

## 8. Desktop App (Tkinter)

Nếu bạn muốn giao diện desktop, chạy:

```bash
python -m code.chat_desktop
```

Tính năng desktop:
- Danh sách phiên bên trái
- Lịch sử chat hiển thị bên phải
- Tạo phiên mới và gửi câu hỏi ngay trong ứng dụng
- Xóa session từng phiên ngay trong app
- Lưu session vào `data/chat_sessions.json`

---

## 9. Pipeline Overview
```text
User Query
   ↓
Embedding
   ↓
FAISS Retrieval
   ↓
Top-k Documents
   ↓
Prompt Construction
   ↓
LLM (Qwen/vLLM)
   ↓
Final Answer
```

---
## 10. CPU-only Mode (llama.cpp server)

Nếu không có GPU CUDA, có thể thay vLLM bằng `llama-cpp-python` server (OpenAI-compatible).

### Bước 1: Cài môi trường CPU + llama.cpp

```bash
# 1. Tạo môi trường mới cho CPU
conda create -n cpu_llm python=3.10 -y
conda activate cpu_llm

# 2. Cài llama.cpp server
pip install llama-cpp-python[server]

# 3. Cài thư viện RAG (nếu chưa có)
pip install langchain-openai langchain-huggingface langchain-community sentence-transformers faiss-cpu
pip install -r requirements.txt
```

### Bước 2: Tải model GGUF (bắt buộc cho CPU)

```bash
# Cài công cụ tải model từ Hugging Face
pip install huggingface_hub

# Tải model Qwen 2.5 3B GGUF (Q4_K_M)
huggingface-cli download Qwen/Qwen2.5-3B-Instruct-GGUF qwen2.5-3b-instruct-q4_k_m.gguf --local-dir . --local-dir-use-symlinks False
```

### Bước 3: Khởi chạy llama.cpp server (Terminal 1)

```bash
python3 -m llama_cpp.server \
   --model qwen2.5-3b-instruct-q4_k_m.gguf \
   --host 0.0.0.0 \
   --port 8000 \
   --n_ctx 4096\
   --n_threads 4 \
   --model_alias qwen-rag
```

Gợi ý:
- `--n_threads 4`: số lõi CPU dành cho model (chỉnh theo máy).
- `--n_ctx 2048`: độ dài ngữ cảnh.
- Giữ `CHAT_MODEL=qwen-rag` và `REWRITE_MODEL=qwen-rag` trong `.env` để khớp với `--model_alias qwen-rag`.