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

### 1. Clone repository

```bash
git clone https://github.com/your_username/Multi-turnRAG.git
cd Multi-turnRAG
```

### 2. Create virtual environment
```bash
python -m venv rag_env
```

### Activate:

### Windows:

```bash
rag_env\Scripts\activate
```
### Mac/Linux:

```bash
source rag_env/bin/activate
``` 
### 3. Install dependencies

```bash
pip install -r requirements.txt
``` 

### 4. Setup environment variables

### Create a .env file in root directory:

OPENAI_API_KEY=your_api_key_here
📂 Add Data

Create a data/ folder and add your PDF files:

data/
 ├── document1.pdf
 ├── document2.pdf
🏗️ Build Vector Database
```bash
python -m code.build_index
``` 

### This step:

Loads PDF documents
Splits into chunks
Embeds text
Stores vectors in FAISS
💬 Run Chatbot
```bash
python -m code.chat
```

### Then start asking questions in terminal.

🧠 Pipeline Overview
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
LLM (GPT)
   ↓
Final Answer
