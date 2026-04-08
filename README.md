# 🧠 Multi-turn RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot system designed for conversational AI with multi-turn capability.

## 🚀 Features

- Load PDF documents (e.g., legal documents)
- Chunk text into manageable segments
- Build FAISS vector database
- Retrieve relevant context
- Generate answers using LLM (OpenAI GPT)
- Modular pipeline (loader, chunker, retriever, llm, etc.)

---

## 📁 Project Structure
rag_chatbot/
│
├── code/
│ ├── init.py
│ ├── build_index.py # Build FAISS index from documents
│ ├── chat.py # Run chatbot
│ ├── loader.py # Load PDF files
│ ├── chunker.py # Split documents into chunks
│ ├── vectorstore.py # FAISS index (build/load)
│ ├── retriever.py # Retrieve relevant documents
│ ├── llm.py # LLM interaction (OpenAI)
│ ├── formatter.py # Prompt construction
│ └── config.py # Configuration
│
├── data/ # (ignored) Input PDF files
├── faiss_index/ # (ignored) Vector database
├── .env # API key (not committed)
├── .gitignore
├── requirements.txt
└── README.md

---

## ⚙️ Setup

### 1. Clone repository

```bash
git clone https://github.com/your_username/Multi-turnRAG.git
cd Multi-turnRAG

2. Create virtual environment
python -m venv rag_env
Activate:

Windows:

rag_env\Scripts\activate

Mac/Linux:

source rag_env/bin/activate
3. Install dependencies
pip install -r requirements.txt
4. Setup environment variables

Create a .env file in root directory:

OPENAI_API_KEY=your_api_key_here
📂 Add Data

Create a data/ folder and add your PDF files:

data/
 ├── document1.pdf
 ├── document2.pdf
🏗️ Build Vector Database
python -m code.build_index

This step:

Loads PDF documents
Splits into chunks
Embeds text
Stores vectors in FAISS
💬 Run Chatbot
python -m code.chat

Then start asking questions in terminal.

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
