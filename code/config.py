from pathlib import Path
import os
from dotenv import load_dotenv

# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = BASE_DIR / "code"
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "faiss_index"
ENV_PATH = BASE_DIR / ".env"

# Load environment variables
load_dotenv(ENV_PATH)

# =========================
# API / Models
# =========================
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1").strip()
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "none").strip() or "none"

CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen-rag")
REWRITE_MODEL = os.getenv("REWRITE_MODEL", CHAT_MODEL)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# =========================
# RAG Settings
# =========================
TOP_K = int(os.getenv("TOP_K", 7))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", 3))

# Nếu cần lọc file PDF theo đuôi
SUPPORTED_EXTENSIONS = [".pdf"]

# Có in rewritten query ra màn hình không
SHOW_REWRITTEN_QUERY = os.getenv("SHOW_REWRITTEN_QUERY", "true").lower() == "true"