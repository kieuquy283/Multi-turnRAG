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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

if not OPENAI_API_KEY:
    raise ValueError(
        "Không tìm thấy OPENAI_API_KEY. "
        "Hãy tạo file .env ở thư mục gốc project và thêm:\n"
        "OPENAI_API_KEY=your_api_key_here"
    )

CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
REWRITE_MODEL = os.getenv("REWRITE_MODEL", CHAT_MODEL)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# =========================
# RAG Settings
# =========================
TOP_K = int(os.getenv("TOP_K", 5))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
HISTORY_TURNS = int(os.getenv("HISTORY_TURNS", 3))

# Nếu cần lọc file PDF theo đuôi
SUPPORTED_EXTENSIONS = [".pdf"]

# Có in rewritten query ra màn hình không
SHOW_REWRITTEN_QUERY = os.getenv("SHOW_REWRITTEN_QUERY", "true").lower() == "true"