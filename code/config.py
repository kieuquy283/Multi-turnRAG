from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

ENV_PATH = BASE_DIR.parent / ".env"
load_dotenv(ENV_PATH)

DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "faiss_index"

EMBEDDING_MODEL = "intfloat/multilingual-e5-base"
TOP_K = 5
MAX_CONTEXT_CHUNKS = 5