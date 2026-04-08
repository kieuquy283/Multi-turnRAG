from code.config import *
from code.loader import load_documents
from code.chunker import build_legal_chunks
from code.vectorstore import build_faiss

print("Loading PDFs...")
docs = load_documents()

print("Chunking...")
chunks = build_legal_chunks(docs)

print("Building FAISS...")
build_faiss(chunks, INDEX_DIR, EMBEDDING_MODEL)

print("DONE")