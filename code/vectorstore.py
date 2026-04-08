import shutil
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

def build_embeddings(model_name):
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

def build_faiss(chunks, index_dir, model_name):
    embeddings = build_embeddings(model_name)
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(str(index_dir))
    return db

def load_faiss(index_dir, model_name):
    embeddings = build_embeddings(model_name)
    return FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True
    )