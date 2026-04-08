from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from code.config import DATA_DIR

def load_documents():
    documents = []

    pdf_files = list(Path(DATA_DIR).glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(f"Không tìm thấy file PDF nào trong: {DATA_DIR}")

    for pdf_file in pdf_files:
        loader = PyPDFLoader(str(pdf_file))
        docs = loader.load()

        for doc in docs:
            doc.metadata["source_file"] = pdf_file.name

        documents.extend(docs)

    return documents