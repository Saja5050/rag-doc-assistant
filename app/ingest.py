import os, glob
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

DATA_DIR = os.environ.get("DATA_DIR", "data")
INDEX_DIR = os.environ.get("INDEX_DIR", "vectorstore")
EMBEDDINGS_MODEL = os.environ.get("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def load_documents(data_dir: str):
    docs = []
    for path in glob.glob(os.path.join(data_dir, "**", "*"), recursive=True):
        if os.path.isdir(path):
            continue
        ext = Path(path).suffix.lower()
        if ext in [".txt", ".md"]:
            docs.extend(TextLoader(path, encoding="utf-8").load())
        elif ext == ".pdf":
            docs.extend(PyPDFLoader(path).load())
    if not docs:
        raise RuntimeError(f"No documents found in '{data_dir}'. Add .txt/.md/.pdf files and retry.")
    return docs

def build_index(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    vectordb = FAISS.from_documents(chunks, embeddings)
    os.makedirs(INDEX_DIR, exist_ok=True)
    vectordb.save_local(INDEX_DIR)
    return len(chunks)

if __name__ == "__main__":
    print("Loading documents...")
    docs = load_documents(DATA_DIR)
    print(f"Loaded {len(docs)} documents.")
    print("Building index...")
    n_chunks = build_index(docs)
    print(f"Done. Created {n_chunks} chunks. Index saved to '{INDEX_DIR}'.")
