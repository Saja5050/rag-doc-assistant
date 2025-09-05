import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

INDEX_DIR = os.environ.get("INDEX_DIR", "vectorstore")
EMBEDDINGS_MODEL = os.environ.get("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")

app = FastAPI(title="AI Document Assistant (RAG)")

class AskRequest(BaseModel):
    question: str = Field(..., description="Natural language question")
    k: int = Field(4, ge=1, le=10, description="How many chunks to retrieve")

def load_vectordb():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDINGS_MODEL)
    try:
        db = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load vector index at '{INDEX_DIR}'. Did you run 'python app/ingest.py' first? Error: {e}"
        )
    return db

def build_rag_chain():
    retriever = load_vectordb().as_retriever(search_kwargs={"k": 4})
    llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)

    template = (
        "You are a helpful assistant that answers strictly based on the provided context. "
        "If the answer is not in the context, say you don't know.\n\n"
        "Question: {question}\n\n"
        "Context:\n{context}\n\n"
        "Answer:"
    )
    prompt = ChatPromptTemplate.from_template(template)
    parser = StrOutputParser()

    def chain_run(question: str, k: int = 4):
        retriever.search_kwargs["k"] = k
        docs = retriever.get_relevant_documents(question)
        context = "\n\n".join(d.page_content for d in docs)
        response = (prompt | llm | parser).invoke({"question": question, "context": context})
        sources = [d.metadata.get("source", "N/A") for d in docs]
        return response, sources

    return chain_run

rag_chain = None

@app.on_event("startup")
def startup_event():
    global rag_chain
    rag_chain = build_rag_chain()

@app.get("/health")
def health():
    return {"status": "ok"}

class AskResponse(BaseModel):
    answer: str
    sources: List[str]

@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    try:
        answer, sources = rag_chain(req.question, req.k)
        return AskResponse(answer=answer.strip(), sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
