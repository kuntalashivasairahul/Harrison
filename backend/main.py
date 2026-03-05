# backend/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal
import os

import rag

from rag import retrieve
from fusion import fuse_context
from evidence import extract_evidence
from llm import ask_llm

app = FastAPI(title="HarrisonGPT")

SMART_SUMMARY_K = int(os.getenv("SMART_SUMMARY_K", "48"))
SMART_SUMMARY_FINAL_K = int(os.getenv("SMART_SUMMARY_FINAL_K", "12"))
SMART_SUMMARY_RERANK_POOL = int(os.getenv("SMART_SUMMARY_RERANK_POOL", "16"))

# --------------------------------------------------------------------
# REQUEST SCHEMA
# --------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    mode: Literal["qa", "smart_summary"] = "smart_summary"  # default to smart summary


# --------------------------------------------------------------------
# API ENDPOINT
# --------------------------------------------------------------------

@app.post("/ask")
def ask_question(req: QueryRequest):
    query = req.query
    mode = req.mode

    # 1️⃣ Retrieve
    if mode == "smart_summary":
        retrieved_chunks = retrieve(
            query,
            k=SMART_SUMMARY_K,
            final_k=SMART_SUMMARY_FINAL_K,
            rerank_pool=SMART_SUMMARY_RERANK_POOL,
        )
    else:
        retrieved_chunks = retrieve(query)

    # 2️⃣ Fuse context
    fused_context = fuse_context(retrieved_chunks)

    # 3️⃣ Extract structured evidence statements
    evidence = extract_evidence(retrieved_chunks)

    # 4️⃣ Ask LLM
    answer = ask_llm(
        fused_context=fused_context,
        question=query,
        mode=mode,
        evidence=evidence,
    )

    return {
        "query": query,
        "mode": mode,
        "answer": answer
    }


@app.get("/health")
def health_check():
    faiss_loaded = rag.index is not None
    chunks_loaded = isinstance(rag.chunks, list) and len(rag.chunks) > 0
    groq_key_present = bool(os.getenv("GROQ_API_KEY"))

    return {
        "status": "ok" if (faiss_loaded and chunks_loaded and groq_key_present) else "degraded",
        "faiss_loaded": faiss_loaded,
        "chunks_loaded": chunks_loaded,
        "groq_key_present": groq_key_present,
    }
