# backend/main.py

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Literal
import os

from backend.retrieval import rag
from backend.retrieval.rag import retrieve
from backend.utils.fusion import fuse_context
from backend.processing.evidence import extract_evidence, extract_sources
from backend.llm.llm import ask_llm, REFUSAL_STR
from backend.retrieval.rerank import top_score
from backend.utils.scoring import calculate_confidence

app = FastAPI(title="HarrisonGPT")

SMART_SUMMARY_K = int(os.getenv("SMART_SUMMARY_K", "48"))
SMART_SUMMARY_FINAL_K = int(os.getenv("SMART_SUMMARY_FINAL_K", "12"))
SMART_SUMMARY_RERANK_POOL = int(os.getenv("SMART_SUMMARY_RERANK_POOL", "16"))

# --------------------------------------------------------------------
# REQUEST / RESPONSE SCHEMAS
# --------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    mode: Literal["qa", "smart_summary"] = "smart_summary"  # default to smart summary


class QueryResponse(BaseModel):
    """Structured response returned by the /ask endpoint.

    Fields
    ------
    answer     : The LLM-generated answer to the query.
    confidence : Confidence level of the answer (e.g. "High", "Medium", "Low").
                 Defaults to "Pending" until the confidence-scoring backend is
                 implemented in Phase 2.
    sources    : Ordered list of source page references supporting the answer.
                 Defaults to an empty list until source extraction is wired up.
    """

    answer: str
    confidence: str = Field(default="Pending", description="Confidence level of the answer")
    sources: List[str] = Field(default_factory=list, description="Source page references")


# --------------------------------------------------------------------
# API ENDPOINT
# --------------------------------------------------------------------

@app.post("/ask", response_model=QueryResponse)
def ask_question(req: QueryRequest) -> QueryResponse:
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

    # ----------------------------------------------------------------
    # Phase 3 – populate confidence and sources from scoring pipeline.
    # ----------------------------------------------------------------

    # 5️⃣ Extract the top Cross-Encoder score for confidence scoring.
    #    Returns 0.0 safely when retrieved_chunks is empty.
    best_score: float = top_score(retrieved_chunks)

    # 6️⃣ Extract unique, sorted page references for the sources field.
    #    Returns [] safely when retrieved_chunks is empty.
    sources: List[str] = extract_sources(retrieved_chunks)

    # 7️⃣ Determine whether verification actually ran.
    #    verify_answer() is called unconditionally inside ask_llm() whenever
    #    the LLM returns a real response.  We consider the answer "verified"
    #    when ask_llm() did not return a known refusal/error sentinel.
    was_verified: bool = bool(
        answer
        and answer != REFUSAL_STR
        and not answer.startswith("LLM call failed:")
    )

    # 8️⃣ Calculate the unified confidence label.
    confidence: str = calculate_confidence(
        top_reranker_score=best_score,
        evidence_count=len(evidence),
        was_verified=was_verified,
    )

    return QueryResponse(
        answer=answer,
        confidence=confidence,
        sources=sources,
    )


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
