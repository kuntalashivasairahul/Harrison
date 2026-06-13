# backend/main.py

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Literal
from pathlib import Path
import os

from backend.retrieval import rag
from backend.retrieval.rag import retrieve
from backend.utils.fusion import fuse_context
from backend.processing.evidence import extract_evidence, extract_sources
from backend.llm.llm import ask_llm, REFUSAL_STR
from backend.agents.confidence_scorer import calculate_confidence
from backend.rendering.page_resolver import resolve_page_urls
from backend.agents.query_optimizer import optimize_query
from backend.agents.semantic_cache import SemanticCache
from backend.agents.context_router import route_and_sort_context
from backend.retrieval.embeddings import embed_text

app = FastAPI(title="HarrisonGPT")

# --------------------------------------------------------------------
# STATIC FILES – serve pre-rendered Harrison page images
# storage/pages/small/  →  /pages/small/<filename>
# storage/pages/full/   →  /pages/full/<filename>
# --------------------------------------------------------------------
_STORAGE_DIR = Path(__file__).resolve().parents[2] / "storage" / "pages"
_STORAGE_DIR.mkdir(parents=True, exist_ok=True)   # ensure dir exists at startup
app.mount("/pages", StaticFiles(directory=str(_STORAGE_DIR)), name="pages")

SMART_SUMMARY_K = int(os.getenv("SMART_SUMMARY_K", "48"))
SMART_SUMMARY_FINAL_K = int(os.getenv("SMART_SUMMARY_FINAL_K", "12"))
SMART_SUMMARY_RERANK_POOL = int(os.getenv("SMART_SUMMARY_RERANK_POOL", "16"))

# --------------------------------------------------------------------
# SEMANTIC CACHE — global singleton, loaded once at startup from disk.
# Provides sub-100ms responses for repeated or near-identical queries.
# --------------------------------------------------------------------
_cache = SemanticCache()

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
    answer         : The LLM-generated answer to the query.
    confidence     : Confidence level of the answer ("High", "Medium", or "Low").
    sources        : Ordered list of source page references (e.g. ["p.142"]).
    visual_context : One entry per source page, each containing the original
                     page_label and absolute thumbnail_url / full_url for the
                     pre-rendered page images served at /pages/*.
    """

    answer: str
    confidence: str = Field(default="Pending", description="Confidence level of the answer")
    sources: List[str] = Field(default_factory=list, description="Source page references")
    visual_context: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Image URLs for each source page (thumbnail_url, full_url)",
    )


# --------------------------------------------------------------------
# API ENDPOINT
# --------------------------------------------------------------------

@app.post("/ask", response_model=QueryResponse)
def ask_question(req: QueryRequest, request: Request) -> QueryResponse:
    raw_query = req.query
    mode = req.mode

    # ----------------------------------------------------------------
    # 0️⃣  QueryOptimizer — pre-retrieval gatekeeper & context enhancer
    # ----------------------------------------------------------------
    # The agent runs a fast LLM call (llama-3.1-8b-instant) to:
    #   a) Detect whether the query is medical in nature.
    #   b) Expand acronyms and add clinical framing.
    # If the LLM is unavailable, optimize_query() falls back to the
    # original query transparently (CODING_RULES.md §3.2).
    optimized = optimize_query(raw_query)

    # ── Gatekeeper: non-medical queries short-circuit immediately ──
    # This bypasses all retrieval and LLM generation, saving compute
    # and keeping Harrison-specific context intact.
    if not optimized["is_medical_query"]:
        return QueryResponse(
            answer=(
                "HarrisonGPT is a medical reference assistant grounded exclusively "
                "in Harrison's Principles of Internal Medicine. I'm only able to "
                "answer clinical questions about diseases, diagnosis, treatment, and "
                "pharmacology. Please rephrase your query as a medical question."
            ),
            confidence="High",   # Highly confident this is out of scope.
            sources=[],
            visual_context=[],
        )

    # ── Enhancement: use the expanded query for retrieval & generation ──
    # The display question shown to the LLM remains the raw user input so
    # the answer reads naturally; the expanded_query drives FAISS/BM25
    # for higher semantic recall.
    search_query: str = optimized["expanded_query"] or raw_query

    # ── Adaptive Retrieval Depth ──────────────────────────────────────
    # The QueryOptimizer classifies each query as "simple" (single-fact
    # lookup) or "complex" (multi-part: pathophysiology + management,
    # diagnostic criteria, scoring systems, etc.).
    # "simple" → final_k=5  : fast, tight context, lower Groq token cost.
    # "complex" → final_k=12 : wider context window prevents fragmentation
    #                          of multi-section clinical protocols.
    # The fallback value in the optimizer is always "complex", so missing
    # or failed LLM calls conservatively maximise recall.
    _complexity: str     = optimized.get("complexity", "complex")
    dynamic_final_k: int = 5 if _complexity == "simple" else 12

    # ----------------------------------------------------------------
    # 1️⃣  Semantic Cache — check before any retrieval or LLM work
    # ----------------------------------------------------------------
    # embed_text() reuses the already-loaded MiniLM model — no extra
    # memory overhead.  The (1, 384) array is flattened to a plain list
    # for JSON-serialisable storage.
    query_embedding: List[float] = embed_text(search_query).flatten().tolist()

    cached = _cache.check_cache(query_embedding)
    if cached is not None:
        # ── Cache HIT: return instantly, zero FAISS/Groq cost ──
        return QueryResponse(
            answer=cached["answer"],
            confidence=cached["confidence"],
            sources=cached["sources"],
            visual_context=cached["visual_context"],
        )

    # ── Cache MISS: run the full pipeline ──

    # 2️⃣ Retrieve (final_k scaled by query complexity)
    if mode == "smart_summary":
        retrieved_chunks = retrieve(
            search_query,
            k=SMART_SUMMARY_K,
            final_k=dynamic_final_k,
            rerank_pool=SMART_SUMMARY_RERANK_POOL,
        )
    else:
        retrieved_chunks = retrieve(search_query, final_k=dynamic_final_k)

    # 2.5️⃣ ContextRouter — deduplicate & chronological sort
    # Drops near-identical chunks (>90% overlap) and re-orders survivors
    # by ascending page number so the LLM reads Harrison sequentially.
    # Pure function: no LLM call, sub-millisecond, crash-safe.
    retrieved_chunks = route_and_sort_context(retrieved_chunks)

    # 3️⃣ Fuse context
    fused_context = fuse_context(retrieved_chunks)

    # 4️⃣ Extract structured evidence statements
    evidence = extract_evidence(retrieved_chunks)

    # 5️⃣ Ask LLM
    #    question= uses raw_query so the answer is phrased naturally for
    #    the user; the enriched context already reflects search_query.
    #    ask_llm() calls verify_answer() internally and returns the final
    #    post-verification text.  We capture it as draft_answer here so we
    #    can also pass it as original_answer to calculate_confidence.
    #    If the draft is inaccessible (llm.py not modified), passing the
    #    same string as both arguments produces zero divergence — no penalty.
    draft_answer: str = ask_llm(
        fused_context=fused_context,
        question=raw_query,
        mode=mode,
        evidence=evidence,
    )
    answer: str = draft_answer

    # ----------------------------------------------------------------
    # Phase 3 – populate confidence and sources from scoring pipeline.
    # ----------------------------------------------------------------

    # 6️⃣ Extract unique, sorted page references for the sources field.
    #    Returns [] safely when retrieved_chunks is empty.
    sources: List[str] = extract_sources(retrieved_chunks)

    # 7️⃣ Calculate the deterministic confidence label.
    #    ConfidenceScorer combines two signals:
    #      a) Average Cross-Encoder score across all retrieved chunks
    #         (not just the top-1) for a richer retrieval quality estimate.
    #      b) Length-ratio divergence between the draft and verified answer
    #         — a proxy for how many unsupported claims the verifier pruned.
    #    ask_llm() fuses both generation and verification internally, so we
    #    pass draft_answer as both arguments (zero divergence, no penalty).
    #    Upgrade llm.py to return (draft, verified) to unlock the penalty.
    confidence: str = calculate_confidence(
        chunks=retrieved_chunks,
        original_answer=draft_answer,
        verified_answer=draft_answer,
    )

    # 🔟 Resolve source page labels to image URLs.
    #    base_url is derived from the live Request so this works on any
    #    host/port without hardcoding (localhost, staging, or production).
    base_url: str = str(request.base_url).rstrip("/")
    visual_context: List[Dict[str, str]] = resolve_page_urls(
        sources=sources,
        base_url=base_url,
    )

    # ── Persist to semantic cache so future similar queries are instant ──
    _cache.save_to_cache(
        query_embedding=query_embedding,
        response_data={
            "answer":         answer,
            "confidence":     confidence,
            "sources":        sources,
            "visual_context": visual_context,
        },
    )

    return QueryResponse(
        answer=answer,
        confidence=confidence,
        sources=sources,
        visual_context=visual_context,
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


# --------------------------------------------------------------------
# ADMIN ENDPOINTS
# --------------------------------------------------------------------

@app.delete("/admin/cache")
def clear_semantic_cache():
    """
    Wipe all entries from the in-memory semantic cache and reset the
    persistent ``artifacts/semantic_cache.json`` file to an empty list.

    Use this during development whenever you deploy a pipeline change
    (new model, new prompt, new offset constant) that would make cached
    responses stale.

    Returns
    -------
    JSON object with:
    - ``status``         : ``"success"``
    - ``message``        : Human-readable confirmation.
    - ``entries_cleared``: Number of cache entries that were removed.
    """
    entries_before: int = _cache.size
    _cache.clear()
    return {
        "status":          "success",
        "message":         "Semantic cache cleared successfully.",
        "entries_cleared": entries_before,
    }

