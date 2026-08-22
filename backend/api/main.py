# backend/main.py

import logging
import os
import secrets
import threading
import time as _time
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# backend.config first: it loads backend/.env, and configure_logging() reads
# HARRISON_LOG_LEVEL from the environment.  Imported the other way round, a log
# level set in .env was read before the file had been loaded and never applied.
# This module imports nothing under backend/ and logs nothing, so it is safe
# ahead of configure_logging().
from backend.config import (
    DEFAULT_K,
    DEFAULT_RERANK_POOL,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    LLM_TOTAL_REQUEST_BUDGET_SECONDS,
    RERANK_MODEL,
    RERANK_SCORE_THRESHOLD,
    RRF_K,
)

# Configure the "backend" logger before importing anything under it, so that
# diagnostics emitted during module import are captured too.  See
# backend/logging_config.py for why this is needed under uvicorn.
from backend.logging_config import configure_logging
from backend.observability import (
    metrics,
    new_request_id,
    request_id_var,
    start_request_budget,
)

configure_logging()

log = logging.getLogger(__name__)

from backend.agents.confidence_scorer import answer_declines, calculate_confidence
from backend.agents.context_router import route_and_sort_context
from backend.agents.query_optimizer import optimize_query
from backend.agents.semantic_cache import SemanticCache
from backend.llm.llm import ask_llm, key_manager, llm_router, resolve_models
from backend.processing.evidence import extract_evidence, extract_sources
from backend.rendering.page_resolver import resolve_page_urls
from backend.retrieval import rag
from backend.retrieval.embeddings import embed_text, embedding_dimension
from backend.retrieval.embeddings import warmup as embeddings_warmup
from backend.retrieval.rag import retrieve
from backend.retrieval.rerank import warmup_reranker
from backend.utils.fusion import fuse_context, selected_chunk_ids

MAX_QUERY_CHARS = int(os.getenv("HARRISON_MAX_QUERY_CHARS", "2000"))
RATE_LIMIT_PER_MINUTE = int(os.getenv("HARRISON_RATE_LIMIT_PER_MINUTE", "30"))
ADMIN_TOKEN = os.getenv("HARRISON_ADMIN_TOKEN", "").strip()
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("HARRISON_CORS_ORIGINS", "").split(",")
    if origin.strip()
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pay every one-off cost here rather than at import or on first request.

    Loading the encoder, the FAISS index, the chunk registry and the BM25 index
    is deliberate startup work.  Doing it at import time — as this project used
    to — made every test run and diagnostic script pay for it, and made module
    import depend on a live network call to Google's model-list API.
    """
    started = _time.perf_counter()

    embeddings_warmup()
    warmup_reranker()
    rag.warmup()

    try:
        prod, backup = resolve_models()
        log.info("HarrisonGPT: LLM models resolved  prod=%s  backup=%s", prod, backup)
    except Exception:
        log.exception("HarrisonGPT: model discovery failed at startup; defaults will be used.")

    log.info("HarrisonGPT: warm-up complete in %.1fs.", _time.perf_counter() - started)

    if not ADMIN_TOKEN:
        log.warning(
            "HarrisonGPT: HARRISON_ADMIN_TOKEN is unset — /admin/* endpoints are disabled."
        )
    yield


app = FastAPI(title="HarrisonGPT", lifespan=lifespan)

# --------------------------------------------------------------------
# CORS — opt-in only.  An empty HARRISON_CORS_ORIGINS means no browser
# origin may call this API cross-site, which is the right default for a
# service that answers from a licensed textbook.
# --------------------------------------------------------------------
if CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )


# --------------------------------------------------------------------
# RATE LIMITING — fixed window per client, in-process.
# Retrieval runs a cross-encoder and two Gemini calls per request; without
# a bound, one client can drain the whole key pool.  Deliberately simple:
# a single process, no dependency.  Put a real limiter in front of this if
# you ever run more than one worker.
# --------------------------------------------------------------------
_rate_lock = threading.Lock()
_rate_hits: dict[str, deque] = {}


def _rate_limit_exceeded(client: str) -> bool:
    if RATE_LIMIT_PER_MINUTE <= 0:
        return False
    now = _time.monotonic()
    with _rate_lock:
        hits = _rate_hits.setdefault(client, deque())
        while hits and now - hits[0] > 60.0:
            hits.popleft()
        if len(hits) >= RATE_LIMIT_PER_MINUTE:
            return True
        hits.append(now)
        # Keep the table from growing without bound across many clients.
        if len(_rate_hits) > 10_000:
            for key in [k for k, v in _rate_hits.items() if not v]:
                _rate_hits.pop(key, None)
        return False


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Tag the request so every log line it produces can be correlated."""
    incoming = request.headers.get("X-Request-ID", "").strip()
    request_id = incoming[:64] if incoming else new_request_id()
    token = request_id_var.set(request_id)
    try:
        response = await call_next(request)
    finally:
        request_id_var.reset(token)
    response.headers["X-Request-ID"] = request_id
    return response


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.url.path == "/ask":
        client = request.client.host if request.client else "unknown"
        if _rate_limit_exceeded(client):
            metrics.increment("rate_limited")
            log.warning("rate_limit: client=%s exceeded %d req/min", client, RATE_LIMIT_PER_MINUTE)
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"detail": "Rate limit exceeded. Please retry in a minute."},
                headers={"Retry-After": "60"},
            )
    return await call_next(request)


# --------------------------------------------------------------------
# EXCEPTION BOUNDARY — never return a stack trace to a caller.
# --------------------------------------------------------------------
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    metrics.increment("unhandled_error")
    log.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error. The failure has been logged."},
    )


# --------------------------------------------------------------------
# ADMIN AUTH — a shared token supplied via X-Admin-Token.
# With no token configured the admin surface is closed, not open.
# --------------------------------------------------------------------
def require_admin(x_admin_token: str = Header(default="")) -> None:
    if not ADMIN_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin endpoints are disabled. Set HARRISON_ADMIN_TOKEN to enable them.",
        )
    if not secrets.compare_digest(x_admin_token or "", ADMIN_TOKEN):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing admin token.",
        )

# --------------------------------------------------------------------
# STATIC FILES – serve pre-rendered Harrison page images
# storage/pages/small/  →  /pages/small/<filename>
# storage/pages/full/   →  /pages/full/<filename>
# --------------------------------------------------------------------
_STORAGE_DIR = Path(__file__).resolve().parents[2] / "storage" / "pages"
_STORAGE_DIR.mkdir(parents=True, exist_ok=True)   # ensure dir exists at startup
app.mount("/pages", StaticFiles(directory=str(_STORAGE_DIR)), name="pages")

# --------------------------------------------------------------------
# WEB UI — Jinja templates + the design-system stylesheets.
# Jinja2 is already present as a Starlette dependency, so this adds no new
# package (CODING_RULES §6.1).  Both are path-resolution only at import: no
# I/O, no network, nothing that would breach import-time purity.
# --------------------------------------------------------------------
_UI_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(_UI_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(_UI_DIR / "templates"))


@app.get("/", include_in_schema=False)
def landing_page(request: Request):
    """Marketing page.  Reads the corpus size from the already-warm index.

    ``rag.chunks`` is forced by the lifespan handler, so this is a length check
    on a loaded list rather than a load.  When the index is degraded it is
    empty, and the page shows a dash instead of claiming zero passages.
    """
    count = len(rag.chunks) if isinstance(rag.chunks, list) else 0
    return templates.TemplateResponse(
        request=request,
        name="landing.html",
        context={"chunk_count": f"{count:,}" if count else "\u2014"},
    )


@app.get("/chat", include_in_schema=False)
def chat_page(request: Request):
    """The ask interface.  Posts to /ask from the same origin, so the browser
    never needs HARRISON_CORS_ORIGINS to be set."""
    return templates.TemplateResponse(
        request=request,
        name="chat.html",
        context={"max_query_chars": MAX_QUERY_CHARS},
    )


SMART_SUMMARY_K = int(os.getenv("SMART_SUMMARY_K", "48"))
SMART_SUMMARY_FINAL_K = int(os.getenv("SMART_SUMMARY_FINAL_K", "12"))
SMART_SUMMARY_RERANK_POOL = int(os.getenv("SMART_SUMMARY_RERANK_POOL", "16"))
# Bumped to v4 with the groundedness floor below: a cached entry stores the
# confidence that was computed when it was written, so the 14 answers already
# on disk -- including the citation-free thyroid-storm refusal that shipped as
# Medium -- would keep serving the pre-fix label forever.  The signature is an
# exact-match gate, so bumping it retires them; they recompute on next ask.
CACHE_SCHEMA_VERSION = "semantic-cache-v4"

# --------------------------------------------------------------------
# SEMANTIC CACHE — global singleton, loaded once at startup from disk.
# Provides sub-100ms responses for repeated or near-identical queries.
# --------------------------------------------------------------------
_cache = SemanticCache()


def _vectorstore_fingerprint() -> dict[str, int | None]:
    """Small cache-busting fingerprint for the loaded vectorstore."""
    return {
        "faiss_dim": int(rag.index.d) if rag.index is not None else None,
        "faiss_ntotal": int(rag.index.ntotal) if rag.index is not None else None,
        "chunk_count": len(rag.chunks) if isinstance(rag.chunks, list) else None,
    }


def _cache_signature(
    *,
    mode: str,
    disable_verifier: bool,
    final_k: int,
) -> dict[str, Any]:
    """Exact-match metadata required before semantic cache similarity."""
    retrieval_k = SMART_SUMMARY_K if mode == "smart_summary" else DEFAULT_K
    rerank_pool = (
        SMART_SUMMARY_RERANK_POOL
        if mode == "smart_summary"
        else DEFAULT_RERANK_POOL
    )
    return {
        "schema": CACHE_SCHEMA_VERSION,
        "mode": mode,
        "disable_verifier": bool(disable_verifier),
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
        "retrieval_k": retrieval_k,
        "final_k": final_k,
        "rerank_pool": rerank_pool,
        "rrf_k": RRF_K,
        "rerank_model": RERANK_MODEL,
        "rerank_score_threshold": RERANK_SCORE_THRESHOLD,
        **_vectorstore_fingerprint(),
    }


def _final_k_for(mode: str, complexity: str) -> int:
    """Return the mode-aware final context count with configured caps."""
    dynamic_final_k = 5 if complexity == "simple" else 12
    if mode == "smart_summary":
        return min(dynamic_final_k, SMART_SUMMARY_FINAL_K)
    return dynamic_final_k


def _candidate_signatures(*, mode: str, disable_verifier: bool) -> list[dict[str, Any]]:
    """Every signature this request could legitimately hit.

    ``final_k`` is the only signature field the query optimizer decides, via
    ``complexity``, and it has exactly two possible values.  Probing both is a
    pair of in-memory matmuls; waiting for the optimizer to tell us which one
    to probe cost a full Groq round-trip on every cache hit.
    """
    candidates: list[dict[str, Any]] = []
    for complexity in ("simple", "complex"):
        signature = _cache_signature(
            mode=mode,
            disable_verifier=disable_verifier,
            final_k=_final_k_for(mode, complexity),
        )
        if signature not in candidates:
            candidates.append(signature)
    return candidates


def _should_save_to_cache(
    *,
    disable_verifier: bool,
    returned_path: str,
    was_truncated: bool,
) -> bool:
    """Cache only fully verified, complete answers."""
    return (
        not disable_verifier
        and returned_path == "verified"
        and not was_truncated
    )

# --------------------------------------------------------------------
# REQUEST / RESPONSE SCHEMAS
# --------------------------------------------------------------------

class QueryRequest(BaseModel):
    # Bounded: an unbounded query is embedded by BGE-M3 and sent to the LLM,
    # so a multi-megabyte string is both a CPU and a cost amplifier.
    query: str = Field(min_length=1, max_length=MAX_QUERY_CHARS)
    mode: Literal["qa", "smart_summary"] = "smart_summary"  # default to smart summary
    disable_verifier: bool = False


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
    timings        : Timing breakdown of different pipeline stages in seconds.
    """

    answer: str
    confidence: str = Field(default="Pending", description="Confidence level of the answer")
    sources: list[str] = Field(default_factory=list, description="Source page references")
    visual_context: list[dict[str, str]] = Field(
        default_factory=list,
        description="Image URLs for each source page (thumbnail_url, full_url)",
    )
    timings: dict[str, float] = Field(
        default_factory=dict,
        description="Timing breakdown of the request stages in seconds",
    )


# --------------------------------------------------------------------
# API ENDPOINT
# --------------------------------------------------------------------

@app.post("/ask", response_model=QueryResponse)
def ask_question(req: QueryRequest, request: Request) -> QueryResponse:
    import time
    start_total = time.perf_counter()
    timings = {
        "optimizer": 0.0,
        "retrieval": 0.0,
        "reranking": 0.0,
        "draft_generation": 0.0,
        "verification": 0.0,
        "retry": 0.0,
    }

    raw_query = req.query
    mode = req.mode

    # Open the wall-clock budget for everything below.  Stage deadlines clamp
    # against what is left of it and retries stop when it is spent.
    start_request_budget(LLM_TOTAL_REQUEST_BUDGET_SECONDS)

    # ----------------------------------------------------------------
    # 0️⃣  Semantic Cache — probed before any LLM call
    # ----------------------------------------------------------------
    # This used to sit *behind* the optimizer, so every hit paid a full Groq
    # round-trip (measured 697ms) to look up an answer already on disk.  The
    # embedding is taken from the raw query rather than the expanded one for
    # the same reason: the expansion is an LLM output, and keying on it makes
    # the key unobtainable without the call the cache exists to avoid.
    #
    # A hit skips the non-medical gatekeeper, which is safe: only a query that
    # already passed it can be in the cache, and the 0.95 similarity floor
    # keeps a non-medical query from matching a medical entry.
    query_embedding: list[float] = embed_text(raw_query).flatten().tolist()

    for candidate in _candidate_signatures(mode=mode, disable_verifier=req.disable_verifier):
        cached = _cache.check_cache(query_embedding, metadata=candidate)
        if cached is None:
            continue
        total_time = time.perf_counter() - start_total
        timings["total_request"] = total_time
        log.info("ask_question: TIMINGS (cache hit)  total_request=%.3fs", total_time)
        metrics.increment("ask_total")
        metrics.increment("cache_hit")
        metrics.observe_timings(timings)
        # visual_context is rebuilt from the cached page labels against THIS
        # request's host.  It used to be served straight from the cache, which
        # baked in whichever host first populated the entry — serve the same
        # cache behind a different hostname and every image link pointed at the
        # old one.  base_url is deliberately not part of the cache signature;
        # the labels are host-independent, the URLs are not.
        return QueryResponse(
            answer=cached["answer"],
            confidence=cached["confidence"],
            sources=cached["sources"],
            visual_context=resolve_page_urls(
                sources=cached.get("sources", []),
                base_url=str(request.base_url).rstrip("/"),
            ),
            timings=timings,
        )

    # ----------------------------------------------------------------
    # 1️⃣  QueryOptimizer — pre-retrieval gatekeeper & context enhancer
    # ----------------------------------------------------------------
    # The agent runs a fast LLM call (llama-3.1-8b-instant) to:
    #   a) Detect whether the query is medical in nature.
    #   b) Expand acronyms and add clinical framing.
    # If the LLM is unavailable, optimize_query() falls back to the
    # original query transparently (CODING_RULES.md §3.2).
    t_opt_start = time.perf_counter()
    optimized = optimize_query(raw_query)
    timings["optimizer"] = time.perf_counter() - t_opt_start
    optimizer_failed: bool = not optimized["optimizer_used"]
    fallback_to_original: bool = optimizer_failed or (optimized["expanded_query"] == raw_query)

    # ── Gatekeeper: non-medical queries short-circuit immediately ──
    # This bypasses all retrieval and LLM generation, saving compute
    # and keeping Harrison-specific context intact.
    if not optimized["is_medical_query"]:
        total_time = time.perf_counter() - start_total
        timings["total_request"] = total_time
        log.info(
            "ask_question: TIMINGS (non-medical query early exit)  total_request=%.3fs  optimizer=%.3fs",
            total_time,
            timings["optimizer"],
        )
        metrics.increment("ask_total")
        metrics.increment("ask_non_medical")
        metrics.observe_timings(timings)
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
            timings=timings,
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
    dynamic_final_k = _final_k_for(mode, _complexity)
    # The signature the answer will actually be stored under.  The probe above
    # tried both candidates; this is the one this request really used.
    cache_signature = _cache_signature(
        mode=mode,
        disable_verifier=req.disable_verifier,
        final_k=dynamic_final_k,
    )

    # 2️⃣ Retrieve (final_k scaled by query complexity)
    if mode == "smart_summary":
        retrieved_chunks = retrieve(
            search_query,
            k=SMART_SUMMARY_K,
            final_k=dynamic_final_k,
            rerank_pool=SMART_SUMMARY_RERANK_POOL,
            timings=timings,
        )
    else:
        retrieved_chunks = retrieve(
            search_query,
            final_k=dynamic_final_k,
            timings=timings,
        )

    # 2.5️⃣ ContextRouter — deduplicate & chronological sort
    # Drops near-identical chunks (>90% overlap) and re-orders survivors
    # by ascending page number so the LLM reads Harrison sequentially.
    # Pure function: no LLM call, sub-millisecond, crash-safe.
    retrieved_chunks = route_and_sort_context(retrieved_chunks)

    # 3️⃣ Fuse context
    fused_context = fuse_context(retrieved_chunks)

    # 4️⃣ Extract structured evidence for the chunks the context could not carry.
    #    Building both blocks from the same list sent every context chunk to the
    #    model twice; excluding them removes the duplication without losing a
    #    single chunk.
    evidence = extract_evidence(
        retrieved_chunks,
        exclude_chunk_ids=selected_chunk_ids(retrieved_chunks),
    )

    # 5️⃣ Ask LLM
    #    question= uses raw_query so the answer is phrased naturally for
    #    the user; the enriched context already reflects search_query.
    #    ask_llm() returns a 4-tuple: (final_answer, draft_answer, was_truncated, returned_path).
    final_answer, draft_answer, was_truncated, returned_path = ask_llm(
        fused_context=fused_context,
        question=raw_query,
        mode=mode,
        evidence=evidence,
        timings=timings,
        disable_verifier=req.disable_verifier,
    )
    answer: str = final_answer

    # ----------------------------------------------------------------
    # Phase 3 – populate confidence and sources from scoring pipeline.
    # ----------------------------------------------------------------

    # 6️⃣ Extract unique, sorted page references for the sources field.
    #    Returns [] safely when retrieved_chunks is empty.
    sources: list[str] = extract_sources(retrieved_chunks)

    # 7️⃣ Calculate the deterministic confidence label.
    #    ConfidenceScorer combines two signals:
    #      a) Average Cross-Encoder score across all retrieved chunks
    #         (not just the top-1) for a richer retrieval quality estimate.
    #      b) Length-ratio divergence between the draft and verified answer
    #         — a proxy for how many unsupported claims the verifier pruned.
    #    Pass the actual draft_answer and final verified answer to unlock the penalty.
    confidence: str = calculate_confidence(
        chunks=retrieved_chunks,
        original_answer=draft_answer,
        verified_answer=answer,
    )

    # ── Confidence cap: enforce trust levels based on returned_path + truncation ──
    #
    # Rule table:
    #   verified          + not truncated → any confidence allowed (High OK)
    #   verified          + truncated     → Medium max (draft was cut)
    #   draft_fallback    + any           → Medium max (verifier failed)
    #   graceful_fallback + any           → Low max (both layers failed)
    #   no_grounding      + any           → Low max (no usable context)
    #   provider_failure  + any           → Low max (API/quota/outage)
    #   no [p:NNN] citation + any path    → Low max (answer is not grounded)
    #   answer declines up front + any path → Low max (it is a refusal)
    #
    # These are the only paths ask_llm() returns; the table previously also
    # handled a "partial_verified" path that nothing ever produced.
    if returned_path in ("graceful_fallback", "no_grounding", "provider_failure"):
        confidence = "Low"
    elif returned_path == "draft_fallback" and confidence == "High":
        confidence = "Medium"
    elif was_truncated and confidence == "High":
        confidence = "Medium"

    # The rules above key off the *path*, which says how the answer was produced,
    # not whether it ended up grounded.  When retrieval brings back plausible but
    # off-target pages the model declines in its own words -- "The provided
    # context does not contain information regarding the clinical features or
    # management of thyroid storm" -- on the `verified` path, where none of the
    # caps fire.  That shipped as Medium confidence with three Harrison pages and
    # their thumbnails attached: a refusal dressed as a moderately confident
    # answer, which is the wrong direction for this system to be wrong in.
    # An answer carrying no citation is not grounded in the context, whatever
    # path produced it.  REFUSAL_STR has no citation either and is already Low.
    #
    # Citation presence alone is too weak a test, though: re-running the thyroid
    # storm query live returned "... are not detailed in the provided chapters"
    # *with* a [p:3074] citation attached to a true but tangential fact about
    # subacute thyroiditis, and it shipped Medium again.  A padded refusal is
    # still a refusal, so ask the answer what it says about the context.
    if "[p:" not in answer or answer_declines(answer):
        confidence = "Low"

    # ── Consolidated request-level structured log (single grep-friendly line) ──
    log.info(
        "ask_question: FINAL  returned_path=%s  confidence=%s  was_truncated=%s  "
        "optimizer_failed=%s  fallback_to_original_query=%s  "
        "expanded_query_equals_raw=%s  cache_hit=False  disable_verifier=%s  "
        "mode=%s  source_count=%d  query=%r",
        returned_path,
        confidence,
        was_truncated,
        optimizer_failed,
        fallback_to_original,
        search_query == raw_query,
        req.disable_verifier,
        mode,
        len(sources),
        raw_query,
    )

    # 🔟 Resolve source page labels to image URLs.
    #    base_url is derived from the live Request so this works on any
    #    host/port without hardcoding (localhost, staging, or production).
    base_url: str = str(request.base_url).rstrip("/")
    visual_context: list[dict[str, str]] = resolve_page_urls(
        sources=sources,
        base_url=base_url,
    )

    # ── Persist only fully verified responses to semantic cache ──
    if _should_save_to_cache(
        disable_verifier=req.disable_verifier,
        returned_path=returned_path,
        was_truncated=was_truncated,
    ):
        _cache.save_to_cache(
            query_embedding=query_embedding,
            metadata=cache_signature,
            audit_data={
                "raw_query": raw_query,
                "search_query": search_query,
                "returned_path": returned_path,
                "was_truncated": was_truncated,
                "optimizer_failed": optimizer_failed,
                "fallback_to_original_query": fallback_to_original,
            },
            # Only host-independent data is persisted; visual_context is
            # derived from `sources` on read.
            response_data={
                "answer":     answer,
                "confidence": confidence,
                "sources":    sources,
            },
        )
    else:
        log.info(
            "ask_question: semantic cache save skipped  disable_verifier=%s  "
            "returned_path=%s  was_truncated=%s",
            req.disable_verifier,
            returned_path,
            was_truncated,
        )

    total_time = time.perf_counter() - start_total
    timings["total_request"] = total_time

    log.info(
        "ask_question: TIMINGS  total_request=%.3fs  optimizer=%.3fs  retrieval=%.3fs  "
        "reranking=%.3fs  draft_generation=%.3fs  verification=%.3fs  retry=%.3fs",
        total_time,
        timings["optimizer"],
        timings["retrieval"],
        timings["reranking"],
        timings["draft_generation"],
        timings["verification"],
        timings["retry"],
    )

    metrics.increment("ask_total")
    metrics.increment("cache_miss")
    metrics.increment(f"returned_path_{returned_path}")
    metrics.increment(f"confidence_{confidence.lower()}")
    if optimizer_failed:
        metrics.increment("optimizer_failed")
    if was_truncated:
        metrics.increment("truncated")
    metrics.observe_timings(timings)

    return QueryResponse(
        answer=answer,
        confidence=confidence,
        sources=sources,
        visual_context=visual_context,
        timings=timings,
    )


@app.get("/health")
def health_check():
    faiss_loaded = rag.index is not None
    chunks_loaded = isinstance(rag.chunks, list) and len(rag.chunks) > 0
    gemini_key_present = key_manager.has_keys()
    faiss_dim = int(rag.index.d) if rag.index is not None else None
    embed_dim = embedding_dimension()
    embedding_index_dim_match = (
        faiss_dim is not None
        and embed_dim == faiss_dim
    )

    healthy = bool(
        faiss_loaded
        and chunks_loaded
        and gemini_key_present
        and embedding_index_dim_match
    )

    body = {
        "status": "ok" if healthy else "degraded",
        "faiss_loaded": faiss_loaded,
        "chunks_loaded": chunks_loaded,
        "faiss_dim": faiss_dim,
        "embedding_dim": embed_dim,
        "embedding_index_dim_match": embedding_index_dim_match,
        "gemini_key_present": gemini_key_present,
        "gemini_key_count": key_manager.key_count,
        "gemini_available_key_count": key_manager.available_key_count,
        "llm_providers": llm_router.status(),
    }

    # A degraded system must say so in the status line, not only in the body —
    # returning 200 here meant no load balancer or orchestrator could ever act
    # on the diagnosis this endpoint was built to produce.
    return JSONResponse(
        status_code=status.HTTP_200_OK if healthy else status.HTTP_503_SERVICE_UNAVAILABLE,
        content=body,
    )


@app.get("/metrics")
def read_metrics():
    """Aggregate pipeline counters and stage latencies for this process.

    The per-request ``timings`` were previously returned to the caller and
    then thrown away; these are the same numbers accumulated, so questions
    like "how often does the verifier fall back" and "what is p95 retrieval"
    can be answered without replaying logs.
    """
    return metrics.snapshot()


# --------------------------------------------------------------------
# ADMIN ENDPOINTS
# --------------------------------------------------------------------

@app.delete("/admin/cache", dependencies=[Depends(require_admin)])
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
