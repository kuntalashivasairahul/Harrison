# ARCHITECTURE.md
# HarrisonGPT — System Architecture Reference

> **Immutability Notice.**
> The pipeline stages documented below are **core, non-negotiable components**.
> AI tools, future contributors, and automated refactoring agents **must not**
> remove, reorder, or bypass any stage — especially the hard filter,
> `verify_answer()`, and `resolve_page_urls()` steps.

---

## 1. End-to-End Pipeline

The request lifecycle for `/ask` is:

1. `optimize_query()` calls the approved router: Groq first when explicitly enabled, Gemini Flash-Lite second, then a deterministic local fallback. It classifies the query, rejects non-medical requests, expands the search query, and labels complexity. This is the **only** query-expansion stage; the rule-based `expand_query()` that used to fan retrieval out over four templated variants has been removed.
2. The API embeds the search query with `BAAI/bge-m3` and checks the semantic cache against exact runtime metadata.
3. A cache miss runs a single hybrid FAISS/BM25 pass, RRF fusion (`RRF_K=60`), neighbor expansion (restricted to chunks within one page of their parent), cross-encoder reranking, and the `RERANK_SCORE_THRESHOLD=-3.0` hard filter.
4. `route_and_sort_context()` deduplicates and page-orders chunks; `fuse_context()` constructs the prompt context within `SMART_SUMMARY_CONTEXT_CHAR_LIMIT` (default 12,000 characters).

   **Ordering contract:** fusion selects which chunks survive the budget in
   descending cross-encoder score, then emits the survivors in page order.
   Selecting in page order instead — as it previously did — makes the budget
   discard the highest-numbered pages rather than the least relevant chunks,
   silently dropping the top-ranked chunk whenever it came from late in the
   textbook.
5. Evidence and source labels are extracted from the retrieved chunks.
6. `ask_llm()` uses the stage-aware router, which tries every deployment
   registered for the stage in priority order — `gemini-primary`, then
   `gemini-draft-fallback` on a different Gemini model. A provider-side outage
   on one model no longer fails the request. `KeyManager` uses `GEMINI_API_KEY` plus `GEMINI_API_KEY_1` through `_10` as distinct round-robin projects and temporarily cools down individual projects after quota responses.
7. Unless `disable_verifier` is requested, `verify_answer()` performs a grounded Gemini rewrite at temperature `0.0`. A complete verified response is the only response eligible for semantic-cache persistence.
8. `backend/agents/confidence_scorer.py` scores the final response from average cross-encoder relevance and draft-to-verified length divergence; return-path and truncation caps are then applied by the API.
9. Source labels are resolved into page-image URLs and returned with timing data in `QueryResponse`.

---

## 1a. Import-Time Purity

No module under `backend/` may do expensive or networked work at import time.
Specifically:

- `backend/retrieval/rag.py` loads `chunks.json`, the FAISS index, and the BM25
  index **on first access**, via PEP 562 `__getattr__`. `rag.warmup()` forces it.
- `backend/retrieval/embeddings.py` constructs the BGE-M3 encoder on first call
  to `get_model()`. `warmup()` forces it.
- `backend/llm/llm.py` resolves `PROD_MODEL` / `BACKUP_MODEL` through
  `resolve_models()` on first use, not at import. Reading the module attributes
  still works and triggers resolution lazily.

All of it is forced deliberately in the FastAPI `lifespan` handler, so startup
pays the cost once and the first request does not. This is what keeps the test
suite hermetic and fast — importing `backend.api.main` costs ~0.2s with the
heavy modules stubbed, and the full suite runs in ~3s with no network access.

`backend/logging_config.py` must be imported and `configure_logging()` called
**before** any other `backend.*` import in an entry point, so that diagnostics
emitted during module import are captured. Modules obtain loggers with
`logging.getLogger(__name__)`; reaching for `"uvicorn.error"` re-hides the
problem that module exists to solve.

---

## 2. FastAPI Schema — Immutable Contract

The `QueryResponse` schema is a **frozen contract**. Fields must not be
renamed, removed, or re-typed without a full migration plan.

```python
class QueryRequest(BaseModel):
    query: str
    mode: Literal["qa", "smart_summary"] = "smart_summary"
    disable_verifier: bool = False

class QueryResponse(BaseModel):
    answer:         str                   # Verified LLM answer
    confidence:     str                   # "High" | "Medium" | "Low"
    sources:        List[str]             # e.g. ["p.142", "p.512"]
    visual_context: List[Dict[str, str]]  # [{page_label, thumbnail_url, full_url}]
    timings:        Dict[str, float]      # stage durations in seconds
```

**Immutability rules:**
- `confidence` must always be populated by `calculate_confidence()` — never hardcoded.
- `sources` must always come from `extract_sources()` — never inferred from the answer text.
- `visual_context` must always be populated by `resolve_page_urls()` — never constructed inline.

---

## 3. Directory Structure & Module Boundaries

```
Harrison/                              ← Project root
│
├── backend/                           ← All Python source code
│   ├── api/
│   │   └── main.py                    ← FastAPI app, endpoints, schema (QueryRequest/QueryResponse)
│   │                                    ONLY place where HTTP concerns live
│   │
│   ├── retrieval/
│   │   ├── rag.py                     ← Core retrieval pipeline:
│   │   │                                _hybrid_candidates(), _pretrim_for_rerank(),
│   │   │                                retrieve(); lazy vectorstore via warmup()
│   │   │                                Uses RERANK_SCORE_THRESHOLD from config.py
│   │   ├── rerank.py                  ← CrossEncoder wrapper: rerank(), warmup_reranker()
│   │   └── embeddings.py              ← embed_text() — FAISS query embedding
│   │
│   ├── llm/
│   │   ├── llm.py                     ← ask_llm(), verify_answer(), Gemini key management
│   │   ├── router.py                  ← approved deployment routing and cooldown state
│   │   ├── contracts.py               ← provider-neutral request/result/error contracts
│   │   └── model_registry.json        ← Stage 1 provider allowlist
│   │                                    REFUSAL_STR sentinel defined here
│   │
│   ├── processing/
│   │   └── evidence.py                ← extract_evidence(), extract_sources()
│   │                                    Pure functions; no I/O or HTTP
│   │
│   ├── rendering/
│   │   └── page_resolver.py           ← resolve_page_urls()
│   │                                    URL constructor only; no disk I/O
│   │
│   ├── agents/
│   │   └── confidence_scorer.py       ← calculate_confidence()
│   │                                    Pure function; no I/O
│   │
│   ├── utils/
│   │   └── fusion.py                  ← fuse_context(), clean_text()
│   │
│   ├── config.py                      ← Paths, models, retrieval + deadline constants
│   │                                    Deadlines read the environment HERE; call
│   │                                    sites import them, never os.getenv directly
│   ├── logging_config.py              ← configure_logging(); call before backend imports
│   ├── observability.py               ← request-id context, RequestIdFilter, metrics
│   ├── requirements.txt               ← Pinned runtime dependencies
│   ├── requirements-dev.txt           ← Test/lint dependencies
│   └── .env                           ← Secrets (git-ignored)
│
├── artifacts/                         ← Runtime data — DO NOT SCAN
│   ├── vectorstore/
│   │   ├── index.faiss                ← FAISS index (binary)
│   │   └── chunks.json                ← Chunk metadata (text, page, chunk_id)
│   └── retrieval_logs/                ← Per-query diagnostics (opt-in, capped, query withheld)
│
├── storage/                           ← Static media — DO NOT SCAN
│   └── pages/
│       ├── small/                     ← page_NNN_small.webp  (thumbnails)
│       └── full/                      ← page_NNN_full.png    (full-resolution)
│
├── evaluation/                        ← Evaluation harnesses and metrics scripts
│
├── PROJECT_CONTEXT.md                 ← ← ← YOU ARE HERE (sister doc)
├── ARCHITECTURE.md                    ← ← ← THIS FILE
└── CODING_RULES.md                    ← ← ← Enforcement companion
```

---

## 4. Module Dependency Graph

```
api/main.py
    ├── retrieval/rag.py          (retrieve)
    ├── utils/fusion.py           (fuse_context)
    ├── processing/evidence.py    (extract_evidence, extract_sources)
    ├── llm/llm.py                (ask_llm, resolve_models)
    ├── observability.py          (metrics, request ids)
    ├── agents/confidence_scorer.py (calculate_confidence)
    └── rendering/page_resolver.py (resolve_page_urls)

retrieval/rag.py
    ├── retrieval/embeddings.py   (embed_text)
    ├── retrieval/rerank.py       (rerank)
    └── rank_bm25                 (BM25Okapi)

retrieval/embeddings.py
    └── sentence_transformers     (SentenceTransformer)

retrieval/rerank.py
    └── sentence_transformers     (CrossEncoder)

llm/llm.py
    └── google.genai              (Gemini client and types)

processing/evidence.py
    └── utils/fusion.py           (clean_text)
```

**Rule:** The `api/` layer is the **only** layer that wires modules together.
Individual modules (`retrieval/`, `llm/`, `processing/`, `rendering/`,
`utils/`) must remain independent of each other — they do not import sideways.

---

## 5. Retrieval Scoring Deep-Dive

### 5.1 RRF Score Formula

```
rrf_score(chunk) = 1/(K + faiss_rank) + 1/(K + bm25_rank)
```

Where `K = 60` (the standard Reciprocal Rank Fusion constant). Chunks that
appear in only one index receive only one term.

### 5.2 Confidence Score Decision Matrix

| Condition                                                      | Label      |
|----------------------------------------------------------------|------------|
| No retrieved chunks, or half or more carry no usable cross-encoder score | **Low**  |
| Average score `< -2.0`                                          | **Low**  |
| Draft-to-verified length divergence `> 0.40`                    | **Low**  |
| Average score `>= -0.5`, at least two scored chunks, none unscored | **High** |
| Everything else                                                 | **Medium** |

### 5.3 Hard Filter Threshold

```python
RERANK_SCORE_THRESHOLD = -3.0  # raw ms-marco-MiniLM logit
```

Chunks scoring below this value are dropped **after** reranking but **before**
context fusion. It is defined in `backend/config.py` and used by
`backend/retrieval/rag.py`; it is not configurable at the API layer.

---

## 5.4 Endpoints

| Method | Path | Auth | Notes |
|---|---|---|---|
| `POST` | `/ask` | none | Rate-limited per client. `query` bounded to `HARRISON_MAX_QUERY_CHARS` (2000). |
| `GET` | `/health` | none | **503** when degraded, 200 when ok. Never rate-limited. |
| `GET` | `/metrics` | none | Counters and p50/p95 per pipeline stage, in-process. |
| `DELETE` | `/admin/cache` | `X-Admin-Token` | 503 when `HARRISON_ADMIN_TOKEN` is unset — closed by default, not open. |

Every response carries `X-Request-ID`; a supplied one is echoed back so a
caller's trace id survives into the logs.

---

## 6. Page Rendering Subsystem

### URL Construction Pattern

```
Input label : "p.142"
thumbnail   : {base_url}/pages/small/page_142_small.webp
full        : {base_url}/pages/full/page_142_full.png
```

- `base_url` is derived from the live `Request` object in FastAPI — **never
  hardcoded** — so the same code works on localhost, staging, and production.
- `PageResolver` (`resolve_page_urls`) is URL-only; it does not verify that
  image files exist on disk. A missing file returns HTTP 404 from the
  StaticFiles mount.
- The StaticFiles mount is configured at startup:
  `app.mount("/pages", StaticFiles(directory="storage/pages"), name="pages")`

---

## 7. Verification Layer Contract

`verify_answer()` in `backend/llm/llm.py` is the **final factual guardrail**
before the response is returned to the caller.

```
draft_answer  ──→  verify_answer(draft, context, mode, model)  ──→  verified_answer
```

- Called with `temperature=0.0` (deterministic).
- Instruction: keep supported claims, rewrite partial claims, remove
  unsupported claims. **Never invent** new page numbers.
- Bypassed when the request sets `disable_verifier=true`; otherwise the
  verifier retries quota failures using the Gemini key pool.
- A complete verified answer is cacheable. Draft fallbacks, disabled
  verification, and truncated responses are not persisted in the semantic cache.

---

*Last updated: 2026-08-21 | Maintainer: HarrisonGPT AI Governance*
