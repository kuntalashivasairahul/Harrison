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

1. `optimize_query()` classifies the query, rejects non-medical requests, expands the search query, and labels complexity.
2. The API embeds the search query with `BAAI/bge-m3` and checks the semantic cache against exact runtime metadata.
3. A cache miss runs hybrid FAISS/BM25 retrieval, RRF fusion (`RRF_K=60`), neighbor expansion, cross-encoder reranking, and the `RERANK_SCORE_THRESHOLD=-3.0` hard filter.
4. `route_and_sort_context()` deduplicates and page-orders chunks; `fuse_context()` constructs the prompt context within its fixed 12,000-character limit.
5. Evidence and source labels are extracted from the retrieved chunks.
6. `ask_llm()` calls Gemini through `google-genai`. `KeyManager` uses `GEMINI_API_KEY_1` through `_10` (or `GEMINI_API_KEY` as slot 1), rotates clients round-robin, and marks quota-exhausted keys for the process lifetime.
7. Unless `disable_verifier` is requested, `verify_answer()` performs a grounded Gemini rewrite at temperature `0.0`. A complete verified response is the only response eligible for semantic-cache persistence.
8. `backend/agents/confidence_scorer.py` scores the final response from average cross-encoder relevance and draft-to-verified length divergence; return-path and truncation caps are then applied by the API.
9. Source labels are resolved into page-image URLs and returned with timing data in `QueryResponse`.

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
│   │   │                                expand_query(), _hybrid_candidates(),
│   │   │                                _pretrim_for_rerank(), retrieve()
│   │   │                                Uses RERANK_SCORE_THRESHOLD from config.py
│   │   ├── rerank.py                  ← CrossEncoder wrapper: rerank(), top_score()
│   │   └── embeddings.py              ← embed_text() — FAISS query embedding
│   │
│   ├── llm/
│   │   └── llm.py                     ← ask_llm(), verify_answer()
│   │                                    Gemini key management, prompts, and verification
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
│   ├── config.py                      ← Global path & model constants
│   ├── requirements.txt               ← Python dependencies
│   └── .env                           ← Secrets (git-ignored)
│
├── artifacts/                         ← Runtime data — DO NOT SCAN
│   ├── vectorstore/
│   │   ├── index.faiss                ← FAISS index (binary)
│   │   └── chunks.json                ← Chunk metadata (text, page, chunk_id)
│   └── retrieval_logs/                ← Per-query JSON diagnostics (auto-written)
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
    ├── llm/llm.py                (ask_llm, REFUSAL_STR)
    ├── retrieval/rerank.py       (top_score)
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
| No retrieved chunks, unusable scores, or average score `< -2.0` | **Low**  |
| Draft-to-verified length divergence `> 0.40`                    | **Low**  |
| Average score `>= -0.5` and at least two chunks                 | **High** |
| Everything else                                                 | **Medium** |

### 5.3 Hard Filter Threshold

```python
RERANK_SCORE_THRESHOLD = -3.0  # raw ms-marco-MiniLM logit
```

Chunks scoring below this value are dropped **after** reranking but **before**
context fusion. It is defined in `backend/config.py` and used by
`backend/retrieval/rag.py`; it is not configurable at the API layer.

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

*Last updated: 2026-05-30 | Maintainer: HarrisonGPT AI Governance*
