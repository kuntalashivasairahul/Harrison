# ARCHITECTURE.md
# HarrisonGPT — System Architecture Reference

> **Immutability Notice.**
> The pipeline stages documented below are **core, non-negotiable components**.
> AI tools, future contributors, and automated refactoring agents **must not**
> remove, reorder, or bypass any stage — especially the hard filter,
> `verify_answer()`, and `resolve_page_urls()` steps.

---

## 1. End-to-End Pipeline

The complete request lifecycle for a single `/ask` call:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          POST /ask  (FastAPI)                            │
│                    { query: str, mode: "qa"|"smart_summary" }            │
└─────────────────────────────────┬────────────────────────────────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │      1. Query Expansion      │
                    │  expand_query(max_queries=4) │
                    │  → always includes original  │
                    │  → up to 3 rule-based paraphrases│
                    └─────────────┬──────────────┘
                                  │  List[str]  (≤ 4 queries)
                    ┌─────────────▼──────────────┐
                    │   2. Hybrid Retrieval (×N)  │
                    │   FAISS (dense ANN, k=30)   │
                    │      +                      │
                    │   BM25Okapi (lexical, k=30) │
                    │   — per expanded query —    │
                    └─────────────┬──────────────┘
                                  │  List[Dict]  (raw candidates)
                    ┌─────────────▼──────────────┐
                    │   3. RRF Fusion  (K = 60)   │
                    │  Merge + deduplicate by     │
                    │  chunk_id across all queries │
                    │  RRF score = Σ 1/(K+rank)   │
                    └─────────────┬──────────────┘
                                  │  merged List[Dict]
                    ┌─────────────▼──────────────┐
                    │  4. Low-Value Chunk Filter  │
                    │  Drop: figure captions,     │
                    │  references, <20 char lines │
                    └─────────────┬──────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │  5. Neighbor Chunk Expansion│
                    │  For each top chunk, add    │
                    │  chunk_id ± 1 (if not duped)│
                    │  Cap at rerank_pool size    │
                    └─────────────┬──────────────┘
                                  │  rerank pool (≤ 24 chunks)
                    ┌─────────────▼──────────────┐
                    │  6. Cross-Encoder Reranking │
                    │  ms-marco-MiniLM-L-6-v2     │
                    │  Scores: raw logits          │
                    │  top_n = final_k (6 or 12)  │
                    └─────────────┬──────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │  7. Hard Score Filter ⚠️    │
                    │  RERANK_SCORE_THRESHOLD=-2.0│
                    │  Drop chunks below threshold │
                    │  → IMMUTABLE SAFETY GATE ←  │
                    └─────────────┬──────────────┘
                                  │  final_chunks: List[Dict]
                    ┌─────────────▼──────────────┐
                    │   8. Context Fusion         │
                    │   fuse_context(chunks)      │
                    │   Assembles fused_context   │
                    │   string for LLM prompt     │
                    └─────────────┬──────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │  9. Evidence Extraction     │
                    │  extract_evidence(chunks)   │
                    │  → EVIDENCE: <stmt> [p:NNN] │
                    │  extract_sources(chunks)    │
                    │  → ["p.142", "p.512", ...]  │
                    └─────────────┬──────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │  10. Page URL Resolution    │
                    │  resolve_page_urls(sources) │
                    │  → thumbnail_url + full_url │
                    │  → IMMUTABLE: visual ground │
                    └─────────────┬──────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │  11. LLM Generation (Groq)  │
                    │  ask_llm(context, question, │
                    │          mode, evidence)    │
                    │  model: llama-3.3-70b       │
                    │  temp: 0.1 (ss) / 0.2 (qa) │
                    └─────────────┬──────────────┘
                                  │  draft_answer: str
                    ┌─────────────▼──────────────┐
                    │  12. Conditional Verify ⚠️  │
                    │  verify_answer() — ALWAYS   │
                    │  runs unless REFUSAL_STR    │
                    │  temp=0.0, grounded rewrite │
                    │  → IMMUTABLE SAFETY GATE ←  │
                    └─────────────┬──────────────┘
                                  │  verified_answer: str
                    ┌─────────────▼──────────────┐
                    │  13. Confidence Scoring     │
                    │  calculate_confidence(      │
                    │    top_reranker_score,      │
                    │    evidence_count,          │
                    │    was_verified)            │
                    │  → "High" | "Medium" | "Low"│
                    └─────────────┬──────────────┘
                                  │
                    ┌─────────────▼──────────────┐
                    │   14. Final JSON Response   │
                    │   QueryResponse schema:     │
                    │   { answer, confidence,     │
                    │     sources, visual_context }│
                    └────────────────────────────┘
```

---

## 2. FastAPI Schema — Immutable Contract

The `QueryResponse` schema is a **frozen contract**. Fields must not be
renamed, removed, or re-typed without a full migration plan.

```python
class QueryRequest(BaseModel):
    query: str
    mode: Literal["qa", "smart_summary"] = "smart_summary"

class QueryResponse(BaseModel):
    answer:         str                   # Verified LLM answer
    confidence:     str                   # "High" | "Medium" | "Low"
    sources:        List[str]             # e.g. ["p.142", "p.512"]
    visual_context: List[Dict[str, str]]  # [{page_label, thumbnail_url, full_url}]
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
│   │   │                                RERANK_SCORE_THRESHOLD defined here
│   │   ├── rerank.py                  ← CrossEncoder wrapper: rerank(), top_score()
│   │   └── embeddings.py              ← embed_text() — FAISS query embedding
│   │
│   ├── llm/
│   │   └── llm.py                     ← ask_llm(), verify_answer()
│   │                                    BASE_QA_PROMPT, SMART_SUMMARY_PROMPT defined here
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
│   ├── utils/
│   │   ├── fusion.py                  ← fuse_context(), clean_text()
│   │   └── scoring.py                 ← calculate_confidence()
│   │                                    Pure function; no I/O
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
    ├── utils/scoring.py          (calculate_confidence)
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
    └── groq                      (Groq client)

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
| `was_verified == False`                                        | **Low**    |
| `top_reranker_score < 1.0`                                     | **Low**    |
| `top_reranker_score ≥ 5.0 AND evidence_count ≥ 2`             | **High**   |
| Everything else                                                | **Medium** |

### 5.3 Hard Filter Threshold

```python
RERANK_SCORE_THRESHOLD = -2.0  # raw ms-marco-MiniLM logit
```

Chunks scoring below this value are dropped **after** reranking but **before**
context fusion. This is not configurable at the API layer — it is a safety
constant in `backend/retrieval/rag.py`.

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
- Skipped **only** when `ask_llm()` returns `REFUSAL_STR` or an LLM error
  sentinel — both of which are already safe non-answers.
- The `was_verified` flag in `calculate_confidence()` is set to `True` when
  the answer is neither `REFUSAL_STR` nor an `"LLM call failed:"` prefix.

---

*Last updated: 2026-05-30 | Maintainer: HarrisonGPT AI Governance*
