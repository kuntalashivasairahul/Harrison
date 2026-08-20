# PROJECT_CONTEXT.md
# HarrisonGPT — Definitive Project Context

> **Machine-readable governance document.**
> This file is the canonical source of truth for all AI-assisted development,
> onboarding, and architectural decisions on the HarrisonGPT repository.
> Do **not** modify this file without updating `ARCHITECTURE.md` and
> `CODING_RULES.md` to stay in sync.

---

## 1. Project Identity

| Field              | Value                                                                 |
|--------------------|-----------------------------------------------------------------------|
| **Project Name**   | HarrisonGPT                                                           |
| **Type**           | Production-grade Medical RAG (Retrieval-Augmented Generation) System |
| **Source Corpus**  | Harrison's Principles of Internal Medicine (20th+ Edition)            |
| **Primary Mode**   | `smart_summary` — high-recall, Perplexity-style medical search        |
| **Secondary Mode** | `qa` — focused question-answering over Harrison text                  |

---

## 2. Primary Goal

**High-recall, Perplexity-style medical search that prioritizes accuracy and
citation grounding over raw creativity.**

- Every claim in a generated response **must** be traceable to a page in
  Harrison's textbook via `[p:NNNN]` citation markers.
- The system is designed for clinicians, medical students, and exam preparation.
  Hallucination is a patient-safety risk — not merely an accuracy problem.
- Creativity, fluency, and formatting are **secondary** to factual fidelity.

---

## 3. Current Operational State

- **Hardware target**: Apple Silicon (M4) — fully native, no CUDA dependency.
- **Vector index**: FAISS (CPU) loaded from `artifacts/vectorstore/index.faiss`.
- **Lexical index**: BM25Okapi (rank-bm25) built in-memory from
  `artifacts/vectorstore/chunks.json`, **lazily on first use** and forced during
  the FastAPI `lifespan` warm-up. Tokenization is punctuation-aware.
- **Serving**: FastAPI + Uvicorn, live at `http://127.0.0.1:8000`.
- **Visual grounding**: Pre-rendered Harrison page images served as static
  files from `storage/pages/` via the `/pages` StaticFiles mount.
- **Confidence scoring**: Fully operational (High / Medium / Low) from average
  cross-encoder score, draft-to-verified divergence, and the count of chunks
  with no usable score. Capped by the return path and by truncation in the API.
- **Verification layer**: `verify_answer()` runs on every non-refusal response
  unless the request sets `disable_verifier`. Thinking is disabled for this
  stage — Gemini 2.5 spends `max_output_tokens` on an internal reasoning pass,
  which used to truncate the verifier and force a `draft_fallback` on every
  smart summary.
- **Observability**: `backend/logging_config.py` makes `backend.*` logs visible
  under uvicorn; every request carries an `X-Request-ID`; `/metrics` exposes
  counters and p50/p95 per stage.
- **Test suite**: 252 hermetic tests, ~4s, on `.venv312`. No network, no model
  weights, no index. There is no integration tier yet — that is the largest
  outstanding gap in the project.

---

## 4. Core Technology Stack

### 4.1 API & Serving

| Library       | Role                                      | Key File                        |
|---------------|-------------------------------------------|---------------------------------|
| `fastapi`     | HTTP API framework, schema validation     | `backend/api/main.py`           |
| `uvicorn`     | ASGI server                               | runtime                         |
| `pydantic`    | Request/Response schema enforcement       | `backend/api/main.py`           |
| `python-dotenv` | Secrets and environment config          | `backend/.env`                  |

### 4.2 Retrieval & Indexing

| Library                | Role                                     | Key File                        |
|------------------------|------------------------------------------|---------------------------------|
| `faiss-cpu`            | Dense vector search (ANN)                | `backend/retrieval/rag.py`      |
| `rank-bm25`            | Sparse lexical search (BM25Okapi)        | `backend/retrieval/rag.py`      |
| `sentence-transformers`| Embedding (`BAAI/bge-m3`, 1024 dimensions) + Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) | `backend/retrieval/embeddings.py`, `backend/retrieval/rerank.py` |

### 4.3 Language Model

| Library  | Role                                                  | Key File                |
|----------|-------------------------------------------------------|-------------------------|
| `google-genai` | Gemini inference, dynamic model selection, and rotating key clients | `backend/llm/llm.py` |
| `groq` | Optional Stage 1 query optimizer provider | `backend/llm/groq_provider.py` |

Stage 1 routes query optimization through Groq only when `GROQ_ENABLED=true`
and a valid `GROQ_API_KEY` is configured. Gemini remains the only draft and
verification provider. Provider eligibility is restricted to
`backend/llm/model_registry.json`; gateways and additional providers are not
enabled until a later evaluated stage.

### 4.4 Text Processing & Utilities

| Library / Module            | Role                                               | Key File                              |
|-----------------------------|----------------------------------------------------|---------------------------------------|
| `backend/utils/fusion.py`   | Context window construction (`fuse_context`)       | `backend/utils/fusion.py`             |
| `backend/agents/confidence_scorer.py` | Confidence label calculation (`calculate_confidence`) | `backend/agents/confidence_scorer.py` |
| `backend/processing/evidence.py` | Evidence extraction & source deduplication  | `backend/processing/evidence.py`      |

### 4.5 Page Rendering & Visual Grounding

| Library / Module                      | Role                                             | Key File                                |
|---------------------------------------|--------------------------------------------------|-----------------------------------------|
| `PyMuPDF` (fitz)                      | PDF page rendering to PNG/WebP (pre-processing)  | offline pre-processing step             |
| `backend/rendering/page_resolver.py`  | Maps page labels → static image URLs             | `backend/rendering/page_resolver.py`    |
| `storage/pages/small/`                | Thumbnail WebP images (`page_NNN_small.webp`)    | `storage/pages/small/`                  |
| `storage/pages/full/`                 | Full-resolution PNG images (`page_NNN_full.png`) | `storage/pages/full/`                   |

---

## 5. Key Runtime Constants

These values are defined in `backend/config.py`.
They must **not** be changed without updating this document.

```python
# backend/config.py
EMBEDDING_MODEL  = "BAAI/bge-m3"
EMBEDDING_DIM    = 1024
RERANK_MODEL     = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_K        = 30       # FAISS/BM25 candidates per query
# final_k is chosen per request by _final_k_for(): 5 for a simple query,
# 12 for a complex one, capped by SMART_SUMMARY_FINAL_K in smart_summary mode.
DEFAULT_RERANK_POOL = 24    # Pool size fed to cross-encoder
RRF_K            = 60       # Reciprocal Rank Fusion K hyperparameter

RERANK_SCORE_THRESHOLD = -3.0   # Hard filter: drop chunks below this logit

# LLM deadlines — read from the environment HERE and imported by call sites.
# Never re-read these with os.getenv() at a call site: that is how config.py
# and the live values drifted apart previously.
LLM_OPTIMIZER_DEADLINE_SECONDS = 8.0
LLM_DRAFT_DEADLINE_SECONDS     = 60.0
LLM_VERIFIER_DEADLINE_SECONDS  = 60.0
LLM_PROVIDER_COOLDOWN_SECONDS  = 60.0
```

---

## 6. Artifact & Data Boundaries

| Path                          | Contents                                          | AI Scan Allowed? |
|-------------------------------|---------------------------------------------------|------------------|
| `artifacts/vectorstore/`      | FAISS index, chunks JSON                          | **NO**           |
| `artifacts/retrieval_logs/`   | Per-query retrieval diagnostics (JSON)            | **NO**           |
| `storage/pages/`              | Pre-rendered Harrison page images                 | **NO**           |
| `backend/`                    | All Python source modules                         | YES              |
| `evaluation/`                 | Evaluation scripts and metrics                    | YES              |
| `tests/`                      | Hermetic test suite                               | YES              |
| `scripts/`                    | Operational tools and `probe_*` diagnostics       | YES              |
| `docs/archive/`               | Superseded analysis reports — **stale**           | NO               |

Reading an artifact file to *measure* a specific defect is legitimate; letting
it into general context is not.

---

## 7. Environment Variables

All secrets are stored in `backend/.env` (git-ignored).

| Variable | Default | Purpose |
|---|---|---|
| `GEMINI_API_KEY`, `GEMINI_API_KEY_1` … `_10` | — | Main plus optional numbered Gemini key pool. Round-robin; a 429/quota key enters a temporary cooldown. |
| `GEMINI_RATE_LIMIT_COOLDOWN_SECONDS` | `60` | Cooldown applied to a key after a 429/quota response. |
| `GROQ_ENABLED` | `false` | Enables the Stage 1 Groq optimizer. Requires `GROQ_API_KEY` too. |
| `GROQ_API_KEY` | — | Groq key. Optimizer stage only — never draft or verification. |
| `GROQ_OPTIMIZER_MODEL` | `openai/gpt-oss-20b` | **Overrides `model_registry.json`.** Changing the registry alone does nothing while this is set. |
| `GROQ_REASONING_EFFORT` | `low` | Reasoning budget for reasoning-capable Groq models. |
| `SMART_SUMMARY_MAX_TOKENS` | `8000` | Generation/verification ceiling for `smart_summary`. Bounded by `max_output_tokens` in the registry (8192); exceeding it is clamped and logged. |
| `QA_MAX_TOKENS` | `3000` | Same ceiling for `qa`. |
| `SMART_SUMMARY_CONTEXT_CHAR_LIMIT` | `12000` | Fused-context character budget, applied by `utils/fusion.py` to every mode. |
| `SMART_SUMMARY_K` | `48` | Candidate retrieval K in `smart_summary` mode. |
| `SMART_SUMMARY_FINAL_K` | `12` | Cap on the complexity-driven final-K in `smart_summary` mode. |
| `SMART_SUMMARY_RERANK_POOL` | `16` | Rerank pool in `smart_summary` mode. |
| `LLM_OPTIMIZER_DEADLINE_SECONDS` | `8` | Optimizer request deadline. |
| `LLM_DRAFT_DEADLINE_SECONDS` | `60` | Draft request deadline. |
| `LLM_VERIFIER_DEADLINE_SECONDS` | `60` | Verifier request deadline. |
| `LLM_PROVIDER_COOLDOWN_SECONDS` | `60` | Non-Gemini deployment cooldown after a rate limit. |
| `LLM_DRAFT_MAX_ATTEMPTS` | `3` | Draft retry budget. |
| `LLM_VERIFIER_MAX_ATTEMPTS` | `2` | Verifier retry budget. |
| `HARRISON_ADMIN_TOKEN` | — | Required by `X-Admin-Token` on `/admin/*`. **Unset closes the admin surface (503), it does not open it.** |
| `HARRISON_MAX_QUERY_CHARS` | `2000` | Upper bound on `query`. |
| `HARRISON_RATE_LIMIT_PER_MINUTE` | `30` | Per-client `/ask` limit. `0` disables. |
| `HARRISON_CORS_ORIGINS` | — | Comma-separated allowed origins. Empty means no cross-origin access. |
| `HARRISON_LOG_LEVEL` | `INFO` | Level for the `backend` logger. |
| `HARRISON_RETRIEVAL_LOGS` | `false` | Enables per-query retrieval diagnostics on disk. |
| `HARRISON_RETRIEVAL_LOG_RETENTION` | `200` | Files kept when retrieval logging is on. |
| `HARRISON_RETRIEVAL_LOG_QUERIES` | `false` | Whether the raw clinical query is written to those files. Off by default. |

---

## 8. Health & Observability

The `/health` endpoint reports index, embedding, and Gemini-key readiness:

```json
{
  "status": "ok | degraded",
  "faiss_loaded": true,
  "chunks_loaded": true,
  "faiss_dim": 1024,
  "embedding_dim": 1024,
  "embedding_index_dim_match": true,
  "gemini_key_present": true,
  "gemini_key_count": 1,
  "gemini_available_key_count": 1,
  "llm_providers": []
}
```

`/health` returns **HTTP 503** when degraded and 200 when ok, so an
orchestrator can act on it. `/metrics` reports pipeline counters and p50/p95
per stage for the current process.

A `degraded` status means at least one of: FAISS index missing, chunks JSON
empty, no Gemini key is available, or the active embedding model dimension
does not match the loaded FAISS index.

---

## 9. Smart Summary Runtime Behavior

`smart_summary` uses the mode-specific retrieval values above. The query
optimizer returns `simple` or `complex`; the API chooses final-K `5` or `12`,
respectively, then caps it at `SMART_SUMMARY_FINAL_K`. The model output is
forced to begin with `Topic received — generating Harrison Smart Summary.`.
Sections are generated only when supported by available content; the runtime
does not pad omitted sections with placeholder text.

Context fusion has a fixed 12,000-character `SAFE_CHAR_LIMIT` for both modes.
`SMART_SUMMARY_CONTEXT_CHAR_LIMIT` is therefore informational in the current
implementation, not a live override of that fusion budget.

---

## 10. Local Runtime

Create or refresh the supported Python 3.12 environment with:

```bash
./scripts/setup_env.sh
.venv312/bin/python -m uvicorn backend.api.main:app --reload --host 127.0.0.1 --port 8000
```

---

*Last updated: 2026-08-21 | Maintainer: HarrisonGPT AI Governance*
