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
- **Lexical index**: BM25Okapi (rank-bm25) built in-memory at startup from
  `artifacts/vectorstore/chunks.json`.
- **Serving**: FastAPI + Uvicorn, live at `http://127.0.0.1:8000`.
- **Visual grounding**: Pre-rendered Harrison page images served as static
  files from `storage/pages/` via the `/pages` StaticFiles mount.
- **Confidence scoring**: Fully operational (High / Medium / Low) based on
  cross-encoder scores, evidence count, and verification status.
- **Verification layer**: `verify_answer()` runs unconditionally on every
  non-refusal LLM response.

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
| `sentence-transformers`| Embedding (`all-MiniLM-L6-v2`) + Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) | `backend/retrieval/embeddings.py`, `backend/retrieval/rerank.py` |

### 4.3 Language Model

| Library  | Role                                                  | Key File                |
|----------|-------------------------------------------------------|-------------------------|
| `groq`   | LLM inference via Groq Cloud (`llama-3.3-70b-versatile`) | `backend/llm/llm.py` |

### 4.4 Text Processing & Utilities

| Library / Module            | Role                                               | Key File                              |
|-----------------------------|----------------------------------------------------|---------------------------------------|
| `backend/utils/fusion.py`   | Context window construction (`fuse_context`)       | `backend/utils/fusion.py`             |
| `backend/utils/scoring.py`  | Confidence label calculation (`calculate_confidence`) | `backend/utils/scoring.py`         |
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

These values are defined in `backend/config.py` and `backend/retrieval/rag.py`.
They must **not** be changed without updating this document.

```python
# backend/config.py
EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL     = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_K        = 30       # FAISS/BM25 candidates per query
DEFAULT_FINAL_K  = 6        # Final chunks passed to LLM
DEFAULT_RERANK_POOL = 24    # Pool size fed to cross-encoder
RRF_K            = 60       # Reciprocal Rank Fusion K hyperparameter

# backend/retrieval/rag.py
RERANK_SCORE_THRESHOLD = -2.0   # Hard filter: drop chunks below this logit
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

---

## 7. Environment Variables

All secrets are stored in `backend/.env` (git-ignored).

| Variable                       | Default  | Purpose                                       |
|--------------------------------|----------|-----------------------------------------------|
| `GROQ_API_KEY`                 | —        | Groq Cloud inference key (**required**)       |
| `SMART_SUMMARY_MAX_TOKENS`     | `2200`   | Max tokens for smart_summary LLM call         |
| `QA_MAX_TOKENS`                | `900`    | Max tokens for qa LLM call                    |
| `SMART_SUMMARY_CONTEXT_CHAR_LIMIT` | `18000` | Character cap applied to fused context     |
| `SMART_SUMMARY_K`              | `48`     | Candidate retrieval K in smart_summary mode   |
| `SMART_SUMMARY_FINAL_K`        | `12`     | Final-K in smart_summary mode                 |
| `SMART_SUMMARY_RERANK_POOL`    | `16`     | Rerank pool in smart_summary mode             |

---

## 8. Health & Observability

The `/health` endpoint reports three liveness signals:

```json
{
  "status": "ok | degraded",
  "faiss_loaded": true,
  "chunks_loaded": true,
  "groq_key_present": true
}
```

A `degraded` status means at least one of: FAISS index missing, chunks JSON
empty, or `GROQ_API_KEY` not set.

---

*Last updated: 2026-05-30 | Maintainer: HarrisonGPT AI Governance*
