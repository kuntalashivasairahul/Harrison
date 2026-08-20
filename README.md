# HarrisonGPT

HarrisonGPT is a production-grade, high-recall **Medical Retrieval-Augmented Generation (RAG) System** grounded exclusively in *Harrison's Principles of Internal Medicine* (20th+ Edition). Designed for clinicians, medical students, and exam preparation, the system prioritizes factual fidelity and citation grounding over raw creativity.

---

## 🚀 Key Features

* **Stage-Aware LLM Routing:** Uses an approved local registry: Groq optimizes queries when explicitly enabled, Gemini remains the only draft and verification provider, and all provider fallback decisions are logged.
* **Low-Latency Semantic Caching:** Utilizes a disk-persistent semantic cache (`artifacts/semantic_cache.json`) to serve clinically equivalent queries instantly ($\ge 0.95$ Cosine Similarity) in under ~1ms.
* **Hybrid Retrieval Pipeline:** Merges FAISS dense search using `BAAI/bge-m3` (1024 dimensions) with BM25Okapi sparse lexical search via Reciprocal Rank Fusion (RRF), alongside local context neighbor chunk expansion.
* **Cross-Encoder Rerank Filtering:** Scores chunks via an `ms-marco-MiniLM-L-6-v2` cross-encoder, filtering out noisy passages scoring below `-3.0`.
* **Double-Pass Grounding & Verification:** Employs a post-hoc self-consistency check (`verify_answer`) to compare draft answers against raw context source page boundaries, rephrasing or redacting any claims not fully supported by the textbook.
* **Visual Grounding & Image URL Resolution:** Resolves textbook citations (e.g., `p.2787`) to static thumbnail WebP or full-resolution PNG page image URLs hosted on the local static server.

---

## 🛠️ Technology Stack

* **API & Serving:** FastAPI, Uvicorn, Pydantic, Python-dotenv
* **Vector & Lexical Search:** FAISS (CPU), Rank-BM25
* **AI Embeddings & Reranking:** SentenceTransformers (`BAAI/bge-m3`, 1024 dimensions, and `ms-marco-MiniLM-L-6-v2`)
* **Large Language Models:** Google Gen AI SDK (`google-genai`) for grounded answers and Groq for the optional query-optimizer route.

---

## 🗺️ System Control Flow & Architecture

For a detailed walkthrough, step-by-step trace, and sequence diagram of how control flows from request intake to response delivery, refer to [workflow.md](workflow.md).

---

## 💻 Setup & Installation

### 1. Prerequisites
* Python 3.12
* Virtual Environment manager (venv)
* [git-lfs](https://git-lfs.com) — the FAISS index and chunk registry are
  tracked in LFS. Install it **before** cloning:
  ```bash
  brew install git-lfs && git lfs install   # macOS
  ```
  Without it, `artifacts/vectorstore/*` checks out as small text pointer files
  and the API starts in a degraded state.

### 2. Environment Setup
Clone the repository. The checked-in production vectorstore under
`artifacts/vectorstore/` is sufficient to serve queries immediately; it
contains a `66 MB` FAISS index and `32 MB` matching chunk registry. Do not
separate these two files.

On macOS or Linux, initialize the Python virtual environment:
```bash
./scripts/setup_env.sh
# Or, after setup:
.venv312/bin/pip install -r backend/requirements.txt -r backend/requirements-dev.txt
```

`.venv312` (Python 3.12) is the **only** supported runtime — it runs the server,
the test suite, and the diagnostic scripts. Dependencies are pinned in
`backend/requirements.txt`; test and lint tools in `backend/requirements-dev.txt`.

On Windows PowerShell:
```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_env.ps1
```

### 3. Environment Variables Config
Create a `.env` file inside the `backend/` directory:
```env
GEMINI_API_KEY="your-google-ai-api-key"

# Optional rotation pool; the main key and numbered keys are all distinct.
# GEMINI_API_KEY_1="..."
# GEMINI_API_KEY_2="..."
# ... through GEMINI_API_KEY_10

# Optional customizations
SMART_SUMMARY_MAX_TOKENS=3000
QA_MAX_TOKENS=3000
SMART_SUMMARY_CONTEXT_CHAR_LIMIT=12000
SMART_SUMMARY_K=48
SMART_SUMMARY_FINAL_K=12
SMART_SUMMARY_RERANK_POOL=16
# Temporary cooldown applied to a Gemini key after a 429 response.
GEMINI_RATE_LIMIT_COOLDOWN_SECONDS=60

# Stage 1 Groq optimizer. Disabled unless both variables are set.
GROQ_ENABLED=false
GROQ_API_KEY=""
GROQ_OPTIMIZER_MODEL=openai/gpt-oss-20b
LLM_OPTIMIZER_DEADLINE_SECONDS=8
LLM_DRAFT_DEADLINE_SECONDS=30
LLM_VERIFIER_DEADLINE_SECONDS=30
```

---

## ⚙️ Running Locally

Start the ASGI development server from the workspace root:
```bash
.venv312/bin/python -m uvicorn backend.api.main:app --reload --host 127.0.0.1 --port 8000
```

On Windows PowerShell:
```powershell
.\.venv312\Scripts\python.exe -m uvicorn backend.api.main:app --reload --host 127.0.0.1 --port 8000
```
The server will start at `http://127.0.0.1:8000`.

### Portable Assets

The production retrieval index is committed so a normal `git clone` can answer
questions without rebuilding embeddings. It is tracked in **git-lfs**
(see `.gitattributes`), so rebuilding the index no longer adds its full size to
git history on every commit. Note that the copies committed before LFS was
adopted remain as ordinary blobs in history — this stops future growth, it does
not shrink the past. `backend/.env`, semantic-cache data,
staging indexes, backups, rendered page images, and the original PDF are
intentionally excluded. Copy your own keys into `backend/.env` from
`.env.example` on each machine. The API still runs without the page-image
archive; only `/pages/...` visual links will be empty. See
[`artifacts/vectorstore/README.md`](artifacts/vectorstore/README.md) for the
index checksums and deliberate rebuild command.

---

## 🔌 API Documentation

### 1. Ask Endpoint
* **Path:** `/ask`
* **Method:** `POST`
* **Request JSON Schema:**
  ```json
  {
    "query": "management of acute pancreatitis",
    "mode": "smart_summary"
  }
  ```
  *(Modes: `smart_summary` (default) or `qa`)*

* **Response JSON Schema:**
  ```json
  {
    "answer": "Topic received — generating Harrison Smart Summary...",
    "confidence": "High",
    "sources": ["p.2157", "p.2158"],
    "visual_context": [
      {
        "page_label": "p.2157",
        "thumbnail_url": "http://127.0.0.1:8000/pages/small/page_2114_small.webp",
        "full_url": "http://127.0.0.1:8000/pages/full/page_2114_full.png"
      }
    ]
  }
  ```

### 2. Health Endpoint
* **Path:** `/health`
* **Method:** `GET`
* **Response JSON Schema:**
  ```json
  {
    "status": "ok",
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

---

## 🧪 Evaluation

### Test suite

```bash
.venv312/bin/python -m pytest
```

The suite is hermetic — no network, no model weights, no FAISS index — and runs
in a few seconds. Collection is scoped to `tests/` by `pyproject.toml`; the
programs under `scripts/` are named `probe_*.py` because they are interactive
diagnostics, not tests.

### Retrieval diagnostics

```bash
.venv312/bin/python scripts/probe_retrieval.py --query "Ranson criteria pancreatitis"
.venv312/bin/python scripts/probe_retrieval.py --staging      # staging vectorstore
```

### Evaluation harness

The `evaluation/` directory contains tools and test configurations (e.g. `test_queries.json`) to validate the RAG pipeline recall, accuracy, and latency metrics. Use the project Python runtime for custom scripts:
```bash
.venv312/bin/python -m evaluation.run_eval
```

### Smart Summary Configuration

For `mode: "smart_summary"`, `SMART_SUMMARY_K` (default `48`) controls the
candidate retrieval count and `SMART_SUMMARY_RERANK_POOL` (default `16`)
controls the rerank pool. Query complexity selects a final context count of
`5` (simple) or `12` (complex), capped by `SMART_SUMMARY_FINAL_K` (default
`12`). `SMART_SUMMARY_MAX_TOKENS` controls the generation and normal
verification token ceiling for this mode. The first response line is enforced
as `Topic received — generating Harrison Smart Summary.`; headings are
generated from available evidence rather than padded with empty sections.

`SMART_SUMMARY_CONTEXT_CHAR_LIMIT` (default `12000`) sets the fused-context
character budget applied by `backend/utils/fusion.py` to every mode. Chunks are
selected against this budget in descending cross-encoder score and then emitted
in page order, so raising or lowering it changes how much context survives, not
which chunks are preferred.

`SMART_SUMMARY_MAX_TOKENS` is additionally bounded by `max_output_tokens` for
the `gemini-primary` deployment in `backend/llm/model_registry.json` (currently
`3000`). Requesting more than the registry allows is clamped, and the clamp is
logged at WARNING; lift the registry value to raise the real ceiling.

### Stage 1 Provider Policy

`backend/llm/model_registry.json` is the only approved-provider allowlist.
Groq is used only for query optimization, then Gemini Flash-Lite is attempted,
then the optimizer returns its deterministic local fallback. Gemini remains the
sole draft and verifier provider. OpenRouter, OmniRoute, Cerebras, Mistral,
NVIDIA NIM, and OpenCode are not enabled in Stage 1. Do not route textbook
context through generic gateway auto-routing or context compression.
