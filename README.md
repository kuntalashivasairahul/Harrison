# HarrisonGPT

HarrisonGPT is a production-grade, high-recall **Medical Retrieval-Augmented Generation (RAG) System** grounded exclusively in *Harrison's Principles of Internal Medicine* (20th+ Edition). Designed for clinicians, medical students, and exam preparation, the system prioritizes factual fidelity and citation grounding over raw creativity.

---

## 🚀 Key Features

* **Stage-Aware LLM Routing:** Uses an approved local registry: Groq optimizes queries when explicitly enabled, Gemini drafts and is the **only** verification provider, Mistral and Groq back the draft stage when Gemini is rate-limited, and all provider fallback decisions are logged.
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
* **Large Language Models:** Google Gen AI SDK (`google-genai`) for grounded answers and verification; Groq for the optional query-optimizer route; Mistral over stdlib `urllib` for draft failover.

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
SMART_SUMMARY_MAX_TOKENS=8000
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

# Mistral draft failover. Disabled unless both variables are set.
MISTRAL_ENABLED=false
MISTRAL_API_KEY=""
MISTRAL_DRAFT_MODEL=mistral-large-latest

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
Groq is used first for query optimization, then Gemini Flash-Lite is attempted,
then the optimizer returns its deterministic local fallback.

The **draft** stage falls over in priority order:

| Priority | Deployment | Provider | Reached when |
|---|---|---|---|
| 10 | `gemini-primary` | Gemini | always tried first |
| 20 | `gemini-draft-fallback` | Gemini | primary returned a fallback-eligible error |
| 25 | `mistral-draft` | Mistral | both Gemini drafts failed |
| 30 | `groq-draft` | Groq | Mistral also failed, **and** the prompt fits 5k input tokens |

**Gemini remains the sole verifier.** A Mistral or Groq draft is still verified
by Gemini against the same retrieved context, so the grounding and citation
checks in `verify_answer()` apply unchanged; the failover widens who may write a
first pass, never who approves it. A non-Gemini entry in the registry carrying
the `verifier` stage is a test failure, not a review comment.

Mistral outranks Groq deliberately. Groq's 8k-token-per-minute ceiling means
`groq-draft` cannot carry a full `smart_summary` prompt at all, so ordering it
first would have it decline exactly the requests the failover exists for.

**Groq's free tier allows 8000 tokens per minute on every model it serves**
(`gpt-oss-120b`, `gpt-oss-20b`, `qwen3.6-27b` — verified against the
`x-ratelimit-limit-tokens` response header). `groq-draft` is therefore capped at
`max_input_tokens: 5000` / `max_output_tokens: 2500`, well under a full
`smart_summary` draft (~11.5k tokens), so it realistically only rescues shorter
`qa` queries. Over-budget prompts are rejected by the router before the call
rather than returning HTTP 413. `groq/compound` is the only model with real
headroom (70k TPM) and is **not** eligible: it performs its own tool use and web
search, which would put ungrounded claims into a medical answer.

**Mistral's free tier is where the real headroom is** — 1 request/second,
500,000 tokens/minute, 1 billion tokens/month. That is ~60× Groq's per-minute
allowance, so `mistral-draft` carries a full `smart_summary` prompt (12,000
input / 8,192 output tokens, matching `gemini-primary`) rather than only short
`qa` queries. It is reached over `https://api.mistral.ai/v1/chat/completions`
using `urllib` from the standard library; no SDK and no new dependency. Set
`MISTRAL_ENABLED=true` and `MISTRAL_API_KEY` to arm it — with either missing the
deployment stays dark and the draft chain behaves exactly as before.

Free-tier Mistral keys carry the same posture as the free-tier Gemini keys this
project already uses: the provider may train on submitted data. Only retrieved
textbook context and the user's question are sent, which is what every other
configured provider already receives.

OpenRouter, OmniRoute, Cerebras, NVIDIA NIM, and OpenCode are not enabled. Do
not route textbook context through generic gateway auto-routing or context
compression — the objection is to an *unknown* downstream provider and
retention policy, which is why named direct providers are approved and gateways
are not.
