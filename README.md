# HarrisonGPT

> **Built with AI assistance, reviewed by a human.** Every change was read and
> accepted by a person before it landed, and the design decisions are human
> calls. Details, including a case where that review caught something, are in
> [Provenance](#provenance-built-with-ai-reviewed-by-a-human). This is a study
> aid, **not a diagnostic device**.

HarrisonGPT is a production-grade, high-recall **Medical Retrieval-Augmented Generation (RAG) System** grounded exclusively in *Harrison's Principles of Internal Medicine* (20th+ Edition). Designed for clinicians, medical students, and exam preparation, the system prioritizes factual fidelity and citation grounding over raw creativity.

---

## Key Features

* **Stage-Aware LLM Routing:** Uses an approved local registry: Groq optimizes queries when explicitly enabled, Gemini drafts and verifies, Mistral and Groq back the draft stage when Gemini is rate-limited, Mistral also backs the verifier stage, no model is ever allowed to verify its own draft, and all provider fallback decisions are logged.
* **Low-Latency Semantic Caching:** Utilizes a disk-persistent semantic cache (`artifacts/semantic_cache.json`) to serve clinically equivalent queries instantly ($\ge 0.95$ Cosine Similarity) in under ~1ms.
* **Hybrid Retrieval Pipeline:** Merges FAISS dense search using `BAAI/bge-m3` (1024 dimensions) with BM25Okapi sparse lexical search via Reciprocal Rank Fusion (RRF), alongside local context neighbor chunk expansion.
* **Cross-Encoder Rerank Filtering:** Scores chunks via an `ms-marco-MiniLM-L-6-v2` cross-encoder, filtering out noisy passages scoring below `-3.0`.
* **Double-Pass Grounding & Verification:** Employs a post-hoc self-consistency check (`verify_answer`) to compare draft answers against raw context source page boundaries, rephrasing or redacting any claims not fully supported by the textbook.
* **Visual Grounding & Image URL Resolution:** Resolves textbook citations (e.g., `p.2787`) to static thumbnail WebP or full-resolution PNG page image URLs hosted on the local static server.

---

## Technology Stack

* **API & Serving:** FastAPI, Uvicorn, Pydantic, Python-dotenv
* **Vector & Lexical Search:** FAISS (CPU), Rank-BM25
* **AI Embeddings & Reranking:** SentenceTransformers (`BAAI/bge-m3`, 1024 dimensions, and `ms-marco-MiniLM-L-6-v2`)
* **Large Language Models:** Google Gen AI SDK (`google-genai`) for grounded answers and verification; Groq for the optional query-optimizer route; Mistral over stdlib `urllib` for draft failover.

---

## System Control Flow & Architecture

For a detailed walkthrough, step-by-step trace, and sequence diagram of how control flows from request intake to response delivery, refer to [workflow.md](workflow.md).

---

## Setup & Installation

### 1. Prerequisites
* Python 3.12
* Virtual Environment manager (venv)
* The corpus — the FAISS index and chunk registry are **not in this repo**.
  They are licensed content, so a fresh clone starts without them and the
  API reports `status: "degraded"` on `/health` until you supply them.
  `docs/CORPUS.md` covers all three retrieval paths. `git-lfs` is no longer
  required.

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

## Running Locally

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
questions without rebuilding embeddings. It is **not** tracked in git (see `docs/CORPUS.md`)
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

## API Documentation

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

## Evaluation

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
| 22 | `mistral-draft` | Mistral | both Gemini drafts failed |
| 25 | `gemini-flash-3.6` | Gemini | Mistral also failed |
| 26 | `gemini-flash-3` | Gemini | as above |
| 30 | `groq-draft` | Groq | everything above failed, **and** the prompt fits 5k input tokens |

The **verifier** stage falls over the same way, minus Groq and minus
`mistral-draft`:

| Priority | Deployment | Provider | Reached when |
|---|---|---|---|
| 10 | `gemini-primary` | Gemini | always tried first |
| 20 | `gemini-draft-fallback` | Gemini | primary returned a fallback-eligible error |
| 25 | `gemini-flash-3.6` | Gemini | both of the above failed |
| 26 | `gemini-flash-3` | Gemini | as above |
| 27 | `mistral-verifier` | Mistral | every Gemini verifier failed |

Mistral verifies last, not fourth. It sat at priority 24 for a day, ahead of
both `gemini-flash-3.x` deployments, and that cost 20s of a 69s request on
2026-08-27: `mistral-medium-latest` hit its own 20s ceiling and returned
nothing, then `gemini-flash-3.6` verified the same answer in 6.7s. Ordering it
last also matches what the rule actually says — Gemini is the preferred
verifier and Mistral is the hedge for a Gemini-wide outage, which is precisely
the case where every deployment ahead of it has already failed.

Whichever model served the draft is removed from that order for the request
that produced it. `mistral-large-latest` drafts but never verifies: it was
measured at 23–30s live and hit its own 30s ceiling once, and that ceiling
cannot be raised — no draft deployment may claim more than a third of the 90s
request budget, or a cascade runs out of time before it reaches the deployments
behind the failure. `mistral-medium-latest` does the verifier's job inside 20s,
and a timeout on the draft stage is cheap because failover continues past it.

`mistral-draft` sits behind `gemini-draft-fallback` on purpose. It briefly ran at
priority 15, in front of it, and the cost was measured live: a `gemini-primary`
503 sent the draft straight to `mistral-large-latest`, which takes 23–30s and in
one request hit its own 30s timeout and produced nothing, while
`gemini-draft-fallback` served the same draft in 9.7s once it was finally
reached. `gemini-primary` returning "high demand" says nothing about
`gemini-3.5-flash` — they are different models — so one Gemini outage is not a
reason to pay for the slowest deployment in the fleet. Mistral still outranks the
remaining Gemini drafts, so a Gemini-wide quota failure reaches it quickly.

**No model verifies its own draft.** The verifier stage is open to Gemini and
Mistral, but the model that actually served the draft is dropped from the
verifier order at routing time — a model asked to grade its own work is the
least likely to catch its own ungrounded claim. The exclusion is per *model*,
not per provider: barring the whole provider would empty the verifier order
whenever Mistral is dark, trading a self-verified answer for an unverified one.
If the exclusion does empty the order, the stage fails rather than falling back
to the drafter; that caps confidence and returns the `draft_fallback` path,
which is honest, where a self-approved answer labelled `verified` would not be.
Groq never verifies: its 8k-token-per-minute ceiling means it would grade a
`smart_summary` draft against a truncated view of the evidence. The grounding
and citation checks in `verify_answer()` are unchanged. See CODING_RULES §6.1,
amended 2026-08-27.

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

---

## Provenance: built with AI, reviewed by a human

This project was written with substantial help from AI coding assistants, and
saying so plainly is part of taking it seriously. A system that answers medical
questions should not be vague about how it was built.

**What that means concretely.** Of the 54 commits in this repository, 20 carry a
`Co-Authored-By: Claude` trailer — 19 from Claude Opus 5 and one from Claude
Haiku 4.5. The trailers are in the git history and were not added retroactively;
`git log --format='%b' | grep Co-Authored-By` reproduces the count. AI wrote or
substantially edited most of the retrieval pipeline, the LLM router, the test
suite, and this documentation. The remaining commits are human-authored.

**What "reviewed by a human" means here.** Every change was read and accepted by
[@kuntalashivasairahul](https://github.com/kuntalashivasairahul) before it
landed. The design decisions — what the system refuses to answer, where
confidence is floored, which providers are approved, what stays immutable — are
human calls, written down in `CODING_RULES.md`, and the AI is held to them
rather than consulted about them. `CODING_RULES.md` exists precisely because an
assistant with commit access needs constraints it cannot argue its way out of.

**Where that review has already caught something.** Commit `d5739d9` put Mistral
on the verifier stage and rewrote the test that guarded against it *without*
amending the rule that forbade it, so the code and the written rule disagreed
for a day. That was found by audit, the rule was amended on the evidence rather
than quietly deleted, and the episode is recorded in `CODING_RULES.md §6.1`
under "Honest history" instead of being cleaned up. AI assistance does not
remove the need for review; that commit is the argument for it.

**What this does not claim.** It does not claim the code is correct because a
human looked at it, and it does not claim the medical content is safe to act on.
The system is a study aid grounded in one textbook. It refuses rather than
guesses, cites the pages it used, and is **not a diagnostic device** — the
footer on every page says so, and that is not boilerplate.
