# CODING_RULES.md
# HarrisonGPT — AI Governance & Coding Rules

> **Enforcement Level: MANDATORY.**
> These rules govern all human contributors and AI coding assistants working
> on this repository. Violations in a medical AI system are not style
> infractions — they are patient-safety risks.
>
> Every rule below has a rationale. Read it before dismissing a rule as
> inconvenient.

---

## RULE 1 — Accuracy Over Creativity

**Medical accuracy is the highest-priority constraint. It overrides code
elegance, latency, token efficiency, and user experience.**

### 1.1 Never Invent Medical Claims

```
❌ FORBIDDEN: Generating a medical claim not present in the retrieved context.
❌ FORBIDDEN: Synthesizing "common knowledge" medicine when the context is absent.
✅ REQUIRED:  Return REFUSAL_STR if the retrieved context is insufficient.
```

The correct refusal string is defined in `backend/llm/llm.py`:
```python
REFUSAL_STR = "Insufficient information in the provided context."
```

**Do not invent an alternative refusal.** Do not soften it. Do not try to
answer anyway with a disclaimer.

### 1.2 Never Invent Citations

```
❌ FORBIDDEN: Writing [p:9999] or any page marker not present in retrieved_chunks.
❌ FORBIDDEN: Paraphrasing a page number (e.g., "around page 2000").
✅ REQUIRED:  Only propagate page markers that appear in fused_context or evidence.
```

Citation invention is equivalent to fabricating a reference in a peer-reviewed
paper — except the reader may act on it clinically.

### 1.3 The Verify Step is Not Optional

```
❌ FORBIDDEN: Skipping verify_answer() to reduce latency.
❌ FORBIDDEN: Bypassing verify_answer() by returning draft_answer directly.
✅ REQUIRED:  verify_answer() MUST be called on every non-refusal answer.
```

`verify_answer()` is the last safety gate before the response reaches the
caller. It may only be bypassed when an explicit API request sets
`disable_verifier=true`; those responses are capped in confidence and are not
eligible for semantic-cache persistence. See §RULE 3.2 for AI-specific
enforcement.

---

## RULE 2 — Modular Isolation

**Each module has a single responsibility. Cross-layer contamination
introduces bugs that are extremely hard to trace in a pipeline system.**

### 2.1 Layer Boundaries Are Strict

| Layer              | Allowed Dependencies                           | Forbidden                             |
|--------------------|------------------------------------------------|---------------------------------------|
| `api/main.py`      | All backend modules (orchestration only)       | Business logic, retrieval math        |
| `retrieval/`       | `embeddings.py`, `rerank.py`, `rank_bm25`      | FastAPI, `llm/`, `rendering/`         |
| `llm/`             | `google.genai`, `os`, `re`, `pathlib`          | FastAPI, `retrieval/`, `rendering/`   |
| `processing/`      | `utils/fusion.py` only                         | FastAPI, `retrieval/`, `llm/`         |
| `rendering/`       | `re`, `typing`                                 | FastAPI, `retrieval/`, `llm/`         |
| `utils/`           | Standard library only                          | Any domain module                     |

```
❌ FORBIDDEN: retrieval/rag.py importing from fastapi or llm/
❌ FORBIDDEN: llm/llm.py importing from retrieval/ or processing/
❌ FORBIDDEN: Inlining fuse_context() logic inside api/main.py
✅ REQUIRED:  api/main.py orchestrates; domain modules do the work
```

### 2.2 Retrieval Math Lives in retrieval/

**Do not move RRF scoring, BM25 tokenization, FAISS search, neighbor
expansion, or the hard filter into `api/main.py` or `utils/`.**

The following constants are defined in `backend/config.py` and are
**immutable**:

```python
RRF_K                  = 60     # Reciprocal Rank Fusion K
RERANK_SCORE_THRESHOLD = -3.0   # Hard filter logit threshold
```

Any change to these values requires a documented rationale with empirical
evaluation results, updated in both `ARCHITECTURE.md` and this file.

### 2.3 Pure Functions Must Stay Pure

The following functions are **pure** (no I/O, no side effects) and must
remain so:

```python
calculate_confidence(chunks, original_answer, verified_answer) → str
extract_evidence(chunks)  → List[str]
extract_sources(chunks)   → List[str]
resolve_page_urls(sources, base_url) → List[Dict[str, str]]
fuse_context(chunks) → str
clean_text(text) → str
```

`top_score()` was previously listed here. It was removed once its last
production caller disappeared from `api/main.py`; nothing but a test stub
referenced it.

```
❌ FORBIDDEN: Adding logging, DB writes, or HTTP calls inside these functions.
✅ REQUIRED:  Logging and side-effects belong in api/main.py or the caller.
```

---

## RULE 3 — AI Tooling Guardrails

**These rules govern the behavior of AI coding assistants (Copilot, Cursor,
Gemini, Claude, etc.) working on this codebase.**

### 3.1 Forbidden Scan Targets

AI IDEs and agentic tools **must not** index, read, or analyze:

| Path                        | Reason                                          |
|-----------------------------|-------------------------------------------------|
| `artifacts/`                | Binary FAISS index + JSON chunk metadata; large, not code |
| `artifacts/vectorstore/`    | Sensitive vectorized textbook content           |
| `artifacts/retrieval_logs/` | High-volume auto-generated files; not source    |
| `storage/`                  | Binary image assets; not source code            |
| `storage/pages/`            | Thousands of PNG/WebP page renders              |

These directories must be listed in `.gitignore` for data protection and
should be excluded from IDE workspace indexing settings.

### 3.2 Immutable Pipeline Steps

An AI assistant **must NEVER**:

```
❌ Remove or bypass `verify_answer()` outside the documented `disable_verifier` request option.
❌ Remove, comment out, or wrap resolve_page_urls() in a conditional.
❌ Remove, comment out, or bypass the RERANK_SCORE_THRESHOLD filter.
❌ Remove the confidence, sources, or visual_context fields from QueryResponse.
❌ Substitute a hardcoded confidence string ("High") for calculate_confidence().
❌ Silently drop evidence from the LLM prompt to reduce token count.
```

An AI assistant **must ALWAYS**:

```
✅ Preserve all existing docstrings and inline comments in Python files.
✅ Keep the QueryResponse field set intact (answer, confidence, sources, visual_context).
✅ Route new features through api/main.py as orchestration, not into domain modules.
✅ Run the /health endpoint check after any change to config.py or rag.py.
✅ Justify any proposed change to RERANK_SCORE_THRESHOLD or RRF_K in writing.
```

### 3.3 Refactoring Constraints

Before proposing a refactor, an AI assistant must verify:

1. **Does the refactor preserve the documented request flow?** (See `ARCHITECTURE.md §1`)
2. **Does the refactor preserve the `QueryResponse` schema?** (See `ARCHITECTURE.md §2`)
3. **Does the refactor maintain module isolation?** (See RULE 2.1 above)
4. **Does the refactor avoid touching `artifacts/` or `storage/`?**

If any answer is **No**, the refactor requires explicit human approval with
written justification before execution.

---

## RULE 4 — Prompting Standards

**The LLM prompts are load-bearing architecture.** Changes to
`MASTER_MEDICAL_SYNTHESIS_PROMPT` or smart-summary formatting logic in
`backend/llm/llm.py` affects
every response the system produces.

### 4.1 Mandatory Prompt Prohibitions

All base prompts **must** contain explicit instructions to forbid:

```
❌ Step-by-step reasoning leakage: "Step 1", "Step 2", "Step 3"
❌ Chain-of-thought markers: "reasoning:", "thinking:", "final answer:"
❌ Knowledge outside context: "Based on general medical knowledge..."
❌ Invented citations: page numbers not present in the provided context
❌ Empty section headers: headings with no supporting evidence
```

These prohibitions are already enforced in the current prompts. Do not remove
them to make prompts shorter.

### 4.2 Required Prompt Elements

All base prompts **must** contain:

```
✅ Instruction to use ONLY the provided Harrison context
✅ Instruction to cite page numbers using [p:NNN] markers
✅ Instruction to never invent or guess page numbers
✅ Instruction to synthesize evidence into textbook-style prose (not list of bullets)
✅ Instruction to output the final answer directly (no preamble/reasoning chain)
```

### 4.3 Smart Summary Acknowledgement Invariant

The `smart_summary` mode **must** produce a first line of exactly:

```
Topic received — generating Harrison Smart Summary.
```

This is enforced by `_enforce_smart_summary_shape()` in `llm.py`. Do not
remove this function or its call sites.

### 4.4 Verification Prompt Standards

`verify_answer()` uses a separate system + user prompt pair.  Its rules are:

```
✅ Check each factual claim against context
✅ Keep supported claims
✅ Rewrite partially-supported claims to match context
✅ Remove only claims that cannot be supported at all
✅ Never invent new page numbers
✅ Always output a single corrected answer (no meta-commentary)
```

The verification temperature is always `0.0` (deterministic). Do not change it.

---

## RULE 5 — Testing & Validation Requirements

### 5.1 Before Any Merge

Any change that touches the following files requires manual validation against
the `/ask` endpoint with at least one `smart_summary` and one `qa` query:

- `backend/retrieval/rag.py`
- `backend/llm/llm.py`
- `backend/api/main.py`
- `backend/agents/confidence_scorer.py`
- `backend/retrieval/rerank.py`

### 5.2 Confidence Score Smoke Test

After any change to `calculate_confidence()` or `rerank.py`, verify that:

- A query with strong retrieval returns `confidence: "High"`.
- A query with no FAISS index loaded returns `confidence: "Low"`.
- A query returning `REFUSAL_STR` has `was_verified=False` in logs.

### 5.2a Hermetic Test Suite

The suite in `tests/` **must** stay hermetic: no network calls, no model
weights, no FAISS index. That is why it runs in seconds and why CI needs no
credentials.

```
❌ FORBIDDEN: A test in tests/ that loads the real index or calls a live model.
❌ FORBIDDEN: Mutating sys.modules without restoring it — one test that did
              silently replaced the Google SDK for every test collected after it.
❌ FORBIDDEN: Naming a program in scripts/ `test_*.py`. Those are interactive
              diagnostics that load real models; pytest must not collect them.
              They are named `probe_*.py`.
✅ REQUIRED:  Use tests/_api_harness.py to import the HTTP layer cheaply.
```

Integration coverage against the real index and a live model is a separate tier
that does not exist yet. It is the main outstanding gap in this repository.

### 5.3 Health Check Gate

The `/health` endpoint must return `"status": "ok"` before any change is
considered complete:

```bash
curl http://127.0.0.1:8000/health
# Expected fields include: status=ok, faiss_loaded=true, chunks_loaded=true,
# embedding_index_dim_match=true, and gemini_key_present=true.
```

### 5.4 Evaluation Harness

For changes to retrieval parameters (`DEFAULT_K`, `DEFAULT_RERANK_POOL`,
`RERANK_SCORE_THRESHOLD`, `RRF_K`) **or to retrieval behaviour** — the
tokenizer, the low-value filter, neighbour expansion, fusion ordering — run
`scripts/evaluate_rag.py` before and after to confirm no regression in recall
or precision.

This rule previously named only the parameters. Behavioural changes to the same
pipeline have at least as much effect and were not covered.

---

## RULE 5a — Import-Time Purity

No module under `backend/` may perform expensive or networked work at import
time.

```
❌ FORBIDDEN: Loading the FAISS index, chunk registry, BM25 corpus, or an
              encoder at module scope.
❌ FORBIDDEN: Any network call at import — model discovery included.
❌ FORBIDDEN: logging.getLogger("uvicorn.error") in a backend module. Uvicorn
              leaves root bare at WARNING, so backend.* INFO logs were being
              discarded; two modules had worked around it locally, which fixed
              those call sites and hid the cause.
✅ REQUIRED:  Resolve on first use; force it deliberately in the FastAPI
              lifespan handler.
✅ REQUIRED:  Call configure_logging() before any other backend.* import in an
              entry point, so import-time diagnostics are captured.
✅ REQUIRED:  Module loggers via logging.getLogger(__name__).
```

Rationale: import-time work is paid by every test run, every diagnostic script,
and every tool that merely wants a helper function — and a network call at
import makes the app's startup depend on a third party before `/health` can
answer.

---

## RULE 6 — Dependency Governance

### 6.1 Approved Dependencies

The following are the **only** approved runtime dependencies:

```
fastapi
uvicorn
pydantic         ← FastAPI's schema layer; pinned because QueryResponse is a frozen contract
faiss-cpu
sentence-transformers
torch            ← required by sentence-transformers; pinned, not newly introduced
transformers     ← required by sentence-transformers; pinned, not newly introduced
numpy
google-genai
groq             ← query optimizer, and draft failover of last resort; never verification
                   (Mistral is also an approved draft provider — stdlib urllib, no package)
python-dotenv
rank-bm25
PyMuPDF          ← for pre-processing page renders (offline only)
```

**On the pinned transitive dependencies.** `pydantic`, `torch` and
`transformers` are not new capabilities — they were always installed as
transitive dependencies of `fastapi` and `sentence-transformers`. They are now
named and version-pinned in `backend/requirements.txt` because leaving them
floating meant a fresh install resolved a different `transformers`/`torch` pair
with no guarantee of reproducing the same embeddings against the committed
FAISS index. Pinning a dependency you already had is a reproducibility control,
not a new dependency; adding a genuinely new package still requires §6.2.

`groq` predates this list. It serves the optimizer stage, and the `groq-draft`
deployment serves the draft stage at priority 30 — reached only after both
Gemini draft deployments have failed with a fallback-eligible error. The
original "never draft" restriction was written when Groq meant an 8B optimizer
model; it now serves 120B-class models, and a Groq draft is still verified by
Gemini against the same context, so `verify_answer()` and the citation checks
are not weakened. **Groq must never serve the verifier stage** — that would
leave no independent check on a Groq draft. See the provider policy in
`README.md`.

**Mistral is an approved draft provider and adds no dependency.** The
`mistral-draft` deployment (priority 25) is served by
`backend/llm/mistral_provider.py`, which posts to Mistral's chat-completions
endpoint using `urllib` from the standard library. The `mistralai` SDK would
need a §6.2 justification to buy one JSON POST, and `httpx` is a *development*
dependency that §6.1a forbids importing from `backend/` — so neither is used.
Nothing was added to `backend/requirements.txt` for this provider, and there is
nothing to pin.

The verifier restriction is not Groq-specific and must not be read that way:
**Gemini is the only approved verifier.** Any provider added to the draft stage
inherits the same bargain — it may write a first pass, and Gemini decides
whether that pass survives. `tests/test_llm_router.py` asserts this against the
registry for every non-Gemini deployment, so a future entry cannot quietly
acquire the verifier stage.

### 6.1a Development Dependencies

`backend/requirements-dev.txt` carries test and lint tooling. These are not
runtime dependencies and must never be imported by anything under `backend/`:

```
pytest
httpx            ← required by fastapi.testclient
ruff
```

### 6.2 Adding a New Dependency

Adding any new dependency requires:

1. Written justification in the PR description.
2. Confirmation it does not conflict with Apple Silicon (arm64) native builds.
3. No GPU-only libraries (e.g., `faiss-gpu`) — the runtime target is CPU.
4. No `torch` unless adding a new model that genuinely requires it.
5. Update `backend/requirements.txt` and this section.

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│                  HarrisonGPT — DO / DON'T                   │
├────────────────────────┬────────────────────────────────────┤
│          ✅ DO          │            ❌ DON'T                 │
├────────────────────────┼────────────────────────────────────┤
│ Cite [p:NNN] from ctx  │ Invent page numbers                │
│ Return REFUSAL_STR     │ Answer without context             │
│ Call verify_answer()   │ Skip or bypass verify              │
│ Keep QueryResponse     │ Drop confidence/sources fields     │
│ Keep module isolation  │ Entangle retrieval with routing    │
│ Keep REFUSAL_STR exact │ Reword the refusal                 │
│ Keep RRF_K = 60        │ Change fusion math ad-hoc          │
│ Keep threshold = -3.0  │ Relax the hard filter silently     │
│ Log side-effects in API│ Add I/O to pure functions          │
│ Exclude artifacts/     │ Let AI index vectorstore data      │
└────────────────────────┴────────────────────────────────────┘
```

---

*Last updated: 2026-08-21 | Maintainer: HarrisonGPT AI Governance*
