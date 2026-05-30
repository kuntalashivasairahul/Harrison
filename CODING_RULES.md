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
caller. Its presence in `ask_llm()` is an **architectural invariant**, not a
feature flag. See §RULE 3.2 for AI-specific enforcement.

---

## RULE 2 — Modular Isolation

**Each module has a single responsibility. Cross-layer contamination
introduces bugs that are extremely hard to trace in a pipeline system.**

### 2.1 Layer Boundaries Are Strict

| Layer              | Allowed Dependencies                           | Forbidden                             |
|--------------------|------------------------------------------------|---------------------------------------|
| `api/main.py`      | All backend modules (orchestration only)       | Business logic, retrieval math        |
| `retrieval/`       | `embeddings.py`, `rerank.py`, `rank_bm25`      | FastAPI, `llm/`, `rendering/`         |
| `llm/`             | `groq`, `os`, `re`, `pathlib`                  | FastAPI, `retrieval/`, `rendering/`   |
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

The following constants are defined in `backend/retrieval/rag.py` and are
**immutable**:

```python
RRF_K                  = 60     # Reciprocal Rank Fusion K
RERANK_SCORE_THRESHOLD = -2.0   # Hard filter logit threshold
```

Any change to these values requires a documented rationale with empirical
evaluation results, updated in both `ARCHITECTURE.md` and this file.

### 2.3 Pure Functions Must Stay Pure

The following functions are **pure** (no I/O, no side effects) and must
remain so:

```python
calculate_confidence(top_reranker_score, evidence_count, was_verified) → str
extract_evidence(chunks)  → List[str]
extract_sources(chunks)   → List[str]
resolve_page_urls(sources, base_url) → List[Dict[str, str]]
fuse_context(chunks) → str
clean_text(text) → str
top_score(ranked_chunks) → float
```

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
❌ Remove, comment out, or wrap verify_answer() in a conditional.
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

1. **Does the refactor preserve the 14-stage pipeline order?** (See `ARCHITECTURE.md §1`)
2. **Does the refactor preserve the `QueryResponse` schema?** (See `ARCHITECTURE.md §2`)
3. **Does the refactor maintain module isolation?** (See RULE 2.1 above)
4. **Does the refactor avoid touching `artifacts/` or `storage/`?**

If any answer is **No**, the refactor requires explicit human approval with
written justification before execution.

---

## RULE 4 — Prompting Standards

**The LLM prompts are load-bearing architecture.** Changes to
`BASE_QA_PROMPT` or `SMART_SUMMARY_PROMPT` in `backend/llm/llm.py` affect
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
- `backend/utils/scoring.py`
- `backend/retrieval/rerank.py`

### 5.2 Confidence Score Smoke Test

After any change to `calculate_confidence()` or `rerank.py`, verify that:

- A query with strong retrieval returns `confidence: "High"`.
- A query with no FAISS index loaded returns `confidence: "Low"`.
- A query returning `REFUSAL_STR` has `was_verified=False` in logs.

### 5.3 Health Check Gate

The `/health` endpoint must return `"status": "ok"` before any change is
considered complete:

```bash
curl http://127.0.0.1:8000/health
# Expected: {"status":"ok","faiss_loaded":true,"chunks_loaded":true,"groq_key_present":true}
```

### 5.4 Evaluation Harness

For changes to retrieval parameters (`DEFAULT_K`, `DEFAULT_FINAL_K`,
`RERANK_SCORE_THRESHOLD`, `RRF_K`), run the evaluation harness in
`evaluation/` before and after to confirm no regression in recall or
precision metrics.

---

## RULE 6 — Dependency Governance

### 6.1 Approved Dependencies

The following are the **only** approved runtime dependencies:

```
fastapi
uvicorn
faiss-cpu
sentence-transformers
numpy
groq
python-dotenv
rank-bm25
PyMuPDF          ← for pre-processing page renders (offline only)
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
│ Keep threshold = -2.0  │ Relax the hard filter silently     │
│ Log side-effects in API│ Add I/O to pure functions          │
│ Exclude artifacts/     │ Let AI index vectorstore data      │
└────────────────────────┴────────────────────────────────────┘
```

---

*Last updated: 2026-05-30 | Maintainer: HarrisonGPT AI Governance*
