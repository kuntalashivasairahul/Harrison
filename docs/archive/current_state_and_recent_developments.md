# Current State and Recent Developments

Audit date: 2026-07-11

## What Changed Recently

- The vector store was rebuilt in staging with `BAAI/bge-m3` and promoted to production.
- `scripts/ingest_tables_aware.py` now has a real `--promote-only` path that copies existing staging artifacts to production without rebuilding or re-embedding (`scripts/ingest_tables_aware.py:876`, `scripts/ingest_tables_aware.py:1166`).
- The production and staging stores now both contain 16,983 vectors with 1024 dimensions.
- The rerank score threshold was recalibrated from `-2.0` to `-3.0` in code (`backend/retrieval/rag.py:25`) and guarded by `tests/test_rag_threshold.py`.
- Confidence scoring moved to Harrison-calibrated average cross-encoder thresholds in `backend/agents/confidence_scorer.py`.
- The LLM layer has migrated from Groq to Gemini via `google-genai` (`backend/llm/llm.py:7`, `backend/requirements.txt`).
- `disable_verifier` exists on the API request model (`backend/api/main.py:50`) and `ask_llm()` returns explicit internal return paths.

## What Is Stable or Working

- Production and staging vector stores are internally aligned with each other:
  - `ntotal = 16983`
  - `dimension = 1024`
  - `chunk_count = 16983`
  - chunk schema includes `page`, `text`, `source_heading`, and `chunk_type`
- Table-aware ingestion has a safer operational split:
  - build staging,
  - validate staging,
  - promote staging to production,
  - backup production before promotion unless `--no-backup` is used.
- `promote_only()` does not load models or rebuild embeddings.
- Query optimizer fallback remains conservative and typed: original query, `complexity="complex"`, `is_medical_query=True`, `optimizer_used=False`.
- Verification fallback paths distinguish complete verified answers from draft fallback and graceful fallback internally.
- Structured timing fields are returned in `QueryResponse.timings`.

## What Was Migrated

| Item | Intended/current migration | Evidence |
| --- | --- | --- |
| Vector DB | MiniLM to BGE-M3 | `scripts/ingest_tables_aware.py:144`; production FAISS dimension verified as 1024. |
| Rerank threshold | `-2.0` to `-3.0` | `backend/retrieval/rag.py:25`; `tests/test_rag_threshold.py`. |
| LLM provider | Groq to Gemini | `backend/llm/llm.py:7`, `backend/requirements.txt`. |
| Runtime environment | `.venv`/unstable Python to `.venv312` | `scripts/setup_env.sh`, `scripts/run_benchmark.sh`. |
| Confidence scorer | top-score/evidence heuristic to average-score/divergence heuristic | `backend/agents/confidence_scorer.py:66`. |

## What Remains Inconsistent

- Live query embedding has not migrated with the artifacts. `backend/retrieval/embeddings.py` still loads `all-MiniLM-L6-v2`, while production FAISS is 1024-dimensional BGE-M3.
- `backend/config.py` still says the embedding model is `sentence-transformers/all-MiniLM-L6-v2`.
- Docs still describe Groq, `GROQ_API_KEY`, MiniLM, `.venv`, and `RERANK_SCORE_THRESHOLD=-2.0`.
- Several diagnostic scripts still encode production queries with MiniLM, so they are no longer valid for the promoted production index.
- `scripts/evaluate_rag.py` mixes the old `google.generativeai` SDK with the new runtime `google-genai` code and calls a missing `KeyManager.configure_current()` method.

## What Remains Uncertain

- Whether the running server process, if any, was restarted after promotion. `rag.py` loads FAISS/chunks at import time, so a running Uvicorn process keeps old in-memory artifacts until restart.
- Whether `artifacts/semantic_cache.json` contains old MiniLM-based cache entries. The code catches similarity errors, but old cache entries could become inert after a BGE migration.
- Whether page image files in `storage/pages/` fully match the hardcoded `FAISS_TO_IMAGE_OFFSET = 43`; the resolver constructs URLs but does not verify file existence.
- Whether retrieval quality issues observed after migration are dense-retrieval issues, reranker threshold issues, cache issues, or generation/verification issues. The current embedding mismatch must be fixed first before judging retrieval quality.

## Practical Current State

The vector artifacts appear promoted and consistent with BGE-M3, but the live retrieval code is not internally consistent with those artifacts. Cache hits may still return responses, but cache misses that reach FAISS dense search are at risk of failing because a 384-dimensional MiniLM query vector is searched against a 1024-dimensional BGE-M3 index.
