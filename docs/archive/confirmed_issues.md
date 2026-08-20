# Confirmed Issues

Audit date: 2026-07-11

Only directly supported findings are included here. Hypotheses and tuning concerns are in `likely_issues_and_risks.md`.

## Issue 1: Live Query Embeddings Do Not Match the Production FAISS Index

| Field | Detail |
| --- | --- |
| Severity | Critical |
| Files | `backend/retrieval/embeddings.py:5`, `scripts/ingest_tables_aware.py:144`, `backend/retrieval/rag.py:48`, `backend/config.py:12` |
| Evidence | Production FAISS index verified locally as dimension 1024 with 16,983 vectors. `scripts/ingest_tables_aware.py` builds `BAAI/bge-m3` with `EMBEDDING_DIM = 1024`. Live `embed_text()` still loads `all-MiniLM-L6-v2`, which is 384-dimensional. A direct FAISS search using a 384-dimensional vector against the production index raised `AssertionError`; a 1024-dimensional vector succeeded. |
| Why it is a problem | On cache misses, `_hybrid_candidates()` calls `embed_text()` and then `index.search()`. With a 1024-dimensional index and 384-dimensional query vector, dense retrieval cannot run correctly. |
| Recommended next step | Update `backend/retrieval/embeddings.py` to use BGE-M3, normalize embeddings to match ingestion, and centralize the model name in config. Add a startup/health dimension check that compares `index.d` with a query embedding dimension. |

## Issue 2: Central Config Still Points to MiniLM and Is Not Used Consistently

| Field | Detail |
| --- | --- |
| Severity | High |
| Files | `backend/config.py:12`, `backend/retrieval/embeddings.py:5`, `scripts/ingest_tables_aware.py:144` |
| Evidence | `backend/config.py` says `EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"`. `backend/retrieval/embeddings.py` hardcodes MiniLM independently. `scripts/ingest_tables_aware.py` hardcodes BGE-M3 independently. |
| Why it is a problem | There is no single source of truth for embedding model or dimension, which allowed production artifacts and runtime embedding to drift apart. |
| Recommended next step | Move active embedding model and expected dimension into one config location and import it from ingestion, runtime embedding, diagnostics, and health checks. |

## Issue 3: Health Check Can Report OK While Retrieval Is Dimension-Incompatible

| Field | Detail |
| --- | --- |
| Severity | High |
| Files | `backend/api/main.py:326`, `backend/retrieval/rag.py:48`, `backend/retrieval/embeddings.py:5` |
| Evidence | `/health` checks only `rag.index is not None`, chunks loaded, and Gemini keys. It does not check that the live embedding dimension matches `rag.index.d`. |
| Why it is a problem | The server can look healthy while FAISS search fails on the first uncached retrieval request. |
| Recommended next step | Add `embedding_model`, `embedding_dim`, `faiss_dim`, and `embedding_index_dim_match` to `/health`; degrade status when dimensions mismatch. |

## Issue 4: Production Retrieval Diagnostics Still Use MiniLM and Old Thresholds

| Field | Detail |
| --- | --- |
| Severity | High |
| Files | `scripts/test_retrieval.py:64`, `scripts/test_retrieval.py:67`, `scripts/test_retrieval_staging.py:59`, `scripts/test_retrieval_staging.py:68`, `scripts/test_benchmark_step.py:24`, `scripts/test_environment.py:32` |
| Evidence | `scripts/test_retrieval.py` loads production `artifacts/vectorstore/index.faiss` but encodes queries with `all-MiniLM-L6-v2` and labels `RERANK_SCORE_THRESHOLD = -2.0` as live. `scripts/test_retrieval_staging.py` uses BGE-M3 for staging but still labels the live threshold as `-2.0`. `scripts/test_benchmark_step.py` uses MiniLM against production. `scripts/test_environment.py` asserts embeddings are `(1, 384)`. |
| Why it is a problem | These scripts either fail against the 1024-dimensional production index or give operators stale conclusions after the BGE-M3 migration. |
| Recommended next step | Update production diagnostics to BGE-M3/1024 and `RERANK_SCORE_THRESHOLD=-3.0`, or clearly rename old MiniLM scripts as historical benchmark tools. |

## Issue 5: `scripts/evaluate_rag.py` Is Out of Sync With the Gemini Runtime

| Field | Detail |
| --- | --- |
| Severity | High |
| Files | `scripts/evaluate_rag.py:54`, `scripts/evaluate_rag.py:367`, `backend/llm/llm.py:89`, `backend/requirements.txt` |
| Evidence | The script imports `google.generativeai`, but runtime code uses `google-genai` via `from google import genai`. The script calls `key_manager.configure_current()`, but `KeyManager` defines `next_client()`, `make_client()`, `mark_exhausted()`, and `rotate()`, not `configure_current()`. |
| Why it is a problem | The LLM-as-judge harness is likely broken or using an uninstalled/old SDK path. It cannot be trusted for post-migration evaluation until repaired. |
| Recommended next step | Rewrite judge calls to use the same `google-genai` client pattern as `backend/llm/llm.py`, or explicitly add and isolate the old SDK if intentionally used. |

## Issue 6: Semantic Cache Ignores Mode, Verifier Flag, Return Path, and Pipeline Version

| Field | Detail |
| --- | --- |
| Severity | Medium-High |
| Files | `backend/api/main.py:165`, `backend/api/main.py:167`, `backend/agents/semantic_cache.py:226`, `backend/agents/semantic_cache.py:289` |
| Evidence | Cache lookup is keyed only by the query embedding. The API returns cached responses before `disable_verifier` reaches `ask_llm()`. Cache entries store answer/confidence/sources/visual_context only. |
| Why it is a problem | A `qa` request and `smart_summary` request with the same search query can share a response. A request with `disable_verifier=true` can still receive a previously verified response, or vice versa. Old-pipeline responses can survive model/prompt/index changes. |
| Recommended next step | Include `mode`, verifier setting, embedding model/version, prompt/pipeline version, and return-path safety metadata in cache keys or cache eligibility. Clear cache after embedding/prompt/index changes. |

## Issue 7: Context Fusion Claims a Safety Budget but Does Not Enforce It

| Field | Detail |
| --- | --- |
| Severity | Medium |
| Files | `backend/utils/fusion.py:19`, `backend/utils/fusion.py:56` |
| Evidence | `SAFE_TOKEN_LIMIT` and `SAFE_CHAR_LIMIT` are defined and comments say chunks over budget are dropped, but `fuse_chunks()` appends every eligible chunk and never checks `running_chars` or `SAFE_CHAR_LIMIT`. |
| Why it is a problem | Token and latency behavior can be misleading. Oversized prompts can increase cost, latency, truncation risk, or provider failures. |
| Recommended next step | Either enforce the character budget with tests or remove the misleading budget comments. |

## Issue 8: Observability Fields Are Misleading in Two Places

| Field | Detail |
| --- | --- |
| Severity | Medium |
| Files | `backend/api/main.py:267`, `backend/api/main.py:276`, `backend/retrieval/rag.py:397`, `backend/retrieval/rag.py:411` |
| Evidence | API log field `expanded_query_is_static` is populated with `optimizer_failed`, not actual expanded-query equality. Retrieval logs hardcode `verification_performed=True`, even though retrieval cannot know whether verification later ran or was disabled. |
| Why it is a problem | Latency and safety diagnosis can be wrong, especially when separating optimizer fallback, retrieval behavior, verifier behavior, and API return paths. |
| Recommended next step | Log exact fields: `optimizer_used`, `expanded_query_equals_raw`, `cache_hit`, `disable_verifier`, `returned_path`, and `verifier_ran`. Remove verification claims from retrieval logs. |

## Issue 9: Docs and Governance Files Are Stale Against Runtime Code

| Field | Detail |
| --- | --- |
| Severity | Medium |
| Files | `README.md:11`, `README.md:23`, `README.md:50`, `README.md:64`, `README.md:109`, `ARCHITECTURE.md:66`, `ARCHITECTURE.md:94`, `ARCHITECTURE.md:270`, `PROJECT_CONTEXT.md:70`, `PROJECT_CONTEXT.md:76`, `PROJECT_CONTEXT.md:104`, `PROJECT_CONTEXT.md:135`, `PROJECT_CONTEXT.md:154`, `CODING_RULES.md:267`, `CODING_RULES.md:291` |
| Evidence | Docs still describe MiniLM, Groq, `GROQ_API_KEY`, `.venv`, `groq_key_present`, `backend/utils/scoring.py`, and threshold `-2.0`. Runtime code uses Gemini, `GEMINI_API_KEY_*`, `.venv312` for stable scripts, `backend/agents/confidence_scorer.py`, and threshold `-3.0`. |
| Why it is a problem | New engineers and agents will follow stale commands and stale architectural invariants. In this project, stale model/dimension docs are operationally dangerous. |
| Recommended next step | Update docs after fixing runtime embedding. Mark old MiniLM/Groq material as historical if still useful. |

## Issue 10: `SMART_SUMMARY_FINAL_K` Is Defined but Not Used

| Field | Detail |
| --- | --- |
| Severity | Medium-Low |
| Files | `backend/api/main.py:36`, `backend/api/main.py:37`, `backend/api/main.py:188` |
| Evidence | `SMART_SUMMARY_FINAL_K` is read from the environment but smart summary retrieval passes `final_k=dynamic_final_k`, where `dynamic_final_k` is 5 or 12 based on optimizer complexity. |
| Why it is a problem | Operators may believe `SMART_SUMMARY_FINAL_K` changes runtime behavior when it does not. |
| Recommended next step | Remove the unused setting or define how it combines with complexity-based depth, then test it. |

## Issue 11: `ask_llm` Generation Token Settings Ignore the Named Env Limits

| Field | Detail |
| --- | --- |
| Severity | Medium-Low |
| Files | `backend/llm/llm.py:381`, `backend/llm/llm.py:382`, `backend/llm/llm.py:689`, `backend/llm/llm.py:695` |
| Evidence | `SMART_SUMMARY_MAX_TOKENS` and `QA_MAX_TOKENS` are loaded, and verification uses mode-aware limits, but draft generation always passes `max_output_tokens=8192`. |
| Why it is a problem | Env-token tuning does not apply to draft generation, so latency/cost/truncation tuning is harder to reason about. |
| Recommended next step | Use mode-aware token limits for draft generation or rename/env-document the 8192 hard limit as intentional. |

## Issue 12: A Stale Truncation Test Still Expects the Old `ask_llm` Return Shape

| Field | Detail |
| --- | --- |
| Severity | Low-Medium |
| Files | `evaluation/test_truncation_logic.py`, `backend/llm/llm.py:646` |
| Evidence | `ask_llm()` now documents and returns four values. `evaluation/test_truncation_logic.py` still unpacks three values in multiple places. |
| Why it is a problem | Running old evaluation tests gives false failures and obscures the current truncation/return-path behavior. |
| Recommended next step | Update or delete the stale evaluation test in favor of `tests/test_pipeline_e2e.py`. |
