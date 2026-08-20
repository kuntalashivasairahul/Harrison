# Likely Issues and Risks

Audit date: 2026-07-11

These are plausible or partially evidenced risks that should be verified before being treated as confirmed defects.

## Risk 1: Existing Semantic Cache Entries May Be Stale After BGE-M3 Migration

| Field | Detail |
| --- | --- |
| Evidence suggesting risk | The cache is stored in `artifacts/semantic_cache.json`, keyed by embedding only. The embedding model changed at the artifact level from MiniLM to BGE-M3, while the live embedder is currently still MiniLM. |
| Why it matters | After fixing live embeddings to BGE-M3, old MiniLM cache entries may become inert due to vector length mismatch, or stale responses may remain if not cleared. |
| How to verify | Inspect only cache metadata or clear the cache with `DELETE /admin/cache` after embedding migration. Avoid using stale cached answers as retrieval-quality evidence. |
| Recommended next step | Include an embedding/cache version in cache entries and clear cache as part of any embedding, prompt, or index promotion. |

## Risk 2: Query Optimizer Quota Failures Do Not Rotate Exhausted Keys

| Field | Detail |
| --- | --- |
| Evidence suggesting risk | `optimize_query()` calls `key_manager.make_client()` and broadly falls back on exception. Main generation and verification explicitly mark exhausted keys and retry on quota errors. |
| Why it matters | A quota-exhausted current key can cause repeated optimizer fallback even when other keys are available. This increases retrieval breadth and can allow non-medical queries down the expensive path. |
| How to verify | Simulate a quota exception in `optimize_query()` and assert whether `mark_exhausted()` or `next_client()` rotation happens. |
| Recommended next step | Give optimizer a bounded retry/rotation path matching `ask_llm()` but keep the safe fallback. |

## Risk 3: Non-Medical Queries Reach the Expensive Path When Optimizer Fails

| Field | Detail |
| --- | --- |
| Evidence suggesting risk | `_build_fallback()` sets `is_medical_query=True` by design. |
| Why it matters | This preserves recall for medical queries but means optimizer outage disables the non-medical gate. |
| How to verify | Force optimizer failure and send an off-topic query; inspect whether retrieval and LLM generation are attempted. |
| Recommended next step | Consider a cheap local off-topic heuristic before the LLM optimizer, or log this fallback explicitly so operators can distinguish fail-open behavior. |

## Risk 4: Confidence Can Be Overstated for Chunks Missing Rerank Scores

| Field | Detail |
| --- | --- |
| Evidence suggesting risk | `calculate_confidence()` treats missing `score` as `0.0`, which is above the `_HIGH_AVG_SCORE=-0.5` threshold. |
| Why it matters | If a caller passes unscored chunks, confidence may become Medium/High rather than Low. |
| How to verify | Add a unit test with chunks containing no `score` keys. |
| Recommended next step | Treat missing scores as absent/low confidence, or require all chunks used for confidence to carry scores. |

## Risk 5: Context Router Trades Relevance Ordering for Page Ordering

| Field | Detail |
| --- | --- |
| Evidence suggesting risk | `route_and_sort_context()` deduplicates then sorts surviving chunks by ascending page number. |
| Why it matters | Chronological context can help the LLM, but evidence extraction and prompt order may no longer reflect reranker confidence. This could matter for broad queries where the top chunk is not the earliest page. |
| How to verify | Compare answer quality and citations with rerank-order context versus page-order context on a small query set. |
| Recommended next step | Keep page ordering if it improves summaries, but preserve original rerank rank in metadata and diagnostics. |

## Risk 6: Promotion Is Not Atomic Across Index and Chunks

| Field | Detail |
| --- | --- |
| Evidence suggesting risk | `promote_only()` copies staging index to production, then copies staging chunks to production. |
| Why it matters | A crash between copies could leave mismatched production `index.faiss` and `chunks.json`. |
| How to verify | Review filesystem behavior and simulate interrupted promotion in a disposable copy. |
| Recommended next step | Promote into a temp production directory, validate index/chunk count/dimension, then atomically swap or rename. Keep rollback instructions. |

## Risk 7: Page Image Offset May Drift From Current Artifacts

| Field | Detail |
| --- | --- |
| Evidence suggesting risk | `FAISS_TO_IMAGE_OFFSET = 43` is hardcoded and the resolver does not check file existence. |
| Why it matters | The API can return plausible image URLs that 404 or point to the wrong page if page numbering changed during ingestion. |
| How to verify | Sample retrieved source pages, compute resolved URLs, and check corresponding files under `storage/pages/small` and `storage/pages/full`. |
| Recommended next step | Add a small page URL smoke test for known pages after ingestion/promotion. |

## Risk 8: Benchmark Scripts Do Not Fully Match Production Retrieval

| Field | Detail |
| --- | --- |
| Evidence suggesting risk | Benchmark scripts compare MiniLM and BGE shadow indices, use reduced query sets, and use heuristic relevance grading in the safe version. Production also has optimizer expansion, BM25, RRF, neighbor expansion, rerank thresholding, context router, cache, and LLM/verifier. |
| Why it matters | A benchmark can show dense model quality while production behavior is dominated by reranker thresholding, cache, prompt context, or verifier changes. |
| How to verify | Build a production-faithful retrieval-only benchmark path that imports current runtime constants and records pre-rerank, post-rerank, and post-threshold results. |
| Recommended next step | Keep dense shadow benchmark as model-selection evidence, but add a separate production retrieval diagnostic. |

## Risk 9: BM25 May Inject Zero-Score Candidates

| Field | Detail |
| --- | --- |
| Evidence suggesting risk | The BM25 branch sorts all scores and takes `top_bm25 = indexed_scores[:bm25_k]` without filtering scores greater than zero. |
| Why it matters | For weak or token-mismatched queries, arbitrary zero-score chunks can enter RRF, neighbor expansion, and reranking. Cross-encoder filtering may catch them, but they still consume rerank pool slots and latency. |
| How to verify | Log BM25 score distributions for queries with low lexical overlap and count how many top-BM25 candidates have `bm25_score == 0.0`. |
| Recommended next step | Filter zero-score BM25 candidates, or include a diagnostic counter before changing behavior. |

## Risk 10: Query Expansion Order Is Partly Nondeterministic

| Field | Detail |
| --- | --- |
| Evidence suggesting risk | `expand_query()` stores variants in a Python `set`, then iterates it while capping at `max_queries`. |
| Why it matters | The original query is forced first, but which additional variants survive the cap can vary across processes because set order is not stable. That makes retrieval diagnostics and benchmarks less reproducible. |
| How to verify | Run `expand_query()` in multiple fresh Python processes and compare variant order for the same query. |
| Recommended next step | Use an ordered list with explicit de-duplication instead of a set. |

## Risk 11: Multi-Query Retrieval Does Not Reward Repeated Hits Across Expansions

| Field | Detail |
| --- | --- |
| Evidence suggesting risk | The merge step keeps the candidate with the highest per-query RRF score for each chunk rather than summing or otherwise boosting chunks retrieved by multiple expanded queries. |
| Why it matters | A chunk that appears consistently across query expansions may not get a stability boost over a chunk that appears strongly once. |
| How to verify | Compare merged rankings with max-RRF versus summed-RRF on a small query set and inspect whether stable clinically relevant chunks move up. |
| Recommended next step | Treat as a benchmarked retrieval experiment, not a quick correctness fix. |

## Risk 12: Python 3.14 Bytecode Indicates Environment Drift

| Field | Detail |
| --- | --- |
| Evidence suggesting risk | `backend/**/__pycache__` and `tests/__pycache__` contain `cpython-314` bytecode alongside `cpython-312`. |
| Why it matters | The project history specifically mentions Python 3.14 / IDE-hosted crashes. Bytecode itself is not usually harmful, but it signals the old environment has been used recently in the tree. |
| How to verify | Run all commands from `.venv312`, and optionally clean `__pycache__` directories in a deliberate housekeeping change. |
| Recommended next step | Add a `.gitignore`/cleanup check and document `.venv312` as the only supported local runtime. |

## Risk 13: Dynamic Gemini Model Discovery Happens at Import Time

| Field | Detail |
| --- | --- |
| Evidence suggesting risk | `PROD_MODEL, BACKUP_MODEL = get_dynamic_models(...)` runs during `backend/llm/llm.py` import. |
| Why it matters | Import-time network calls or key availability can affect startup latency and test behavior. The code catches failures and falls back to defaults, but operators may not know which model was selected. |
| How to verify | Inspect startup logs with keys present and absent. Measure import/startup latency. |
| Recommended next step | Log chosen models clearly and consider lazy discovery or a startup hook if import behavior becomes problematic. |
