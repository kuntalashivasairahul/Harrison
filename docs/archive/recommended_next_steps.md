# Recommended Next Steps

Audit date: 2026-07-11

## High Impact / Low Risk

1. Fix the live embedding model mismatch.
   - Change `backend/retrieval/embeddings.py` from MiniLM to BGE-M3.
   - Use normalized embeddings to match `scripts/ingest_tables_aware.py`.
   - Add a lightweight startup or `/health` dimension check.
   - Verify a cache-cleared `/ask` query reaches retrieval without FAISS dimension errors.

2. Centralize model and dimension config.
   - Replace independent hardcodes in `backend/config.py`, `backend/retrieval/embeddings.py`, `scripts/ingest_tables_aware.py`, and diagnostics.
   - Store active embedding model and expected dimension in one source of truth.

3. Update production retrieval diagnostics.
   - Fix `scripts/test_retrieval.py`, `scripts/test_benchmark_step.py`, and `scripts/test_environment.py` for BGE-M3/1024 and threshold `-3.0`.
   - Rename old MiniLM benchmark tools if they remain useful historically.

4. Update docs to match current runtime.
   - Replace Groq/llama references with Gemini where code uses Gemini.
   - Replace `GROQ_API_KEY` with `GEMINI_API_KEY` / numbered keys.
   - Replace `.venv` commands with `.venv312`.
   - Replace `RERANK_SCORE_THRESHOLD=-2.0` with `-3.0`.
   - Document production artifacts as BGE-M3/1024.

5. Fix misleading observability fields.
   - Correct `expanded_query_is_static`.
   - Remove hardcoded `verification_performed=True` from retrieval logs.
   - Add `cache_hit` to timing/log metadata.

6. Repair or retire stale evaluation tests.
   - Update `evaluation/test_truncation_logic.py` for the 4-tuple `ask_llm()` return.
   - Ensure the active tests are under `tests/` and do not rely on old SDK calls.

## High Impact / Medium Risk

1. Rework semantic cache keys.
   - Include `mode`, verifier setting, embedding model/version, prompt version, and index version.
   - Consider caching only complete verified responses.
   - Clear cache during any model, prompt, or vectorstore promotion.

2. Repair `scripts/evaluate_rag.py`.
   - Use `google-genai` consistently.
   - Remove `key_manager.configure_current()`.
   - Reuse the runtime `KeyManager` client pattern or isolate judge credentials clearly.

3. Add richer safety metadata to API responses or a debug response mode.
   - `returned_path`
   - `was_truncated`
   - `cache_hit`
   - `optimizer_used`
   - `fallback_to_original_query`
   - `verifier_ran`

4. Enforce or remove the fusion context budget.
   - If enforcing, add tests that prove chunks are dropped before `SAFE_CHAR_LIMIT`.
   - If not enforcing, remove the misleading comments and constants.

5. Make promotion safer.
   - Validate staging index dimension and chunk count before copying.
   - Copy to temp paths and atomically swap.
   - Print `.venv312` restart commands.
   - Add rollback notes pointing to the backup directory.

6. Split return paths more precisely.
   - Separate `no_context`, `empty_generation`, `upstream_error`, `verified`, `draft_fallback`, and `graceful_fallback`.
   - This will make confidence capping and logs easier to interpret.

## Later Improvements

1. Add production-faithful retrieval benchmarking.
   - Import current runtime constants.
   - Record pre-rerank, post-rerank, and post-threshold results.
   - Distinguish dense retrieval quality from reranker filtering.

2. Add page image validation.
   - Sample source pages and verify resolved `storage/pages` files exist.
   - Alert on hardcoded offset drift.

3. Clean environment artifacts deliberately.
   - Remove `cpython-314` bytecode in a housekeeping commit if desired.
   - Keep `.venv312` as the documented supported runtime.

4. Improve structured logs.
   - Use JSON logs with request IDs.
   - Emit optimizer, cache, retrieval, rerank, LLM, verifier, and final response metadata in one correlated trace.

5. Revisit BM25 tokenization.
   - Current tokenization is simple lowercase split.
   - Consider punctuation handling, normalization, or medical abbreviation handling only after the embedding mismatch is fixed.

6. Tune reranker with evidence.
   - Keep `-3.0` until a production-faithful benchmark suggests otherwise.
   - Analyze score distributions by query type before changing thresholds.
