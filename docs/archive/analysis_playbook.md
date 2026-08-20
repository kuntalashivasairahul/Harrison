# Harrison Analysis Playbook

Audit date: 2026-07-11

This playbook is for a new engineer or coding agent auditing the project safely.

## Ground Rules

- Use `.venv312`, not `.venv`.
- Do not rewrite first. Inspect paths, active models, artifact dimensions, and runtime flow.
- Treat `artifacts/vectorstore/`, `artifacts/retrieval_logs/`, and `storage/pages/` as data/artifact areas. Inspect metadata only when needed.
- Separate retrieval failures from reranking failures, generation failures, verifier failures, and cache effects.
- Clear or bypass cache before judging live behavior.
- Restart Uvicorn after any vectorstore promotion, because `backend/retrieval/rag.py` loads FAISS and chunks at import time.

## First Commands

```bash
source .venv312/bin/activate
which python
python --version
git status --short
```

Check production and staging artifact metadata:

```bash
python - <<'PY'
from pathlib import Path
import json
import faiss

root = Path.cwd()
for name, idx_path, chunks_path in [
    ("production", root / "artifacts/vectorstore/index.faiss", root / "artifacts/vectorstore/chunks.json"),
    ("staging", root / "artifacts/vectorstore_staging/table_index.faiss", root / "artifacts/vectorstore_staging/table_chunks.json"),
]:
    print(f"[{name}]")
    idx = faiss.read_index(str(idx_path))
    with chunks_path.open(encoding="utf-8") as f:
        chunks = json.load(f)
    print("index_dim:", idx.d)
    print("index_ntotal:", idx.ntotal)
    print("chunk_count:", len(chunks))
    print("first_keys:", sorted(chunks[0].keys()) if chunks else [])
PY
```

Current expected result after BGE-M3 promotion:

- production `index_dim = 1024`
- staging `index_dim = 1024`
- both stores `ntotal = 16983`
- both chunk registries `chunk_count = 16983`

## Files to Read First

1. `backend/api/main.py`
   - Request/response schema.
   - `/ask` control flow.
   - cache hit path.
   - confidence capping.
   - `/health` checks.
2. `backend/retrieval/rag.py`
   - artifact load paths.
   - FAISS/BM25/RRF flow.
   - rerank threshold.
   - retrieval diagnostics.
3. `backend/retrieval/embeddings.py`
   - active query embedding model.
   - normalization behavior.
4. `scripts/ingest_tables_aware.py`
   - active ingestion embedding model/dimension.
   - staging and production paths.
   - `--promote-only`.
   - backup behavior.
5. `backend/llm/llm.py`
   - Gemini key manager.
   - model discovery.
   - draft and verifier paths.
   - return paths.
6. `backend/agents/confidence_scorer.py`
   - calibrated confidence thresholds.
7. `backend/agents/semantic_cache.py`
   - cache key and stored response shape.

## Do Not Trust These Until Updated

- `README.md` setup/runtime/model sections.
- `ARCHITECTURE.md` threshold and Groq sections.
- `PROJECT_CONTEXT.md` runtime constants and provider sections.
- `scripts/test_retrieval.py` production diagnostics.
- `scripts/test_benchmark_step.py`.
- `scripts/test_environment.py`.
- `scripts/evaluate_rag.py`.
- `evaluation/test_truncation_logic.py`.

## Runtime Checks After Fixing Embeddings

Start server:

```bash
source .venv312/bin/activate
python -m uvicorn backend.api.main:app --reload --host 127.0.0.1 --port 8000
```

Health:

```bash
curl http://127.0.0.1:8000/health
```

Clear cache before live tests:

```bash
curl -X DELETE http://127.0.0.1:8000/admin/cache
```

Retrieval/generation smoke tests:

```bash
curl -s http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"Ranson criteria for acute pancreatitis at admission","mode":"qa","disable_verifier":true}'

curl -s http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"Pathophysiology of acute pancreatitis","mode":"smart_summary"}'
```

Use `disable_verifier=true` only to isolate retrieval and draft generation. Do not use it as a production safety shortcut.

## Distinguishing Failure Sources

### Retrieval Failure

Likely signs:

- FAISS dimension assertion.
- `retrieval` timing present but zero final chunks.
- retrieval log `final_count = 0`.
- BM25 works but dense branch fails.

Where to inspect:

- `backend/retrieval/embeddings.py`
- `backend/retrieval/rag.py`
- latest `artifacts/retrieval_logs/*.json`

First checks:

- Does query embedding dimension equal `index.d`?
- Does `chunks.json` length equal `index.ntotal`?
- Are page and text keys present in chunks?

### Reranking or Threshold Failure

Likely signs:

- Candidates exist before rerank.
- Many chunks are dropped by threshold.
- `below_threshold_dropped` is high.
- `score_threshold` is `-3.0`.

Where to inspect:

- `backend/retrieval/rerank.py`
- `backend/retrieval/rag.py:373`
- retrieval logs: `reranked_count`, `below_threshold_dropped`, `final_count`, per-result `score`.

How to verify:

- Compare top candidates before threshold versus after threshold.
- Temporarily diagnose with scripts only; do not change threshold without evaluation evidence.

### Generation Failure

Likely signs:

- Good chunks and context, but answer is empty, generic, or `error_fallback`.
- `draft_generation` timing exists but response path is `error_fallback`.
- Gemini quota or API errors in logs.

Where to inspect:

- `backend/llm/llm.py`
- `ask_llm: return_path=...` logs.
- `KeyManager` logs.

### Verification Failure

Likely signs:

- Draft answer exists but final answer falls back to draft.
- `returned_path = draft_fallback` or `graceful_fallback`.
- `verification` and `retry` timings are large or errors appear.

Where to inspect:

- `backend/llm/llm.py:738`
- API confidence caps in `backend/api/main.py:250`
- tests in `tests/test_pipeline_e2e.py`.

### Cache Confusion

Likely signs:

- Fast response with timing zeros for skipped stages.
- `disable_verifier` appears to have no effect.
- `qa` and `smart_summary` return same style for same query.

Where to inspect:

- `backend/api/main.py:167`
- `backend/agents/semantic_cache.py:226`
- `DELETE /admin/cache`.

## Safe Promotion Procedure

1. Build staging:

```bash
source .venv312/bin/activate
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/ingest_tables_aware.py
```

2. Validate staging metadata:

```bash
python scripts/test_retrieval_staging.py --query "Ranson criteria for acute pancreatitis"
```

Note: update this script if its threshold/comments drift from production.

3. Promote only:

```bash
python scripts/ingest_tables_aware.py --promote-only
```

4. Restart server.

5. Check `/health`, clear cache, and run one `qa` and one `smart_summary` query.

## Reporting Template

When filing issues, label them as:

- Confirmed issue: directly proven by code, command output, or reproducible behavior.
- Likely issue/risk: plausible from evidence, but needs verification.
- Stale assumption: docs/scripts/comments disagree with runtime code.
- Optional improvement: useful but not required to restore correctness.
