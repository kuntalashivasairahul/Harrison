# scripts/

Operational tools and diagnostics. **None of these are tests** — they load real
models and indexes, which is why they are named `probe_*` rather than `test_*`.
`pyproject.toml` scopes pytest collection to `tests/` so they are never
collected; a `test_*.py` here used to crash `pytest` at collection time.

Run everything with the project runtime: `.venv312/bin/python scripts/<name>.py`

## Ingestion & assets

| Script | Purpose |
|---|---|
| `ingest_tables_aware.py` | **Current ingester.** Table-aware markdown chunking → embeddings → FAISS index. Writes to `artifacts/vectorstore_staging/`. |
| `convert_pdf.py` | One-off: renders the source PDF to the page images under `storage/pages/`. |

## Evaluation & benchmarking

| Script | Purpose |
|---|---|
| `evaluate_rag.py` | Evaluation harness. **RULE 5.4 requires running this before and after any retrieval-parameter change.** Needs a running API and Gemini credentials. |
| `benchmark_retrieval.py` | Retrieval benchmark against the production index. |
| `benchmark_retrieval_safe.py` | Lower-memory benchmark variant. |
| `run_benchmark.sh` | Driver for the benchmark scripts. |

## Diagnostics

| Script | Purpose |
|---|---|
| `probe_retrieval.py` | Trace one query through every retrieval stage. `--staging` targets the staging vectorstore. |
| `probe_environment.py` | Verify the runtime: model loads, embedding dimension, index dimension match. |
| `probe_import_order.py` | Verify torch/sentence-transformers and FAISS coexist in one process. `--cpu` forces CPU. Import order between these has caused native crashes on Apple Silicon. |
| `probe_benchmark_step.py` | Single benchmark step in isolation. |
| `probe_st.py` | Minimal sentence-transformers smoke test. |
| `probe_faiss_st.py` | Minimal FAISS + sentence-transformers smoke test. |

## Deployment

| Script | Purpose |
|---|---|
| `stage_corpus.sh` | Assembles exactly the files the private HF dataset needs (index, chunks, WebP thumbnails) into an upload directory, ~549 MB. Refuses to stage a git-lfs pointer, `data/`, or the 3.8 GB full-res renders. Exists because `hf upload <repo> .` from the repo root would upload roughly 5 GB. |
| `sync_space.sh` | Assembles a Hugging Face Space checkout from an allowlist, then verifies what landed: aborts on `data/`, `storage/`, `artifacts/`, `backend/.env`, or any file over 5 MB. Exists because the licensed corpus is tracked in git history, so adding the Space as a remote and pushing would publish the textbook. |
| `fetch_corpus.py` | Pulls `artifacts/vectorstore/` and `storage/pages/small/` from a private HF dataset before the API starts. Run by `entrypoint.sh`, never by the app: `backend/api/main.py` mounts `StaticFiles("storage/pages")` at import, so this must finish before that import happens, and import-time purity forbids it living under `backend/`. No-op when the corpus is already on disk. |

## Environment setup

| Script | Purpose |
|---|---|
| `setup_env.sh` / `setup_env.ps1` | Create `.venv312` and install runtime + dev dependencies. Warns if `git-lfs` is missing. |

## Removed

- `ingest_tables.py` — superseded by `ingest_tables_aware.py`, which its own
  docstring calls the "production-grade successor". In git history if needed.
- `probe_import_order_cpu.py` — differed from `probe_import_order.py` by one
  device string; now the `--cpu` flag.
- `probe_retrieval_staging.py` — differed from `probe_retrieval.py` by two
  paths; now the `--staging` flag.
