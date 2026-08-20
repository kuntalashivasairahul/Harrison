# Harrison Project Architecture Summary

Audit date: 2026-07-11

## Scope

This repository is a medical/clinical RAG system over Harrison content. The current source path uses converted Markdown (`data/harrison.md`) and table-aware ingestion rather than direct PDF ingestion.

## Core Engineering Areas

| Area | Main files | Current role |
| --- | --- | --- |
| API orchestration | `backend/api/main.py` | FastAPI app, `/ask`, `/health`, `/admin/cache`, request timing, orchestration of optimizer, cache, retrieval, LLM, confidence, and page URL resolution. |
| Retrieval | `backend/retrieval/rag.py`, `backend/retrieval/embeddings.py`, `backend/retrieval/rerank.py` | FAISS dense search, BM25 lexical search, RRF fusion, neighbor expansion, cross-encoder rerank, score threshold filtering, retrieval logs. |
| LLM pipeline | `backend/llm/llm.py`, `backend/agents/query_optimizer.py` | Gemini key rotation, dynamic model selection, query optimization, draft answer generation, verification, fallback paths. |
| Cache and routing | `backend/agents/semantic_cache.py`, `backend/agents/context_router.py` | Embedding-based response cache, post-rerank deduplication, chronological page ordering. |
| Evidence and rendering | `backend/processing/evidence.py`, `backend/rendering/page_resolver.py`, `backend/utils/fusion.py` | Fused prompt context, evidence strings, source extraction, page image URL construction. |
| Ingestion and operations | `scripts/ingest_tables_aware.py`, `scripts/test_retrieval_staging.py`, `scripts/run_benchmark.sh` | Table-aware chunking, BGE-M3 embedding build, staging output, backup and promotion. |
| Evaluation and diagnostics | `scripts/benchmark_retrieval_safe.py`, `scripts/evaluate_rag.py`, `tests/`, `evaluation/` | Retrieval benchmarks, LLM-as-judge harness, unit/regression tests. Several scripts are stale after migration. |

## End-to-End Runtime Flow

1. Client posts to `/ask` with `query`, `mode`, and optional `disable_verifier` (`backend/api/main.py:50`).
2. `optimize_query()` rewrites/classifies the query and falls back to original query on failure (`backend/agents/query_optimizer.py:242`).
3. Non-medical queries short-circuit before retrieval (`backend/api/main.py:120`).
4. Query embedding is computed with `embed_text()` and checked against `SemanticCache` (`backend/api/main.py:165`, `backend/agents/semantic_cache.py:226`).
5. `retrieve()` runs query expansion, FAISS, BM25, RRF, pre-trim, neighbor expansion, rerank, and threshold filtering (`backend/retrieval/rag.py:300`).
6. `route_and_sort_context()` deduplicates and sorts chunks by page (`backend/agents/context_router.py`).
7. `fuse_context()` builds LLM context and `extract_evidence()` builds page-cited evidence (`backend/utils/fusion.py`, `backend/processing/evidence.py`).
8. `ask_llm()` generates a draft answer and, unless disabled, calls `verify_answer()` (`backend/llm/llm.py:631`, `backend/llm/llm.py:738`).
9. `calculate_confidence()` scores the response and the API caps confidence by return path/truncation (`backend/agents/confidence_scorer.py:66`, `backend/api/main.py:250`).
10. `extract_sources()` and `resolve_page_urls()` populate source labels and page image URLs (`backend/processing/evidence.py`, `backend/rendering/page_resolver.py`).
11. The final response is cached and returned with timings (`backend/api/main.py:291`).

## Current Artifact State

Verified locally with `.venv312/bin/python` and FAISS:

| Store | Index path | Chunk path | Vectors | Dimension | Chunk count | Page sample |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Production | `artifacts/vectorstore/index.faiss` | `artifacts/vectorstore/chunks.json` | 16,983 | 1024 | 16,983 | first page 1, last page 4262 |
| Staging | `artifacts/vectorstore_staging/table_index.faiss` | `artifacts/vectorstore_staging/table_chunks.json` | 16,983 | 1024 | 16,983 | first page 1, last page 4262 |

This confirms production and staging artifacts are BGE-M3-shaped. It also means the live query embedder must produce 1024-dimensional BGE-M3 vectors.

## Active Models and Providers

| Layer | Code evidence | Current state |
| --- | --- | --- |
| Production artifacts | `scripts/ingest_tables_aware.py:144` | BGE-M3 (`BAAI/bge-m3`), 1024 dimensions. |
| Live query embedding | `backend/retrieval/embeddings.py:5` | Still hardcoded to `all-MiniLM-L6-v2`, normally 384 dimensions. This is inconsistent with production artifacts. |
| Reranker | `backend/retrieval/rerank.py` | `cross-encoder/ms-marco-MiniLM-L-6-v2`. |
| LLM generation/verification | `backend/llm/llm.py:7`, `backend/llm/llm.py:193` | Google Gemini via `google-genai`, dynamic model selection. |
| Query optimizer | `backend/agents/query_optimizer.py:49` | Uses Gemini `BACKUP_MODEL` through shared `KeyManager`. |

## Key Runtime Paths

| Path | Purpose |
| --- | --- |
| `data/harrison.md` | Converted Markdown source corpus. |
| `artifacts/vectorstore/index.faiss` | Production FAISS index loaded by `backend/retrieval/rag.py`. |
| `artifacts/vectorstore/chunks.json` | Production chunk registry loaded by `backend/retrieval/rag.py`. |
| `artifacts/vectorstore_staging/table_index.faiss` | Staging BGE-M3 index from table-aware ingestion. |
| `artifacts/vectorstore_staging/table_chunks.json` | Staging chunk registry. |
| `artifacts/vectorstore_backup/<timestamp>/` | Promotion backups created by `scripts/ingest_tables_aware.py`. |
| `artifacts/retrieval_logs/` | Per-query retrieval diagnostics. |
| `artifacts/semantic_cache.json` | Disk-persistent response cache. |
| `storage/pages/` | Static page images served at `/pages`. |

## Environment

- The stabilized environment is `.venv312` with Python 3.12.
- Operational scripts `scripts/setup_env.sh` and `scripts/run_benchmark.sh` use `.venv312`.
- Several docs and promotion messages still mention `.venv`, which is stale.
- OpenMP-sensitive scripts set `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`.

## Important Architectural Caveat

The artifact state is BGE-M3/1024, but live query embedding is MiniLM/384. A direct FAISS check showed a 384-dimensional query against the production 1024-dimensional index raises `AssertionError`, while a 1024-dimensional query works. Until `backend/retrieval/embeddings.py` is aligned with BGE-M3, production retrieval cache misses should be considered broken.
