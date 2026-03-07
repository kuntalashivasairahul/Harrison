# HarrisonGPT – AI Agent Instructions

## Repository Rules
When analyzing this repository:

Focus only on:
- `backend/**/*.py`

Ignore:
- `artifacts/`
- `vectorstore/`
- `retrieval_logs/`
- `venv/`
- `data/`
- `__pycache__/`

Do not scan FAISS index files or retrieval log artifacts.

## Scope and Layout
- Production code lives in `backend/` and follows a modular architecture.
- `frontend/` and `data/` are placeholders.
- FastAPI entrypoint is `backend/api/main.py`.

## Runtime Architecture
Request flow is strict and linear:
`/ask` -> `retrieve()` -> `fuse_context()` -> `extract_evidence()` -> `ask_llm()`

Code modules:
- `backend/api/main.py`: API routes (`/ask`, `/health`) and request orchestration.
- `backend/retrieval/rag.py`: multi-query hybrid retrieval (FAISS + BM25), RRF, filtering, neighbor expansion, retrieval logging.
- `backend/retrieval/embeddings.py`: sentence-transformer embedding generation.
- `backend/retrieval/rerank.py`: lazy singleton cross-encoder reranking.
- `backend/utils/fusion.py`: chunk cleaning and citation-preserving context fusion.
- `backend/processing/evidence.py`: evidence extraction from retrieved chunks.
- `backend/llm/llm.py`: prompting, generation, and post-hoc verification with Groq.

Artifacts used by retrieval:
- `artifacts/vectorstore/index.faiss`
- `artifacts/vectorstore/chunks.json`
- `artifacts/retrieval_logs/*.json`

## Project Conventions
- Keep retrieval result schema stable: `chunk_id`, `page`, `text`, `distance`, `score`.
- Preserve citation marker format: `[p:{page}|c:{chunk_id}]`.
- API `mode` values are `qa` and `smart_summary`; default is `smart_summary`.
- Smart Summary has a fixed 13-section structure in `backend/llm/llm.py`; do not reorder sections unless explicitly requested.
- Retrieval logging failures must remain non-fatal.

## Run and Debug Workflows
- Start API from repository root:
  - `uvicorn backend.api.main:app --reload`
- Quick API check:
  - `curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d '{"query":"acute pancreatitis","mode":"smart_summary"}'`
- List available Groq models:
  - `python backend/list_models.py`

## Integration Points and Dependencies
- Required external service: Groq API (`GROQ_API_KEY` via `backend/.env`).
- Required local retrieval artifacts: FAISS index + chunks metadata under `artifacts/vectorstore/`.
- Core dependencies are defined in `backend/requirements.txt`.

## Safe Change Strategy
- Avoid modifying retrieval, reranking, evidence extraction, or verification logic unless explicitly requested.
- Prefer prompt-level changes in `backend/llm/llm.py` for answer style updates.
- Keep API contracts and error-handling behavior backward-compatible.
