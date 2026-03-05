# HarrisonGPT – AI Agent Instructions

## Scope and layout
- Primary implementation is in `backend/`; `frontend/` and `data/` are currently empty placeholders.
- Ignore `backend/venv/` and `backend/__pycache__/` when analyzing or editing code.
- Backend is a single FastAPI service with one endpoint in `backend/main.py`.

## Runtime architecture (read this before edits)
- Request flow is strict and linear: `/ask` → `retrieve()` → `fuse_context()` → `ask_llm()`.
- Retrieval lives in `backend/rag.py` and uses FAISS + metadata from local files:
	- `backend/vectorstore/index.faiss`
	- `backend/vectorstore/chunks.json`
- `backend/rag.py` performs import-time loading of FAISS index and chunk metadata; path handling is relative to the backend working directory.
- `backend/embeddings.py` loads `SentenceTransformer("all-MiniLM-L6-v2")` at import time.
- `backend/rerank.py` uses lazy singleton loading for `CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")`.
- `backend/llm.py` loads `.env` from `backend/.env` and calls Groq chat completions.

## Project-specific conventions
- Keep retrieval result schema stable (`chunk_id`, `page`, `text`, `distance`, `score`) because downstream fusion depends on these keys.
- Preserve citation marker format from `backend/fusion.py`: `[p:{page}|c:{chunk_id}]`.
- `mode` values are `qa` and `smart_summary`; default is `smart_summary` in `backend/main.py`.
- Smart Summary prompt in `backend/llm.py` has a fixed 13-section structure; do not alter section order unless explicitly requested.
- Retrieval logging writes per-query JSON to `backend/retrieval_logs/`; failures are intentionally non-fatal.

## Run and debug workflows
- Start API from backend directory so relative vectorstore/log paths resolve:
	- `cd backend`
	- `uvicorn main:app --reload`
- Quick API check:
	- `curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d '{"query":"acute pancreatitis","mode":"smart_summary"}'`
- List available Groq models:
	- `python list_models.py`

## Integration points and dependencies
- Required external service: Groq API (`GROQ_API_KEY` in `backend/.env`).
- Required local retrieval artifacts: `index.faiss` + `chunks.json` under `backend/vectorstore/`.
- ML dependencies used directly in code: `faiss`, `sentence-transformers`, `numpy`, `groq`, `python-dotenv`, `fastapi`.
- `backend/requirements.txt` is currently empty; if dependency changes are made, update this file as part of the same task.

## Safe change strategy for agents
- When changing retrieval quality, edit `is_low_value_text()` and/or rerank selection logic first; avoid changing API contract.
- When changing answer style, prefer prompt edits in `backend/llm.py` over post-processing.
- Keep error handling non-breaking for retrieval/logging paths to preserve API availability.
