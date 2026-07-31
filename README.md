# HarrisonGPT

HarrisonGPT is a production-grade, high-recall **Medical Retrieval-Augmented Generation (RAG) System** grounded exclusively in *Harrison's Principles of Internal Medicine* (20th+ Edition). Designed for clinicians, medical students, and exam preparation, the system prioritizes factual fidelity and citation grounding over raw creativity.

---

## 🚀 Key Features

* **Query Optimizer Agent:** Performs pre-retrieval scope checks (rejecting off-topic queries), expands medical acronyms (e.g., "MI" $\rightarrow$ "myocardial infarction"), and dynamically adapts retrieval depth based on question complexity.
* **Low-Latency Semantic Caching:** Utilizes a disk-persistent semantic cache (`artifacts/semantic_cache.json`) to serve clinically equivalent queries instantly ($\ge 0.95$ Cosine Similarity) in under ~1ms.
* **Hybrid Retrieval Pipeline:** Merges FAISS dense search using `BAAI/bge-m3` (1024 dimensions) with BM25Okapi sparse lexical search via Reciprocal Rank Fusion (RRF), alongside local context neighbor chunk expansion.
* **Cross-Encoder Rerank Filtering:** Scores chunks via an `ms-marco-MiniLM-L-6-v2` cross-encoder, filtering out noisy passages scoring below `-3.0`.
* **Double-Pass Grounding & Verification:** Employs a post-hoc self-consistency check (`verify_answer`) to compare draft answers against raw context source page boundaries, rephrasing or redacting any claims not fully supported by the textbook.
* **Visual Grounding & Image URL Resolution:** Resolves textbook citations (e.g., `p.2787`) to static thumbnail WebP or full-resolution PNG page image URLs hosted on the local static server.

---

## 🛠️ Technology Stack

* **API & Serving:** FastAPI, Uvicorn, Pydantic, Python-dotenv
* **Vector & Lexical Search:** FAISS (CPU), Rank-BM25
* **AI Embeddings & Reranking:** SentenceTransformers (`BAAI/bge-m3`, 1024 dimensions, and `ms-marco-MiniLM-L-6-v2`)
* **Large Language Models:** Google Gen AI SDK (`google-genai`) with dynamic Gemini model selection and key rotation.

---

## 🗺️ System Control Flow & Architecture

For a detailed walkthrough, step-by-step trace, and sequence diagram of how control flows from request intake to response delivery, refer to the [workflow.md](file:///Users/shivasairahulkuntala/Developer/AI_Projects/nlp_models/Harrison/workflow.md) file.

---

## 💻 Setup & Installation

### 1. Prerequisites
* Python 3.12
* Virtual Environment manager (venv)

### 2. Environment Setup
Clone the repository and initialize the Python virtual environment:
```bash
./scripts/setup_env.sh
# Or, after setup:
.venv312/bin/pip install -r backend/requirements.txt
```

### 3. Environment Variables Config
Create a `.env` file inside the `backend/` directory:
```env
GEMINI_API_KEY="your-google-ai-api-key"

# Optional rotation pool; numbered keys take precedence for their slots.
# GEMINI_API_KEY_1="..."
# GEMINI_API_KEY_2="..."

# Optional customizations
SMART_SUMMARY_MAX_TOKENS=3000
QA_MAX_TOKENS=3000
SMART_SUMMARY_CONTEXT_CHAR_LIMIT=12000
SMART_SUMMARY_K=48
SMART_SUMMARY_FINAL_K=12
SMART_SUMMARY_RERANK_POOL=16
```

---

## ⚙️ Running Locally

Start the ASGI development server from the workspace root:
```bash
.venv312/bin/python -m uvicorn backend.api.main:app --reload --host 127.0.0.1 --port 8000
```
The server will start at `http://127.0.0.1:8000`.

---

## 🔌 API Documentation

### 1. Ask Endpoint
* **Path:** `/ask`
* **Method:** `POST`
* **Request JSON Schema:**
  ```json
  {
    "query": "management of acute pancreatitis",
    "mode": "smart_summary"
  }
  ```
  *(Modes: `smart_summary` (default) or `qa`)*

* **Response JSON Schema:**
  ```json
  {
    "answer": "Topic received — generating Harrison Smart Summary...",
    "confidence": "High",
    "sources": ["p.2157", "p.2158"],
    "visual_context": [
      {
        "page_label": "p.2157",
        "thumbnail_url": "http://127.0.0.1:8000/pages/small/page_2114_small.webp",
        "full_url": "http://127.0.0.1:8000/pages/full/page_2114_full.png"
      }
    ]
  }
  ```

### 2. Health Endpoint
* **Path:** `/health`
* **Method:** `GET`
* **Response JSON Schema:**
  ```json
  {
    "status": "ok",
    "faiss_loaded": true,
    "chunks_loaded": true,
    "faiss_dim": 1024,
    "embedding_dim": 1024,
    "embedding_index_dim_match": true,
    "gemini_key_present": true,
    "gemini_key_count": 1
  }
  ```

---

## 🧪 Evaluation

The `evaluation/` directory contains tools and test configurations (e.g. `test_queries.json`) to validate the RAG pipeline recall, accuracy, and latency metrics. Use the project Python runtime for custom scripts:
```bash
.venv312/bin/python -m evaluation.run_eval
```

### Smart Summary Configuration

For `mode: "smart_summary"`, `SMART_SUMMARY_K` (default `48`) controls the
candidate retrieval count and `SMART_SUMMARY_RERANK_POOL` (default `16`)
controls the rerank pool. Query complexity selects a final context count of
`5` (simple) or `12` (complex), capped by `SMART_SUMMARY_FINAL_K` (default
`12`). `SMART_SUMMARY_MAX_TOKENS` controls the generation and normal
verification token ceiling for this mode. The first response line is enforced
as `Topic received — generating Harrison Smart Summary.`; headings are
generated from available evidence rather than padded with empty sections.

`SMART_SUMMARY_CONTEXT_CHAR_LIMIT` is loaded with a default of `12000`, but
the current fusion implementation applies its own fixed `SAFE_CHAR_LIMIT` of
`12000` characters to every mode. Changing the environment variable alone
does not currently change the fused-context budget.
