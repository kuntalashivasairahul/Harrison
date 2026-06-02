# HarrisonGPT

HarrisonGPT is a production-grade, high-recall **Medical Retrieval-Augmented Generation (RAG) System** grounded exclusively in *Harrison's Principles of Internal Medicine* (20th+ Edition). Designed for clinicians, medical students, and exam preparation, the system prioritizes factual fidelity and citation grounding over raw creativity.

---

## 🚀 Key Features

* **Query Optimizer Agent:** Performs pre-retrieval scope checks (rejecting off-topic queries), expands medical acronyms (e.g., "MI" $\rightarrow$ "myocardial infarction"), and dynamically adapts retrieval depth based on question complexity.
* **Low-Latency Semantic Caching:** Utilizes a disk-persistent semantic cache (`artifacts/semantic_cache.json`) to serve clinically equivalent queries instantly ($\ge 0.95$ Cosine Similarity) in under ~1ms.
* **Hybrid Retrieval Pipeline:** Merges rank results from FAISS dense vector search (`all-MiniLM-L6-v2`) and BM25Okapi sparse lexical search via Reciprocal Rank Fusion (RRF), alongside local context neighbor chunk expansion.
* **Cross-Encoder Rerank Filtering:** Scores chunks via an `ms-marco-MiniLM-L-6-v2` cross-encoder, filtering out noisy passages scoring below `-2.0`.
* **Double-Pass Grounding & Verification:** Employs a post-hoc self-consistency check (`verify_answer`) to compare draft answers against raw context source page boundaries, rephrasing or redacting any claims not fully supported by the textbook.
* **Visual Grounding & Image URL Resolution:** Resolves textbook citations (e.g., `p.2787`) to static thumbnail WebP or full-resolution PNG page image URLs hosted on the local static server.

---

## 🛠️ Technology Stack

* **API & Serving:** FastAPI, Uvicorn, Pydantic, Python-dotenv
* **Vector & Lexical Search:** FAISS (CPU), Rank-BM25
* **AI Embeddings & Reranking:** SentenceTransformers (`all-MiniLM-L6-v2` and `ms-marco-MiniLM-L-6-v2`)
* **Large Language Models:** Groq Cloud SDK running `llama-3.3-70b-versatile` (synthesis & verification) and `llama-3.1-8b-instant` (query optimization)

---

## 🗺️ System Control Flow & Architecture

For a detailed walkthrough, step-by-step trace, and sequence diagram of how control flows from request intake to response delivery, refer to the [workflow.md](file:///Users/shivasairahulkuntala/Developer/AI_Projects/nlp_models/Harrison/workflow.md) file.

---

## 💻 Setup & Installation

### 1. Prerequisites
* Python 3.10+
* Virtual Environment manager (venv)

### 2. Environment Setup
Clone the repository and initialize the Python virtual environment:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt
```

### 3. Environment Variables Config
Create a `.env` file inside the `backend/` directory:
```env
GROQ_API_KEY="your-groq-cloud-api-key"

# Optional customizations
SMART_SUMMARY_MAX_TOKENS=2200
QA_MAX_TOKENS=900
SMART_SUMMARY_CONTEXT_CHAR_LIMIT=18000
```

---

## ⚙️ Running Locally

Start the ASGI development server from the workspace root:
```bash
./.venv/bin/python -m uvicorn backend.api.main:app --reload --host 127.0.0.1 --port 8000
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
    "groq_key_present": true
  }
  ```

---

## 🧪 Evaluation

The `evaluation/` directory contains tools and test configurations (e.g. `test_queries.json`) to validate the RAG pipeline recall, accuracy, and latency metrics. You can run custom test scripts inside the `.venv` shell:
```bash
python -m evaluation.run_eval
```
