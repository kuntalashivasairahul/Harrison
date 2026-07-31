#!/usr/bin/env python3
"""
scripts/benchmark_retrieval_safe.py
===================================
A stable, restart-safe, checkpointed version of benchmark_retrieval.py.
Enforces OMP_NUM_THREADS=1 to prevent native macOS threading crashes.
Uses a reduced workload of 5 queries and runs a fast keyword overlap grade fallback.
Compares a legacy MiniLM baseline against the configured live embedding model.
"""

import os
# Prevent OpenMP conflict crashes on macOS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
import json
import time
from pathlib import Path
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# Setup paths
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from backend.config import EMBEDDING_DIM, EMBEDDING_MODEL

CHUNKS_PATH = _ROOT / "artifacts" / "vectorstore" / "chunks.json"
INDEX_PATH = _ROOT / "artifacts" / "vectorstore" / "index.faiss"
OUTPUT_DIR = _ROOT / "artifacts" / "benchmark"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cache Paths
MINI_CORPUS_CHUNKS_PATH = OUTPUT_DIR / "mini_corpus_chunks.json"
MINI_CORPUS_INDICES_PATH = OUTPUT_DIR / "mini_corpus_indices.json"
MINILM_SHADOW_INDEX_PATH = OUTPUT_DIR / "minilm_shadow.index"
BGE_SHADOW_INDEX_PATH = OUTPUT_DIR / "bge_shadow.index"
CHECKPOINT_RESULTS_PATH = OUTPUT_DIR / "checkpoint_results.json"
REPORT_MD_PATH = OUTPUT_DIR / "benchmark_report.md"
LEGACY_MINILM_MODEL = "all-MiniLM-L6-v2"
LEGACY_MINILM_DIM = 384

# Reduced Benchmark Queries (First 5 Representative Medical Queries)
BENCHMARK_QUERIES = [
    {"id": 1, "query": "What is the initial fluid replacement rate and type for a patient in diabetic ketoacidosis?"},
    {"id": 2, "query": "What are the diagnostic criteria and amylase/lipase thresholds for acute pancreatitis?"},
    {"id": 3, "query": "What are the major and minor Duke criteria for diagnosing infective endocarditis?"},
    {"id": 4, "query": "Explain the pathophysiology of plaque rupture and thrombus formation in myocardial infarction."},
    {"id": 5, "query": "What is the emergency management protocol for severe hyperkalemia with ECG changes?"}
]

def tokenize(text: str) -> list[str]:
    return (text or "").lower().split()

def load_data():
    print("Loading corpus chunks from:", CHUNKS_PATH)
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks.")

    print("Loading production FAISS index from:", INDEX_PATH)
    index = faiss.read_index(str(INDEX_PATH))
    print(f"Loaded production FAISS index with {index.ntotal} vectors.")
    return chunks, index

def gather_mini_corpus(chunks, index, queries):
    # Check cache first
    if MINI_CORPUS_CHUNKS_PATH.exists() and MINI_CORPUS_INDICES_PATH.exists():
        print("Loading shadow mini-corpus chunks and indices from cache...")
        with open(MINI_CORPUS_CHUNKS_PATH, encoding="utf-8") as f:
            mini_corpus_chunks = json.load(f)
        with open(MINI_CORPUS_INDICES_PATH, encoding="utf-8") as f:
            mini_corpus_indices = json.load(f)
        print(f"Loaded shadow mini-corpus with {len(mini_corpus_chunks)} chunks from cache.")
        return mini_corpus_indices, mini_corpus_chunks

    print("Building full-corpus BM25 index for candidate gathering...")
    tokenized_corpus = [tokenize(c.get("text", "")) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    print(f"Loading live embedding model for initial candidate gathering: {EMBEDDING_MODEL}")
    device = "cpu"
    live_model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    mini_corpus_indices = set()

    for idx, item in enumerate(queries, start=1):
        q = item["query"]
        print(f"[{idx}/{len(queries)}] Gathering candidates for: {q[:50]}...")

        # 1. BM25 hits (top-20)
        q_tokens = tokenize(q)
        scores = bm25.get_scores(q_tokens)
        top_bm25_indices = np.argsort(scores)[::-1][:20]
        for i in top_bm25_indices:
            if scores[i] > 0.0:
                mini_corpus_indices.add(int(i))

        # 2. Live-vector hits (top-20)
        q_emb = live_model.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        if q_emb.shape[1] != index.d:
            raise RuntimeError(
                f"Live embedding dim {q_emb.shape[1]} does not match production FAISS dim {index.d}"
            )
        distances, ids = index.search(q_emb, 20)
        for i in ids[0]:
            if 0 <= i < len(chunks):
                mini_corpus_indices.add(int(i))

    mini_corpus_indices = sorted(list(mini_corpus_indices))
    mini_corpus_chunks = [chunks[idx] for idx in mini_corpus_indices]
    print(f"Formed shadow mini-corpus with {len(mini_corpus_chunks)} chunks.")

    # Save cache
    with open(MINI_CORPUS_CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(mini_corpus_chunks, f, ensure_ascii=False, indent=2)
    with open(MINI_CORPUS_INDICES_PATH, "w", encoding="utf-8") as f:
        json.dump(mini_corpus_indices, f)
    print("Saved shadow mini-corpus chunks and indices to cache.")

    return mini_corpus_indices, mini_corpus_chunks

def build_shadow_indices(mini_corpus_chunks):
    device = "cpu"
    texts = [c["text"] for c in mini_corpus_chunks]

    # 1. MiniLM Index
    if MINILM_SHADOW_INDEX_PATH.exists():
        print("Loading MiniLM shadow index from cache...")
        minilm_index = faiss.read_index(str(MINILM_SHADOW_INDEX_PATH))
        if minilm_index.d != LEGACY_MINILM_DIM:
            raise RuntimeError(
                f"Cached MiniLM shadow index dim {minilm_index.d} does not match {LEGACY_MINILM_DIM}"
            )
        print("MiniLM shadow index loaded.")
    else:
        print(f"Embedding mini-corpus with legacy baseline {LEGACY_MINILM_MODEL}...")
        minilm_model = SentenceTransformer(LEGACY_MINILM_MODEL, device=device)
        minilm_embs = minilm_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        minilm_dim = minilm_embs.shape[1]
        print(f"MiniLM dimensions verified: {minilm_dim} (Expected: {LEGACY_MINILM_DIM})")
        assert minilm_dim == LEGACY_MINILM_DIM, f"Expected {LEGACY_MINILM_DIM} dimensions for MiniLM, got {minilm_dim}"
        minilm_index = faiss.IndexFlatL2(minilm_dim)
        minilm_index.add(minilm_embs)
        faiss.write_index(minilm_index, str(MINILM_SHADOW_INDEX_PATH))
        print(f"MiniLM shadow index built and saved with {minilm_index.ntotal} vectors.")

    # 2. BGE-M3 Index
    if BGE_SHADOW_INDEX_PATH.exists():
        print("Loading BGE shadow index from cache...")
        bge_index = faiss.read_index(str(BGE_SHADOW_INDEX_PATH))
        if bge_index.d != EMBEDDING_DIM:
            raise RuntimeError(
                f"Cached live-model shadow index dim {bge_index.d} does not match {EMBEDDING_DIM}"
            )
        print("BGE shadow index loaded.")
    else:
        print(f"Embedding mini-corpus with live model {EMBEDDING_MODEL} (this may download the model)...")
        bge_model = SentenceTransformer(EMBEDDING_MODEL, device=device)
        bge_embs = bge_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        bge_dim = bge_embs.shape[1]
        print(f"Live model dimensions verified: {bge_dim} (Expected: {EMBEDDING_DIM})")
        assert bge_dim == EMBEDDING_DIM, f"Expected {EMBEDDING_DIM} dimensions for {EMBEDDING_MODEL}, got {bge_dim}"
        bge_index = faiss.IndexFlatL2(bge_dim)
        bge_index.add(bge_embs)
        faiss.write_index(bge_index, str(BGE_SHADOW_INDEX_PATH))
        print(f"BGE shadow index built and saved with {bge_index.ntotal} vectors.")

    return minilm_index, bge_index

def get_relevance_grade(query: str, chunk_text: str) -> int:
    # Quick keyword overlap heuristic as a fallback (no Gemini judge on first pass)
    q_words = set(tokenize(query)) - {
        "what", "is", "the", "and", "for", "a", "of", "in", "to", "are",
        "explain", "pathophysiology", "with", "or", "initial", "thresholds"
    }
    c_words = set(tokenize(chunk_text))
    overlap = len(q_words & c_words)
    if overlap >= 4:
        return 2
    elif overlap >= 2:
        return 1
    return 0

def run_retrieval_and_grade(queries, minilm_index, bge_index, mini_corpus_chunks, mini_corpus_indices):
    device = "cpu"

    # Load models only for query encoding
    print(f"Loading MiniLM and live model ({EMBEDDING_MODEL}) for query encoding...")
    minilm_model = SentenceTransformer(LEGACY_MINILM_MODEL, device=device)
    bge_model = SentenceTransformer(EMBEDDING_MODEL, device=device)

    # Load existing checkpoint results if any
    checkpoint_results = []
    processed_query_ids = set()
    if CHECKPOINT_RESULTS_PATH.exists():
        try:
            with open(CHECKPOINT_RESULTS_PATH, encoding="utf-8") as f:
                checkpoint_results = json.load(f)
            processed_query_ids = {r["id"] for r in checkpoint_results}
            print(f"Resuming from checkpoint. Loaded {len(checkpoint_results)} processed queries: {processed_query_ids}")
        except Exception as e:
            print(f"Error loading checkpoint file, restarting benchmark: {e}")
            checkpoint_results = []

    for idx, item in enumerate(queries, start=1):
        q_id = item["id"]
        q = item["query"]
        if q_id in processed_query_ids:
            print(f"[{idx}/{len(queries)}] Skipping already processed query ID {q_id}")
            continue

        print(f"[{idx}/{len(queries)}] Running retrieval and grading for: {q[:60]}...")

        # MiniLM Retrieval
        t0 = time.perf_counter()
        q_emb_minilm = minilm_model.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        minilm_dists, minilm_ids = minilm_index.search(q_emb_minilm, 5)
        latency_minilm = time.perf_counter() - t0

        retrieved_minilm = []
        for rank, (dist, i) in enumerate(zip(minilm_dists[0], minilm_ids[0]), start=1):
            if 0 <= i < len(mini_corpus_chunks):
                chunk = mini_corpus_chunks[i]
                orig_id = mini_corpus_indices[i]
                grade = get_relevance_grade(q, chunk["text"])
                retrieved_minilm.append({
                    "rank": rank,
                    "chunk_id": int(orig_id),
                    "page": chunk.get("page"),
                    "text": chunk["text"],
                    "distance": float(dist),
                    "relevance": grade
                })

        # BGE Retrieval
        t0 = time.perf_counter()
        q_emb_bge = bge_model.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        bge_dists, bge_ids = bge_index.search(q_emb_bge, 5)
        latency_bge = time.perf_counter() - t0

        retrieved_bge = []
        for rank, (dist, i) in enumerate(zip(bge_dists[0], bge_ids[0]), start=1):
            if 0 <= i < len(mini_corpus_chunks):
                chunk = mini_corpus_chunks[i]
                orig_id = mini_corpus_indices[i]
                grade = get_relevance_grade(q, chunk["text"])
                retrieved_bge.append({
                    "rank": rank,
                    "chunk_id": int(orig_id),
                    "page": chunk.get("page"),
                    "text": chunk["text"],
                    "distance": float(dist),
                    "relevance": grade
                })

        checkpoint_results.append({
            "id": q_id,
            "query": q,
            "minilm": {
                "latency_s": latency_minilm,
                "retrieved": retrieved_minilm
            },
            "bge": {
                "latency_s": latency_bge,
                "retrieved": retrieved_bge
            }
        })

        # Checkpoint incremental results
        with open(CHECKPOINT_RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(checkpoint_results, f, ensure_ascii=False, indent=2)
        print(f"  Query {q_id} saved to checkpoint.")

    return checkpoint_results

def compute_metrics(results):
    def calculate_ndcg5(relevances):
        dcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(relevances))
        ideal_relevances = sorted(relevances, reverse=True)
        idcg = sum((2**r - 1) / np.log2(i + 2) for i, r in enumerate(ideal_relevances))
        if idcg == 0:
            return 0.0
        return dcg / idcg

    minilm_metrics = {"hits": 0, "total_relevance": 0.0, "total_ndcg": 0.0, "total_latency": 0.0}
    bge_metrics = {"hits": 0, "total_relevance": 0.0, "total_ndcg": 0.0, "total_latency": 0.0}
    n = len(results)

    for r in results:
        # MiniLM
        minilm_rels = [c["relevance"] for c in r["minilm"]["retrieved"]]
        minilm_metrics["total_relevance"] += sum(minilm_rels) / 5.0
        minilm_metrics["total_ndcg"] += calculate_ndcg5(minilm_rels)
        minilm_metrics["total_latency"] += r["minilm"]["latency_s"]
        if any(rel == 2 for rel in minilm_rels):
            minilm_metrics["hits"] += 1

        # BGE
        bge_rels = [c["relevance"] for c in r["bge"]["retrieved"]]
        bge_metrics["total_relevance"] += sum(bge_rels) / 5.0
        bge_metrics["total_ndcg"] += calculate_ndcg5(bge_rels)
        bge_metrics["total_latency"] += r["bge"]["latency_s"]
        if any(rel == 2 for rel in bge_rels):
            bge_metrics["hits"] += 1

    report = {
        "minilm": {
            "hit_rate_5": minilm_metrics["hits"] / n if n > 0 else 0,
            "avg_relevance": minilm_metrics["total_relevance"] / n if n > 0 else 0,
            "ndcg_5": minilm_metrics["total_ndcg"] / n if n > 0 else 0,
            "avg_latency_ms": ((minilm_metrics["total_latency"] / n) * 1000.0) if n > 0 else 0
        },
        "bge": {
            "hit_rate_5": bge_metrics["hits"] / n if n > 0 else 0,
            "avg_relevance": bge_metrics["total_relevance"] / n if n > 0 else 0,
            "ndcg_5": bge_metrics["total_ndcg"] / n if n > 0 else 0,
            "avg_latency_ms": ((bge_metrics["total_latency"] / n) * 1000.0) if n > 0 else 0
        }
    }
    return report

def write_reports(results, metrics):
    md = f"""# Harrison RAG Retrieval Benchmark Report (Reduced & Stable)

This report evaluates the legacy text embedding baseline (`{LEGACY_MINILM_MODEL}`) against the configured live model (`{EMBEDDING_MODEL}`) on a shadow mini-index built from a subset of Harrison's Principles of Internal Medicine.

## Summary of Results

| Metric | Legacy (`{LEGACY_MINILM_MODEL}`) | Live (`{EMBEDDING_MODEL}`) | Improvement |
| :--- | :---: | :---: | :---: |
| **FAISS Vector Dimension** | {LEGACY_MINILM_DIM} | {EMBEDDING_DIM} | - |
| **Top-5 Hit Rate** | {metrics['minilm']['hit_rate_5']:.2%} | {metrics['bge']['hit_rate_5']:.2%} | {metrics['bge']['hit_rate_5'] - metrics['minilm']['hit_rate_5']:+.2%} |
| **Average Relevance Score (0-2)** | {metrics['minilm']['avg_relevance']:.3f} | {metrics['bge']['avg_relevance']:.3f} | {metrics['bge']['avg_relevance'] - metrics['minilm']['avg_relevance']:+.3f} |
| **NDCG@5** | {metrics['minilm']['ndcg_5']:.3f} | {metrics['bge']['ndcg_5']:.3f} | {metrics['bge']['ndcg_5'] - metrics['minilm']['ndcg_5']:+.3f} |
| **Latency per Search (ms)** | {metrics['minilm']['avg_latency_ms']:.1f} ms | {metrics['bge']['avg_latency_ms']:.1f} ms | {metrics['bge']['avg_latency_ms'] - metrics['minilm']['avg_latency_ms']:+.1f} ms |

## Insights

- **Retrieval Quality**: `{EMBEDDING_MODEL}` achieves a top-5 hit rate of **{metrics['bge']['hit_rate_5']:.2%}** compared to **{metrics['minilm']['hit_rate_5']:.2%}** for `{LEGACY_MINILM_MODEL}`.
- **Ranking Efficiency**: BGE-M3's NDCG@5 is **{metrics['bge']['ndcg_5']:.3f}** vs **{metrics['minilm']['ndcg_5']:.3f}** for MiniLM.
- **Latency Tradeoff**: The search latency for BGE-M3 is **{metrics['bge']['avg_latency_ms']:.1f} ms** compared to **{metrics['minilm']['avg_latency_ms']:.1f} ms** for MiniLM.

## Rebuild Recommendation

{"**INFORMATIONAL**: The live model shows a clear retrieval relevance and hit-rate advantage over the legacy MiniLM baseline." if metrics['bge']['hit_rate_5'] > metrics['minilm']['hit_rate_5'] + 0.05 else "**INFORMATIONAL**: The live model does not show a significant advantage over the legacy MiniLM baseline on this mini-benchmark."}

---

## Query Breakdown

"""
    for r in results:
        md += f"### Query {r['id']}: {r['query']}\n\n"
        md += "| Rank | Model | Chunk ID | Page | Rel | Snippet |\n"
        md += "| :--- | :--- | :---: | :---: | :---: | :--- |\n"
        for c in r["minilm"]["retrieved"]:
            text_preview = c["text"][:80].replace("\n", " ") + "..."
            md += f"| {c['rank']} | MiniLM | {c['chunk_id']} | {c['page']} | {c['relevance']} | {text_preview} |\n"
        for c in r["bge"]["retrieved"]:
            text_preview = c["text"][:80].replace("\n", " ") + "..."
            md += f"| {c['rank']} | BGE-M3 | {c['chunk_id']} | {c['page']} | {c['relevance']} | {text_preview} |\n"
        md += "\n"

    with open(REPORT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(md)
    print(f"Benchmark report saved to {REPORT_MD_PATH}")

def main():
    print("=" * 60)
    print("HARRISONGPT RETRIEVAL BENCHMARK (SAFE & RUNTIME STABLE)")
    print("=" * 60)

    # 1. Load full chunks and index
    chunks, index = load_data()

    # 2. Gather subset for mini-corpus
    mini_corpus_indices, mini_corpus_chunks = gather_mini_corpus(chunks, index, BENCHMARK_QUERIES)

    # 3. Embed & build separate shadow indices for both models (cached)
    minilm_index, bge_index = build_shadow_indices(mini_corpus_chunks)

    # 4. Run retrieval and grading with checkpoints
    results = run_retrieval_and_grade(BENCHMARK_QUERIES, minilm_index, bge_index, mini_corpus_chunks, mini_corpus_indices)

    # 5. Compute metrics
    metrics = compute_metrics(results)

    # 6. Save report
    write_reports(results, metrics)

    print("=" * 60)
    print("BENCHMARK RUN COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
