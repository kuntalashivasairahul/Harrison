# backend/rag.py (v3 - Hybrid FAISS + BM25 + rerank + filtering + logging)
import faiss
import json
import os
import time
from pathlib import Path
from typing import Dict, List

from embeddings import embed_text
from rank_bm25 import BM25Okapi
from rerank import rerank


# Load chunks metadata
BASE_DIR = Path(__file__).resolve().parent
CHUNKS_PATH = BASE_DIR / "vectorstore" / "chunks.json"
INDEX_PATH = BASE_DIR / "vectorstore" / "index.faiss"
LOG_DIR = BASE_DIR / "retrieval_logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

try:
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
except Exception as e:
    print("Error loading chunks metadata:", e)
    chunks = []

# Load FAISS index
try:
    index = faiss.read_index(str(INDEX_PATH))
    print("FAISS index loaded successfully!")
except Exception as e:
    print("Error loading FAISS:", e)
    index = None


# --- BM25 index (built over chunk texts) ---
def _tokenize(text: str) -> List[str]:
    return (text or "").lower().split()


try:
    _bm25_corpus = [[* _tokenize(c.get("text", ""))] for c in chunks]
    bm25 = BM25Okapi(_bm25_corpus) if _bm25_corpus else None
    if bm25 is not None:
        print("BM25 index built successfully!")
except Exception as e:
    print("Error building BM25 index:", e)
    bm25 = None


# --- Filtering utilities ---
def is_low_value_text(text: str) -> bool:
    """
    Return True if text is likely low-value (figure captions, references, very short lines).
    Tweak rules as needed.
    """
    if not text or len(text.strip()) < 20:
        return True
    t = text.strip().lower()
    # common markers
    low_markers = [
        "figure",
        "fig.",
        "table",
        "table ",
        "references",
        "bibliography",
        "copyright",
        "reproduced with permission",
    ]
    for m in low_markers:
        if m in t and len(t) < 300:  # if it looks like a caption/figure and short
            return True
    # many one-line numeric-only strings (page headers) -> ignore
    if all(ch.isdigit() or ch.isspace() or ch in ".,;:-/()" for ch in t):
        return True
    return False


def _hybrid_candidates(
    query: str,
    k: int,
    bm25_k: int,
) -> List[Dict]:
    """
    Run FAISS + BM25, merge and deduplicate candidates by chunk_id.
    """
    candidates_by_id: Dict[int, Dict] = {}

    # --- FAISS branch (kept as before) ---
    if index is not None:
        q_emb = embed_text(query)
        distances, ids = index.search(q_emb, k)
        for dist, idx in zip(distances[0], ids[0]):
            try:
                idx = int(idx)
            except Exception:
                continue
            if idx < 0 or idx >= len(chunks):
                continue
            chunk = chunks[idx]
            text = chunk.get("text", "")
            existing = candidates_by_id.get(idx)
            base = {
                "chunk_id": idx,
                "page": chunk.get("page"),
                "text": text,
                "distance": float(dist),
            }
            if existing is None:
                candidates_by_id[idx] = base
            else:
                # keep the best (smallest) distance and ensure we don't lose text/page
                if base["distance"] < existing.get("distance", float("inf")):
                    existing.update(base)

    # --- BM25 branch ---
    if bm25 is not None and chunks:
        query_tokens = _tokenize(query)
        if query_tokens:
            scores = bm25.get_scores(query_tokens)
            # get top bm25_k doc indices by score
            indexed_scores = list(enumerate(scores))
            indexed_scores.sort(key=lambda x: x[1], reverse=True)
            top_bm25 = indexed_scores[:bm25_k]
            for idx, score in top_bm25:
                if idx < 0 or idx >= len(chunks):
                    continue
                chunk = chunks[idx]
                text = chunk.get("text", "")
                existing = candidates_by_id.get(idx)
                bm25_info = {
                    "chunk_id": idx,
                    "page": chunk.get("page"),
                    "text": text,
                    "bm25_score": float(score),
                }
                if existing is None:
                    candidates_by_id[idx] = bm25_info
                else:
                    # augment existing with bm25_score if better
                    prev_score = existing.get("bm25_score", float("-inf"))
                    if score > prev_score:
                        existing["bm25_score"] = float(score)

    return list(candidates_by_id.values())


def _pretrim_for_rerank(
    candidates: List[Dict],
    final_k: int,
    rerank_pool: int,
) -> List[Dict]:
    """
    Filter low-value texts, then choose a pool for reranking.
    Prioritise candidates that are strong in either FAISS distance or BM25.
    """
    if not candidates:
        return []

    filtered = [c for c in candidates if not is_low_value_text(c.get("text", ""))]
    if not filtered:
        filtered = candidates

    pool_size = max(final_k, rerank_pool)

    def _score_for_sort(c: Dict) -> tuple:
        has_vec = "distance" in c and c["distance"] is not None
        has_bm = "bm25_score" in c and c["bm25_score"] is not None
        # candidates that appear in both get highest priority
        both_bonus = 0
        if has_vec and has_bm:
            both_bonus = -2
        elif has_vec or has_bm:
            both_bonus = -1
        dist = c.get("distance", float("inf"))
        bm = c.get("bm25_score")
        bm_rank = -bm if bm is not None else 0.0  # higher bm25 is better
        return (both_bonus, dist, bm_rank)

    filtered.sort(key=_score_for_sort)
    return filtered[:pool_size]


# --- Main retrieve() ---
def retrieve(
    query: str,
    k: int = 30,
    final_k: int = 6,
    rerank_pool: int = 24,
    bm25_k: int = 30,
) -> List[Dict]:
    """
    Hybrid retrieval:
    1) FAISS vector search for top-k
    2) BM25 lexical search for top-bm25_k
    3) Merge + deduplicate by chunk_id
    4) Filter low-value chunks
    5) Pre-trim to a rerank pool
    6) Cross-encoder rerank to pick top final_k
    """

    if not chunks:
        return []

    # if bm25_k not explicitly set, mirror k
    if bm25_k is None:
        bm25_k = k

    # 1–3) Hybrid retrieval and deduplication
    candidates = _hybrid_candidates(query, k=k, bm25_k=bm25_k)

    # 4–5) Filter + pre-trim for reranker
    rerank_inputs = _pretrim_for_rerank(candidates, final_k=final_k, rerank_pool=rerank_pool)

    # 6) Cross-encoder rerank
    top_candidates = rerank(query, rerank_inputs, top_n=final_k)

    # Prepare final structure (ensuring types are JSON-serializable)
    results = []
    for c in top_candidates:
        results.append(
            {
                "chunk_id": int(c.get("chunk_id")),
                "page": int(c.get("page")) if c.get("page") is not None else None,
                "text": c.get("text"),
                "distance": float(c.get("distance")) if c.get("distance") is not None else None,
                "score": float(c.get("score")) if c.get("score") is not None else None,
            }
        )

    # Logging (append small json for each retrieval)
    try:
        log_obj = {
            "time": time.time(),
            "query": query,
            "candidates_count": len(candidates),
            "filtered_count": len(rerank_inputs),
            "final_count": len(results),
            "results": [
                {
                    "chunk_id": r["chunk_id"],
                    "page": r["page"],
                    "score": r["score"],
                    "distance": r["distance"],
                }
                for r in results
            ],
        }
        fname = LOG_DIR / f"{int(time.time() * 1000)}.json"
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(log_obj, f, ensure_ascii=False, indent=2)
    except Exception as e:
        # do not break retrieval if logging fails
        print("Retrieval logging failed:", e)

    return results
