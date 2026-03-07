# backend/rerank.py
from sentence_transformers import CrossEncoder
from typing import List, Dict
import threading

# Load a small fast cross-encoder suitable for reranking.
# This model is small and practical but performs very well
# for reranking candidates returned by FAISS.
# You can replace the model string later with a biomedical one if needed.
_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Load once (thread-safe helper)
_reranker = None
_reranker_lock = threading.Lock()

def _get_reranker():
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                _reranker = CrossEncoder(_RERANK_MODEL)
    return _reranker

def rerank(query: str, candidates: List[Dict], top_n: int = 6) -> List[Dict]:
    """
    candidates: list of dicts with keys {chunk_id, page, text, distance(optional)}
    Returns top_n candidates sorted by reranker score (descending), each with added 'score'.
    """
    if not candidates:
        return []

    reranker = _get_reranker()

    # Build pairs and run batch prediction
    texts = [c["text"] for c in candidates]
    pairs = [[query, t] for t in texts]

    scores = reranker.predict(pairs, show_progress_bar=False)
    # attach scores
    for c, s in zip(candidates, scores):
        c["score"] = float(s)

    # sort by score desc
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_n]
