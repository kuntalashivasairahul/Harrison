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


def top_score(ranked_chunks: List[Dict]) -> float:
    """
    Return the highest Cross-Encoder ``score`` from an already-reranked chunk
    list (i.e. the output of ``rerank()``).

    The list is expected to be sorted descending by score (as ``rerank()``
    guarantees), so we just read index 0.  Falls back to iterating the whole
    list in case the caller passes an unsorted slice, and returns 0.0 when the
    list is empty or scores are absent.

    This is intentionally a **read-only** helper — it does not re-sort or
    mutate the input in any way.
    """
    if not ranked_chunks:
        return 0.0
    # Fast path: list is already sorted descending by rerank()
    best = ranked_chunks[0].get("score")
    if best is not None:
        return float(best)
    # Fallback: iterate (should not normally happen)
    scores = [c.get("score") for c in ranked_chunks if c.get("score") is not None]
    return float(max(scores)) if scores else 0.0
