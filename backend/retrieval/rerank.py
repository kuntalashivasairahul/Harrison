# backend/rerank.py
import threading

from sentence_transformers import CrossEncoder

from backend.config import RERANK_MODEL

# A small, fast cross-encoder for reranking FAISS/BM25 candidates.
#
# The name comes from config so there is exactly one source of truth.  It used
# to be hardcoded here while the semantic-cache signature recorded
# config.RERANK_MODEL: changing one and not the other left the cache serving
# entries keyed to a model that was no longer running.
_RERANK_MODEL = RERANK_MODEL

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


def warmup_reranker() -> None:
    """Load the cross-encoder during application startup, not the first ask."""
    _get_reranker()

def rerank(query: str, candidates: list[dict], top_n: int = 6) -> list[dict]:
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
    # strict=: a length mismatch here would silently leave candidates unscored
    for c, s in zip(candidates, scores, strict=True):
        c["score"] = float(s)

    # sort by score desc
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_n]
