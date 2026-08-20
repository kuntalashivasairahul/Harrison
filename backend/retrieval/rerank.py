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


def top_score(ranked_chunks: list[dict]) -> float:
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
