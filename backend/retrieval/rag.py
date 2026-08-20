# backend/rag.py (v3 - Hybrid FAISS + BM25 + rerank + filtering + logging)
import json
import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path

import faiss
from rank_bm25 import BM25Okapi

from backend.config import RERANK_SCORE_THRESHOLD, RRF_K
from backend.retrieval.embeddings import embed_text
from backend.retrieval.rerank import rerank

# Load chunks metadata
BASE_DIR = Path(__file__).resolve().parents[2]

VECTORSTORE_DIR = BASE_DIR / "artifacts" / "vectorstore"

CHUNKS_PATH = VECTORSTORE_DIR / "chunks.json"
INDEX_PATH = VECTORSTORE_DIR / "index.faiss"

LOG_DIR = BASE_DIR / "artifacts" / "retrieval_logs"

log = logging.getLogger(__name__)


# --- Tokenizer (shared by the BM25 index and query time) ---
#
# Splitting on whitespace alone made "pancreatitis," and "pancreatitis"
# different terms, so every sentence-final word in the corpus became its own
# token and matched nothing.  Splitting on word characters instead keeps
# hyphenated and numeric clinical terms usable ("BUN", "16,000", "pH", "7.30").
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,/][a-z0-9]+)*")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


# --------------------------------------------------------------------
# LAZY VECTORSTORE LOADING
# --------------------------------------------------------------------
# These three used to be built at module scope, so `import backend.retrieval.rag`
# parsed a 33 MB chunk registry, read the FAISS index, and built a BM25 index
# over the entire corpus — about 13 seconds, paid by every test run, every
# diagnostic script, and every import that only wanted a helper function.
#
# They are now built on first access.  Module-level `rag.index`, `rag.chunks`
# and `rag.bm25` still work via PEP 562 __getattr__, so existing call sites and
# `patch.object(rag, "chunks", ...)` in tests are unaffected.
_load_lock = threading.Lock()
_state: dict[str, object] = {}


def _load() -> dict[str, object]:
    """Load chunks, the FAISS index, and the BM25 index exactly once."""
    if _state:
        return _state
    with _load_lock:
        if _state:
            return _state

        try:
            with open(CHUNKS_PATH, encoding="utf-8") as f:
                loaded_chunks = json.load(f)
        except Exception as e:
            log.error("Error loading chunks metadata: %s", e)
            loaded_chunks = []

        try:
            loaded_index = faiss.read_index(str(INDEX_PATH))
            log.info("FAISS index loaded (ntotal=%d, dim=%d).", loaded_index.ntotal, loaded_index.d)
        except Exception as e:
            log.error("Error loading FAISS index: %s", e)
            loaded_index = None

        try:
            corpus = [_tokenize(c.get("text", "")) for c in loaded_chunks]
            loaded_bm25 = BM25Okapi(corpus) if corpus else None
            if loaded_bm25 is not None:
                log.info("BM25 index built over %d chunks.", len(corpus))
        except Exception as e:
            log.error("Error building BM25 index: %s", e)
            loaded_bm25 = None

        _state.update(chunks=loaded_chunks, index=loaded_index, bm25=loaded_bm25)
        return _state


def warmup() -> None:
    """Build the vectorstore ahead of the first request (called at startup)."""
    _load()


_MISSING = object()


def _get(name: str):
    """Return the lazily-loaded value, unless a test patched the module attr.

    ``patch.object(rag, "chunks", ...)`` writes into the module dict, and that
    must keep winning over the lazily-loaded corpus.
    """
    patched = globals().get(name, _MISSING)
    if patched is not _MISSING:
        return patched
    return _load()[name]


def _chunks():
    return _get("chunks")


def _index():
    return _get("index")


def _bm25():
    return _get("bm25")


def __getattr__(name: str):
    """Expose ``rag.chunks`` / ``rag.index`` / ``rag.bm25`` lazily (PEP 562).

    ``__getattr__`` is only consulted for names *not* already in the module
    dict, so once a test does ``patch.object(rag, "chunks", ...)`` the patched
    value wins and no load is triggered.
    """
    if name in ("chunks", "index", "bm25"):
        return _load()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# --- Filtering utilities ---
def is_low_value_text(text: str) -> bool:
    """
    Return True if text is likely low-value (figure captions, references, very short lines).
    Tweak rules as needed.
    """
    if not text or len(text.strip()) < 20:
        return True
    t = text.strip().lower()
    # Caption / front-matter markers.  "table" is deliberately NOT here: table
    # titles and bodies hold the scoring systems and diagnostic criteria this
    # system exists to reproduce, and dropping short table rows defeats
    # table-aware chunking.
    #
    # Matched on word boundaries, not as substrings — plain `"table" in t`
    # also fired on "treatable", "predictable", "preventable", "intractable".
    low_markers = [
        r"figure",
        r"fig\.",
        r"references",
        r"bibliography",
        r"copyright",
        r"reproduced with permission",
    ]
    if len(t) < 300:  # only short chunks look like captions/front-matter
        for m in low_markers:
            if re.search(rf"\b{m}", t):
                return True
    # many one-line numeric-only strings (page headers) -> ignore
    if all(ch.isdigit() or ch.isspace() or ch in ".,;:-/()" for ch in t):
        return True
    return False


def _hybrid_candidates(
    query: str,
    k: int,
    bm25_k: int,
) -> list[dict]:
    """
    Run FAISS + BM25, merge and deduplicate candidates by chunk_id,
    and compute RRF scores based on individual ranks.
    """
    candidates_by_id: dict[int, dict] = {}

    # --- FAISS branch (vector search) ---
    index = _index()
    chunks = _chunks()
    if index is not None:
        q_emb = embed_text(query)
        distances, ids = index.search(q_emb, k)
        for rank, (dist, idx) in enumerate(zip(distances[0], ids[0], strict=True), start=1):
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
                "faiss_rank": rank,
            }
            if existing is None:
                candidates_by_id[idx] = base
            else:
                # keep the best (smallest) distance and best rank; preserve text/page
                if base["distance"] < existing.get("distance", float("inf")):
                    existing.update(base)
                prev_rank = existing.get("faiss_rank")
                if prev_rank is None or rank < prev_rank:
                    existing["faiss_rank"] = rank

    # --- BM25 branch (lexical search) ---
    bm25 = _bm25()
    if bm25 is not None and chunks:
        query_tokens = _tokenize(query)
        if query_tokens:
            scores = bm25.get_scores(query_tokens)
            # get top bm25_k doc indices by score
            indexed_scores = list(enumerate(scores))
            indexed_scores.sort(key=lambda x: x[1], reverse=True)
            # A zero BM25 score means no lexical overlap.  Including those
            # arbitrary documents pollutes the RRF/reranker pool on weak
            # queries without adding lexical evidence.
            top_bm25 = [item for item in indexed_scores if item[1] > 0][:bm25_k]
            for rank, (idx, score) in enumerate(top_bm25, start=1):
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
                    "bm25_rank": rank,
                }
                if existing is None:
                    candidates_by_id[idx] = bm25_info
                else:
                    # augment existing with best bm25_score and rank
                    prev_score = existing.get("bm25_score", float("-inf"))
                    if score > prev_score:
                        existing["bm25_score"] = float(score)
                    prev_rank = existing.get("bm25_rank")
                    if prev_rank is None or rank < prev_rank:
                        existing["bm25_rank"] = rank

    # --- RRF fusion ---
    for cand in candidates_by_id.values():
        rrf_score = 0.0
        faiss_rank = cand.get("faiss_rank")
        bm25_rank = cand.get("bm25_rank")
        if faiss_rank is not None:
            rrf_score += 1.0 / (RRF_K + faiss_rank)
        if bm25_rank is not None:
            rrf_score += 1.0 / (RRF_K + bm25_rank)
        cand["rrf_score"] = rrf_score

    return list(candidates_by_id.values())


def _is_page_adjacent(page: object, neighbor_page: object, max_gap: int = 1) -> bool:
    """True when two chunks are close enough to read as continuous text."""
    try:
        return abs(int(page) - int(neighbor_page)) <= max_gap
    except (TypeError, ValueError):
        # Unknown page numbers: keep the neighbour rather than lose evidence.
        return True


def _pretrim_for_rerank(
    candidates: list[dict],
    final_k: int,
    rerank_pool: int,
) -> list[dict]:
    """
    Apply low-value filtering, RRF-based ranking, and neighbor expansion
    to produce a capped pool for reranking.
    """
    if not candidates:
        return []

    # 1) Filter low-value texts (fallback to originals if everything is filtered)
    filtered = [c for c in candidates if not is_low_value_text(c.get("text", ""))]
    if not filtered:
        filtered = candidates

    pool_size = max(final_k, rerank_pool)

    def _sort_key(c: dict) -> tuple:
        # Primary: higher RRF score
        rrf_score = c.get("rrf_score")
        if rrf_score is None:
            rrf_score = 0.0
        # Secondary: FAISS distance (smaller is better)
        dist = c.get("distance", float("inf"))
        # Tertiary: BM25 score (higher is better)
        bm = c.get("bm25_score")
        bm_component = -bm if bm is not None else 0.0
        return (-rrf_score, dist, bm_component)

    # 2) RRF-based sort and base pool selection
    filtered.sort(key=_sort_key)
    base_pool = filtered[:pool_size]

    # 3) Neighbor chunk expansion (chunk_id -1, +1) with de-duplication
    chunks = _chunks()
    by_id: dict[int, dict] = {}
    for c in base_pool:
        cid = c.get("chunk_id")
        if cid is None:
            continue
        by_id[cid] = c

    for c in base_pool:
        cid = c.get("chunk_id")
        if cid is None:
            continue
        for neighbor_id in (cid - 1, cid + 1):
            if neighbor_id < 0 or neighbor_id >= len(chunks):
                continue
            if neighbor_id in by_id:
                continue
            neighbor_chunk = chunks[neighbor_id]
            neighbor_text = neighbor_chunk.get("text", "")
            # neighbors still go through filtering
            if is_low_value_text(neighbor_text):
                continue
            # chunk_id +/- 1 is only *textually* adjacent if the two chunks sit
            # on the same or an adjacent page.  Without this check a chunk at a
            # chapter boundary pulled in unrelated material from another
            # specialty and presented it as surrounding context.
            if not _is_page_adjacent(c.get("page"), neighbor_chunk.get("page")):
                continue
            neighbor = {
                "chunk_id": neighbor_id,
                "page": neighbor_chunk.get("page"),
                "text": neighbor_text,
                # neighbors may not have distance / bm25_score; they are
                # still valid for reranking based on text alone.
                "rrf_score": c.get("rrf_score", 0.0) * 0.9,  # slightly below parent
            }
            by_id[neighbor_id] = neighbor

    expanded = list(by_id.values())

    # 4) Cap final rerank pool size for performance
    expanded.sort(key=_sort_key)
    return expanded[:rerank_pool]


# --- Main retrieve() ---
def retrieve(
    query: str,
    k: int = 30,
    final_k: int = 6,
    rerank_pool: int = 24,
    bm25_k: int = 30,
    timings: dict | None = None,
) -> list[dict]:
    """
    Hybrid retrieval for a single, already-optimized query:
    1) FAISS + BM25 hybrid retrieval with RRF fusion
    2) Filter low-value chunks + RRF sort
    3) Neighbor expansion to form the rerank pool
    4) Cross-encoder rerank to pick top final_k

    Query expansion is *not* done here.  ``backend/agents/query_optimizer.py``
    already rewrites the user's question into a clinically-framed search query
    before this function is called, and the rule-based expansion this used to
    run predated that agent.  It wrapped fixed templates around the
    already-expanded string, producing restatements like

        "clinical features, diagnosis and management of pathophysiology and
         management of acute pancreatitis in Harrison"

    which are not grammatical, differ from the original only by boilerplate
    that appears throughout the corpus, and therefore retrieve nearly the same
    neighborhood.  Fusing their ranks rewarded a chunk for scoring well against
    four restatements of one query — not the independent evidence RRF assumes.
    """

    t_start = time.perf_counter()

    if not _chunks():
        return []

    # 1) Single hybrid pass over the optimized query.
    merged_candidates = _hybrid_candidates(query, k=k, bm25_k=bm25_k)

    # 4–5) Filter + RRF sort + neighbor expansion to form rerank pool
    rerank_inputs = _pretrim_for_rerank(merged_candidates, final_k=final_k, rerank_pool=rerank_pool)

    t_retrieval_done = time.perf_counter()
    if timings is not None:
        timings["retrieval"] = t_retrieval_done - t_start

    # 6) Cross-encoder rerank
    t_rerank_start = time.perf_counter()
    top_candidates = rerank(query, rerank_inputs, top_n=final_k)
    t_rerank_done = time.perf_counter()
    if timings is not None:
        timings["reranking"] = t_rerank_done - t_rerank_start

    # 7) Drop chunks whose Cross-Encoder score is below the relevance
    #    threshold.  These are noisy candidates that slipped through
    #    FAISS/BM25 but were not actually relevant to the query.
    #    We do this AFTER rerank() so we never alter ranking math.
    pre_filter_count = len(top_candidates)
    top_candidates = [
        c for c in top_candidates
        if c.get("score") is None or float(c["score"]) >= RERANK_SCORE_THRESHOLD
    ]
    dropped_count = pre_filter_count - len(top_candidates)

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
    if _RETRIEVAL_LOGGING_ENABLED:
        _write_retrieval_log(query, merged_candidates, rerank_inputs, results, pre_filter_count, dropped_count)

    return results


# --------------------------------------------------------------------
# RETRIEVAL DIAGNOSTICS
# --------------------------------------------------------------------
# One JSON file per query, written synchronously in the request path with no
# rotation.  Two problems: the directory grew without bound, and each file
# records the user's raw clinical question, so "what did this person ask
# about" accumulated on disk forever with no retention policy.
#
# Logging is now opt-in, capped, and the query text can be withheld.
_RETRIEVAL_LOGGING_ENABLED = os.getenv("HARRISON_RETRIEVAL_LOGS", "false").strip().lower() in {"1", "true", "yes", "on"}
_RETRIEVAL_LOG_RETENTION = int(os.getenv("HARRISON_RETRIEVAL_LOG_RETENTION", "200"))
_RETRIEVAL_LOG_INCLUDE_QUERY = os.getenv("HARRISON_RETRIEVAL_LOG_QUERIES", "false").strip().lower() in {"1", "true", "yes", "on"}
_log_prune_lock = threading.Lock()


def _prune_retrieval_logs() -> None:
    """Keep only the newest _RETRIEVAL_LOG_RETENTION diagnostic files."""
    if _RETRIEVAL_LOG_RETENTION <= 0:
        return
    with _log_prune_lock:
        files = sorted(LOG_DIR.glob("*.json"))
        for stale in files[: max(0, len(files) - _RETRIEVAL_LOG_RETENTION)]:
            try:
                stale.unlink()
            except OSError:
                pass


def _write_retrieval_log(query, merged_candidates, rerank_inputs, results, pre_filter_count, dropped_count) -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        # One pass over the merged candidates instead of a linear scan per
        # result (this was an O(final_k x candidates) lookup per query).
        by_id = {c.get("chunk_id"): c for c in merged_candidates}
        log_obj = {
            "time": time.time(),
            # The raw clinical question is withheld unless explicitly enabled.
            "query": query if _RETRIEVAL_LOG_INCLUDE_QUERY else None,
            "query_chars": len(query or ""),
            "candidate_count_after_merge": len(merged_candidates),
            "candidates_count": len(merged_candidates),
            "filtered_count": len(rerank_inputs),
            "reranked_count": pre_filter_count,
            "score_threshold": RERANK_SCORE_THRESHOLD,
            "below_threshold_dropped": dropped_count,
            "final_count": len(results),
            # Verification happens later in the API/LLM layer; retrieval
            # cannot truthfully claim whether it ran.
            "verification_performed": None,
            "results": [
                {
                    "chunk_id": r["chunk_id"],
                    "page": r["page"],
                    "score": r["score"],
                    "distance": r["distance"],
                    # optional extra diagnostics if present
                    "faiss_rank": by_id.get(r["chunk_id"], {}).get("faiss_rank"),
                    "bm25_rank": by_id.get(r["chunk_id"], {}).get("bm25_rank"),
                    "rrf_score": by_id.get(r["chunk_id"], {}).get("rrf_score"),
                }
                for r in results
            ],
        }
        # A bare millisecond timestamp collides whenever two retrievals land in
        # the same millisecond — concurrent requests silently overwrote each
        # other's diagnostics.  The suffix makes each filename unique while
        # keeping the timestamp prefix sortable.
        fname = LOG_DIR / f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}.json"
        with open(fname, "w", encoding="utf-8") as f:
            json.dump(log_obj, f, ensure_ascii=False, indent=2)
        _prune_retrieval_logs()
    except Exception as e:
        # do not break retrieval if logging fails
        log.warning("Retrieval logging failed: %s", e)
