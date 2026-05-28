# backend/rag.py (v3 - Hybrid FAISS + BM25 + rerank + filtering + logging)
import faiss
import json
import time
from pathlib import Path
from typing import Dict, List

from backend.retrieval.embeddings import embed_text
from rank_bm25 import BM25Okapi
from backend.retrieval.rerank import rerank


# Reciprocal Rank Fusion hyperparameter (standard choice ~60)
RRF_K = 60

# Minimum Cross-Encoder score a chunk must achieve to be included in the
# final response.  Chunks below this threshold are considered irrelevant
# noise that would pollute the LLM context.  The Cross-Encoder used
# (ms-marco-MiniLM-L-6-v2) produces raw logits; empirically, scores below
# -2.0 indicate near-zero relevance to the query.
# Tune upward (e.g. -1.0) for stricter filtering, downward (e.g. -3.0) to
# be more permissive.  The pipeline still returns [] gracefully if ALL
# chunks are filtered out.
RERANK_SCORE_THRESHOLD: float = -2.0


# Load chunks metadata
BASE_DIR = Path(__file__).resolve().parents[2]

VECTORSTORE_DIR = BASE_DIR / "artifacts" / "vectorstore"

CHUNKS_PATH = VECTORSTORE_DIR / "chunks.json"
INDEX_PATH = VECTORSTORE_DIR / "index.faiss"

LOG_DIR = BASE_DIR / "artifacts" / "retrieval_logs"
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


# --- Query expansion ---
def expand_query(query: str, max_queries: int = 4) -> List[str]:
    """
    Simple rule-based query expansion for multi-query retrieval.
    Always includes the original query and up to max_queries-1 variants.
    """
    q = (query or "").strip()
    if not q:
        return [q]

    variants = {q}
    lower = q.lower()

    # Basic paraphrases geared towards textbook-style retrieval
    variants.add(f"Harrison textbook explanation of {q}")
    variants.add(f"clinical features, diagnosis and management of {q} in Harrison")
    if not lower.startswith("what is"):
        variants.add(f"What is {q}?")
    variants.add(f"high-yield summary of {q} from Harrison")

    # Preserve insertion order while enforcing max_queries
    ordered: List[str] = []
    for v in variants:
        if v not in ordered:
            ordered.append(v)
        if len(ordered) >= max_queries:
            break

    # Ensure original query is first
    if q in ordered:
        ordered.remove(q)
    ordered.insert(0, q)
    return ordered[:max_queries]


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
    Run FAISS + BM25, merge and deduplicate candidates by chunk_id,
    and compute RRF scores based on individual ranks.
    """
    candidates_by_id: Dict[int, Dict] = {}

    # --- FAISS branch (vector search) ---
    if index is not None:
        q_emb = embed_text(query)
        distances, ids = index.search(q_emb, k)
        for rank, (dist, idx) in enumerate(zip(distances[0], ids[0]), start=1):
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
    if bm25 is not None and chunks:
        query_tokens = _tokenize(query)
        if query_tokens:
            scores = bm25.get_scores(query_tokens)
            # get top bm25_k doc indices by score
            indexed_scores = list(enumerate(scores))
            indexed_scores.sort(key=lambda x: x[1], reverse=True)
            top_bm25 = indexed_scores[:bm25_k]
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


def _pretrim_for_rerank(
    candidates: List[Dict],
    final_k: int,
    rerank_pool: int,
) -> List[Dict]:
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

    def _sort_key(c: Dict) -> tuple:
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
    by_id: Dict[int, Dict] = {}
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
) -> List[Dict]:
    """
    Multi-query hybrid retrieval:
    1) Expand query into up to 4 variants
    2) For each variant, run FAISS + BM25 hybrid retrieval with RRF
    3) Merge + deduplicate by chunk_id across queries
    4) Filter low-value chunks + RRF sort
    5) Neighbor expansion to form rerank pool
    6) Cross-encoder rerank to pick top final_k
    """

    if not chunks:
        return []

    # if bm25_k not explicitly set, mirror k
    if bm25_k is None:
        bm25_k = k

    # 1) Expand query (always includes original)
    expanded_queries = expand_query(query, max_queries=4)

    # 2) Run hybrid retrieval per expanded query
    per_query_candidates: List[List[Dict]] = []
    total_before_merge = 0
    for q in expanded_queries:
        cands = _hybrid_candidates(q, k=k, bm25_k=bm25_k)
        per_query_candidates.append(cands)
        total_before_merge += len(cands)

    # 3) Merge and deduplicate strictly by chunk_id across queries,
    #    keeping the best candidate metadata (based on RRF score).
    merged_by_id: Dict[int, Dict] = {}
    for cands in per_query_candidates:
        for c in cands:
            cid = c.get("chunk_id")
            if cid is None:
                continue
            existing = merged_by_id.get(cid)
            if existing is None:
                merged_by_id[cid] = c
            else:
                # choose candidate with higher RRF score (or fallback to presence of score)
                new_rrf = c.get("rrf_score")
                old_rrf = existing.get("rrf_score")
                if old_rrf is None and new_rrf is not None:
                    merged_by_id[cid] = c
                elif new_rrf is not None and old_rrf is not None and new_rrf > old_rrf:
                    merged_by_id[cid] = c

    merged_candidates = list(merged_by_id.values())
    total_after_merge = len(merged_candidates)

    # 4–5) Filter + RRF sort + neighbor expansion to form rerank pool
    rerank_inputs = _pretrim_for_rerank(merged_candidates, final_k=final_k, rerank_pool=rerank_pool)

    # 6) Cross-encoder rerank
    top_candidates = rerank(query, rerank_inputs, top_n=final_k)

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
    try:
        log_obj = {
            "time": time.time(),
            "query": query,
            "expanded_queries": expanded_queries,
            "candidate_count_before_merge": total_before_merge,
            "candidate_count_after_merge": total_after_merge,
            "candidates_count": len(merged_candidates),
            "filtered_count": len(rerank_inputs),
            "reranked_count": pre_filter_count,
            "score_threshold": RERANK_SCORE_THRESHOLD,
            "below_threshold_dropped": dropped_count,
            "final_count": len(results),
            "verification_performed": True,
            "results": [
                {
                    "chunk_id": r["chunk_id"],
                    "page": r["page"],
                    "score": r["score"],
                    "distance": r["distance"],
                    # optional extra diagnostics if present
                    "faiss_rank": next(
                        (c.get("faiss_rank") for c in merged_candidates if c.get("chunk_id") == r["chunk_id"]),
                        None,
                    ),
                    "bm25_rank": next(
                        (c.get("bm25_rank") for c in merged_candidates if c.get("chunk_id") == r["chunk_id"]),
                        None,
                    ),
                    "rrf_score": next(
                        (c.get("rrf_score") for c in merged_candidates if c.get("chunk_id") == r["chunk_id"]),
                        None,
                    ),
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
