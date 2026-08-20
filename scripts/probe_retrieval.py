#!/usr/bin/env python3
"""
scripts/probe_retrieval.py
=========================
Standalone diagnostic script for the HarrisonGPT retrieval pipeline.

What it does
------------
1.  Loads artifacts/vectorstore/chunks.json and index.faiss directly.
2.  Builds a BM25 index (identical tokenisation to rag.py).
3.  Runs FAISS semantic search (k=50) for the diagnostic query.
4.  Runs BM25 lexical search (k=50) for the same query.
5.  Applies RRF fusion (RRF_K=60) — same as the live pipeline.
6.  Loads the Cross-Encoder (ms-marco-MiniLM-L-6-v2) and reranks the
    top 24 RRF candidates, printing scores with the RERANK_SCORE_THRESHOLD
    marked clearly.
7.  Scans ALL chunks for "Ranson" / "Age > 55" / "ranson" and reports each
    hit's chunk_id, page, chunk_type, FAISS rank, BM25 rank, RRF score,
    and Cross-Encoder score.

Usage
-----
    python scripts/probe_retrieval.py
    python scripts/probe_retrieval.py --staging
    python scripts/probe_retrieval.py --query "Ranson criteria pancreatitis" --k 50
    python scripts/probe_retrieval.py --search-term "Glasgow" --no-rerank

Flags
-----
    --query TEXT        Retrieval query (default: built-in Ranson query)
    --k N               FAISS / BM25 candidate pool size (default: 50)
    --rerank-pool N     Candidates passed to Cross-Encoder (default: 24)
    --search-term TEXT  String to brute-force scan for (default: "Ranson")
    --no-rerank         Skip Cross-Encoder step (faster but less diagnostic)
    --device DEVICE     Embedding device: mps | cpu (default: auto-detect)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

# ---------------------------------------------------------------------------
# Paths — same as rag.py
# ---------------------------------------------------------------------------

_ROOT        = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.config import (
    DEFAULT_RERANK_POOL,
    EMBEDDING_DIM,
    EMBEDDING_MODEL,
    RERANK_MODEL,
    RERANK_SCORE_THRESHOLD,
    RRF_K,
)

# Production and staging differ only in where the index lives, so one script
# with a --staging flag replaces the two 515-line near-copies this used to be
# (probe_retrieval.py and probe_retrieval_staging.py differed in two lines).
_VECTORSTORES = {
    "production": ("vectorstore", "chunks.json", "index.faiss"),
    "staging":    ("vectorstore_staging", "table_chunks.json", "table_index.faiss"),
}

# Rebound by main() when --staging is passed.
VECTORSTORE = "production"
CHUNKS_PATH = _ROOT / "artifacts" / "vectorstore" / "chunks.json"
INDEX_PATH  = _ROOT / "artifacts" / "vectorstore" / "index.faiss"


def select_vectorstore(name: str) -> None:
    """Point CHUNKS_PATH / INDEX_PATH at the production or staging store."""
    global VECTORSTORE, CHUNKS_PATH, INDEX_PATH
    directory, chunks_file, index_file = _VECTORSTORES[name]
    VECTORSTORE = name
    CHUNKS_PATH = _ROOT / "artifacts" / directory / chunks_file
    INDEX_PATH = _ROOT / "artifacts" / directory / index_file

# ---------------------------------------------------------------------------
# Pipeline constants — must match rag.py exactly
# ---------------------------------------------------------------------------

DEFAULT_QUERY          = "Ranson criteria for acute pancreatitis at admission"
DEFAULT_K              = 50
DEFAULT_SEARCH_TERM    = "Ranson"

# ---------------------------------------------------------------------------
# ANSI colour helpers
# ---------------------------------------------------------------------------

RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
DIM    = "\033[2m"
WHITE  = "\033[97m"

def _h(text: str) -> str:   return f"{BOLD}{CYAN}{text}{RESET}"
def _ok(text: str) -> str:  return f"{GREEN}{text}{RESET}"
def _warn(text: str) -> str: return f"{YELLOW}{text}{RESET}"
def _err(text: str) -> str: return f"{RED}{text}{RESET}"
def _dim(text: str) -> str: return f"{DIM}{text}{RESET}"

def _hr(char: str = "─", width: int = 72) -> None:
    print(f"{DIM}{char * width}{RESET}")

def _section(title: str) -> None:
    print()
    _hr("═")
    print(f"  {BOLD}{CYAN}{title}{RESET}")
    _hr("═")

def _score_colour(score: float) -> str:
    if score >= RERANK_SCORE_THRESHOLD:
        return f"{GREEN}{score:+.3f}{RESET}"
    return f"{RED}{score:+.3f} ← BELOW THRESHOLD{RESET}"

# ---------------------------------------------------------------------------
# Tokeniser (identical to rag.py)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    return (text or "").lower().split()

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_chunks() -> List[Dict]:
    if not CHUNKS_PATH.exists():
        print(_err(f"✗ chunks.json not found: {CHUNKS_PATH}"))
        sys.exit(1)
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        chunks = json.load(f)
    print(_ok(f"✓ Loaded {len(chunks):,} chunks  ({CHUNKS_PATH.stat().st_size / 1e6:.1f} MB)"))

    # Audit page=0 immediately
    zero = sum(1 for c in chunks if c.get("page", 0) == 0)
    if zero:
        print(_warn(f"  ⚠ {zero:,} chunks still have page=0 "
                    f"({zero / len(chunks) * 100:.1f}% of corpus)"))
    else:
        print(_ok("  ✓ Zero page=0 chunks — page tracking is healthy"))

    # Count chunk types if available
    tables = sum(1 for c in chunks if c.get("chunk_type") == "table")
    prose  = sum(1 for c in chunks if c.get("chunk_type") == "prose")
    if tables or prose:
        print(f"  Chunk types: {_ok(str(tables))} table | {_dim(str(prose))} prose "
              f"| {len(chunks) - tables - prose} other")
    return chunks


def load_faiss_index() -> faiss.IndexFlatL2:
    if not INDEX_PATH.exists():
        print(_err(f"✗ index.faiss not found: {INDEX_PATH}"))
        sys.exit(1)
    idx = faiss.read_index(str(INDEX_PATH))
    print(_ok(f"✓ FAISS index loaded  ({idx.ntotal:,} vectors, dim={idx.d})"))
    return idx


def build_bm25(chunks: List[Dict]) -> BM25Okapi:
    corpus = [_tokenize(c.get("text", "")) for c in chunks]
    bm25 = BM25Okapi(corpus)
    print(_ok(f"✓ BM25 index built  ({len(corpus):,} documents)"))
    return bm25


def _model_dimension(model: SentenceTransformer) -> int:
    if hasattr(model, "get_embedding_dimension"):
        return int(model.get_embedding_dimension())
    return int(model.get_sentence_embedding_dimension())


def load_embedding_model(device: str) -> SentenceTransformer:
    print(f"  Loading SentenceTransformer('{EMBEDDING_MODEL}') on {device.upper()}…", flush=True)
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    model_dim = _model_dimension(model)
    print(_ok(f"✓ Embedding model ready  (device={device.upper()}, dim={model_dim})"))
    if model_dim != EMBEDDING_DIM:
        print(_err(
            f"✗ Embedding model dimension mismatch: config expects {EMBEDDING_DIM}, "
            f"model reports {model_dim}"
        ))
        sys.exit(1)
    return model


def load_cross_encoder() -> CrossEncoder:
    print(f"  Loading CrossEncoder('{RERANK_MODEL}')…", flush=True)
    ce = CrossEncoder(RERANK_MODEL)
    print(_ok(f"✓ Cross-Encoder ready"))
    return ce

# ---------------------------------------------------------------------------
# Search functions
# ---------------------------------------------------------------------------

def faiss_search(
    query: str,
    index: faiss.IndexFlatL2,
    model: SentenceTransformer,
    k: int,
) -> List[Tuple[int, float]]:
    """Returns [(chunk_idx, l2_distance), …] sorted by ascending distance."""
    q_vec = model.encode([query], normalize_embeddings=True).astype("float32")
    dists, ids = index.search(q_vec, k)
    return [(int(ids[0][i]), float(dists[0][i])) for i in range(len(ids[0])) if ids[0][i] >= 0]


def bm25_search(
    query: str,
    bm25: BM25Okapi,
    k: int,
) -> List[Tuple[int, float]]:
    """Returns [(chunk_idx, bm25_score), …] sorted by descending score."""
    tokens = _tokenize(query)
    if not tokens:
        return []
    scores = bm25.get_scores(tokens)
    indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    return [(idx, float(score)) for idx, score in indexed[:k]]


def rrf_fuse(
    faiss_results: List[Tuple[int, float]],
    bm25_results:  List[Tuple[int, float]],
) -> List[Dict]:
    """Reciprocal Rank Fusion — mirrors rag.py _hybrid_candidates()."""
    by_id: Dict[int, Dict] = {}

    for rank, (idx, dist) in enumerate(faiss_results, start=1):
        by_id.setdefault(idx, {"chunk_id": idx})
        by_id[idx]["faiss_rank"]  = rank
        by_id[idx]["faiss_dist"]  = dist

    for rank, (idx, score) in enumerate(bm25_results, start=1):
        by_id.setdefault(idx, {"chunk_id": idx})
        by_id[idx]["bm25_rank"]  = rank
        by_id[idx]["bm25_score"] = score

    for cand in by_id.values():
        rrf = 0.0
        if "faiss_rank" in cand:
            rrf += 1.0 / (RRF_K + cand["faiss_rank"])
        if "bm25_rank" in cand:
            rrf += 1.0 / (RRF_K + cand["bm25_rank"])
        cand["rrf_score"] = rrf

    return sorted(by_id.values(), key=lambda x: x["rrf_score"], reverse=True)


# ---------------------------------------------------------------------------
# Printers
# ---------------------------------------------------------------------------

def _chunk_summary(c: Dict, chunks: List[Dict], label: str = "") -> str:
    meta = chunks[c["chunk_id"]] if 0 <= c["chunk_id"] < len(chunks) else {}
    page  = meta.get("page", "?")
    ctype = meta.get("chunk_type", "?")
    text  = (meta.get("text") or "")
    snippet = text[:120].replace("\n", " ")
    return (
        f"{BOLD}[{label}] chunk_id={c['chunk_id']}  "
        f"page={page}  type={ctype}{RESET}\n"
        f"  faiss_rank={c.get('faiss_rank', '—'):>4}  "
        f"bm25_rank={c.get('bm25_rank', '—'):>4}  "
        f"rrf={c.get('rrf_score', 0.0):.5f}  "
        f"ce_score={_score_colour(c['ce_score']) if 'ce_score' in c else _dim('n/a')}\n"
        f"  {_dim(snippet)}"
    )


def print_top_n(
    results: List[Dict],
    chunks: List[Dict],
    label: str,
    n: int = 5,
) -> None:
    print(f"\n{BOLD}Top {n} — {label}{RESET}")
    _hr()
    for i, c in enumerate(results[:n], start=1):
        print(_chunk_summary(c, chunks, label=f"#{i}"))
        print()


# ---------------------------------------------------------------------------
# Brute-force chunk scanner
# ---------------------------------------------------------------------------

def scan_for_term(
    search_term: str,
    chunks: List[Dict],
    faiss_results: List[Tuple[int, float]],
    bm25_results:  List[Tuple[int, float]],
    fused_results: List[Dict],
    cross_encoder_results: Optional[List[Dict]],
) -> None:
    _section(f'Brute-Force Scan: "{search_term}"')

    faiss_rank_map = {idx: rank for rank, (idx, _) in enumerate(faiss_results, start=1)}
    bm25_rank_map  = {idx: rank for rank, (idx, _) in enumerate(bm25_results,  start=1)}
    rrf_rank_map   = {c["chunk_id"]: (i + 1, c["rrf_score"])
                      for i, c in enumerate(fused_results)}
    ce_score_map   = {c["chunk_id"]: c["ce_score"]
                      for c in (cross_encoder_results or []) if "ce_score" in c}

    term_lower = search_term.lower()
    hits = [
        (i, c) for i, c in enumerate(chunks)
        if term_lower in (c.get("text") or "").lower()
    ]

    if not hits:
        print(_err(f'  ✗ "{search_term}" NOT FOUND in any chunk.'))
        print(_err("    ⇒ The table may be missing from chunks.json entirely."))
        print(_err("    ⇒ Check: did ingest_tables.py parse the Markdown table correctly?"))
        return

    print(_ok(f'  ✓ "{search_term}" found in {len(hits)} chunk(s):\n'))

    for chunk_idx, chunk in hits:
        page   = chunk.get("page", "?")
        ctype  = chunk.get("chunk_type", "?")
        text   = (chunk.get("text") or "").replace("\n", " ")
        snippet_full = (chunk.get("text") or "")

        f_rank = faiss_rank_map.get(chunk_idx, None)
        b_rank = bm25_rank_map.get(chunk_idx, None)
        rrf_info = rrf_rank_map.get(chunk_idx)
        rrf_rank, rrf_score = rrf_info if rrf_info else (None, None)
        ce_score = ce_score_map.get(chunk_idx, None)

        print(f"  {BOLD}chunk_id={chunk_idx}  page={page}  type={ctype}{RESET}")

        # FAISS rank
        if f_rank is not None:
            status = _ok(f"rank #{f_rank}") if f_rank <= 10 else _warn(f"rank #{f_rank}")
            print(f"  FAISS    : {status}")
        else:
            print(f"  FAISS    : {_err('NOT in top-k results — semantic embedding mismatch')}")

        # BM25 rank
        if b_rank is not None:
            status = _ok(f"rank #{b_rank}") if b_rank <= 10 else _warn(f"rank #{b_rank}")
            print(f"  BM25     : {status}")
        else:
            print(f"  BM25     : {_err('NOT in top-k results — lexical mismatch')}")

        # RRF rank
        if rrf_rank is not None:
            status = _ok(f"rank #{rrf_rank}  rrf={rrf_score:.5f}")
            print(f"  RRF fused: {status}")
        else:
            print(f"  RRF fused: {_err('NOT in fused pool — both FAISS and BM25 missed it')}")

        # Cross-Encoder score
        if ce_score is not None:
            passed = ce_score >= RERANK_SCORE_THRESHOLD
            flag = _ok("PASSES threshold") if passed else _err(f"BELOW threshold ({RERANK_SCORE_THRESHOLD}) — DROPPED")
            print(f"  CE score : {_score_colour(ce_score)}  {flag}")
        elif rrf_rank is not None:
            print(f"  CE score : {_warn('Was in RRF pool but not in rerank-pool slice (pool too small?)')}")
        else:
            print(f"  CE score : {_dim('N/A (not in any candidate pool)')}")

        # Diagnosis
        print(f"\n  {BOLD}Diagnosis:{RESET}")
        if f_rank is None and b_rank is None:
            print(_err("  ✗ EMBEDDING + LEXICAL MISS — chunk is in the index but"
                        " neither FAISS nor BM25 ranked it in the top-k."))
            print(_warn("    → Try increasing --k (e.g. --k 200) to see if it"
                        " appears at a lower rank."))
        elif rrf_rank is None:
            print(_err("  ✗ RRF fusion dropped the chunk — it scored low on both arms."))
        elif ce_score is not None and ce_score < RERANK_SCORE_THRESHOLD:
            print(_err(f"  ✗ Cross-Encoder REJECTED the chunk "
                        f"(score={ce_score:.3f} < threshold={RERANK_SCORE_THRESHOLD})."))
            print(_warn("    → The CE model did not consider this chunk relevant to the query."))
            print(_warn("    → Try lowering RERANK_SCORE_THRESHOLD in rag.py, or rephrase the query."))
        elif ce_score is not None:
            print(_ok("  ✓ Chunk passed all filters — it SHOULD appear in the final answer."))
        else:
            print(_warn("  ? Chunk in RRF pool but not reranked (pool slicing). Increase --rerank-pool."))

        # Text preview
        print(f"\n  {DIM}Full chunk text preview (first 400 chars):{RESET}")
        print("  " + snippet_full[:400].replace("\n", "\n  "))
        _hr()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--query",       default=DEFAULT_QUERY,       help="Retrieval query")
    p.add_argument("--k",           default=DEFAULT_K,      type=int, help="FAISS/BM25 top-k")
    p.add_argument("--rerank-pool", default=DEFAULT_RERANK_POOL, type=int,
                   help="Candidates to pass to Cross-Encoder")
    p.add_argument("--search-term", default=DEFAULT_SEARCH_TERM,
                   help="String to brute-force scan the corpus for")
    p.add_argument("--no-rerank",   action="store_true",
                   help="Skip Cross-Encoder reranking (faster)")
    p.add_argument("--staging",     action="store_true",
                   help="Probe artifacts/vectorstore_staging/ instead of production")
    p.add_argument("--device",      default=None,
                   help="Embedding device: mps | cpu (default: auto)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    select_vectorstore("staging" if args.staging else "production")
    print(f"Vectorstore: {VECTORSTORE}  ({CHUNKS_PATH.parent})")

    # ── Device ───────────────────────────────────────────────────────────
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")

    _section("HarrisonGPT — Retrieval Diagnostic")
    print(f"  Query       : {BOLD}{args.query}{RESET}")
    print(f"  Search term : {BOLD}{args.search_term}{RESET}")
    print(f"  k           : {args.k}")
    print(f"  Rerank pool : {args.rerank_pool}")
    print(f"  Device      : {device.upper()}")
    print(f"  Rerank      : {'no' if args.no_rerank else 'yes'}")

    # ── Load data ─────────────────────────────────────────────────────────
    _section("Loading Data")
    t0 = time.perf_counter()
    chunks = load_chunks()
    index  = load_faiss_index()
    bm25   = build_bm25(chunks)
    model  = load_embedding_model(device)
    model_dim = _model_dimension(model)
    if index.d != model_dim:
        print(_err(
            f"✗ FAISS/index dimension mismatch: index.d={index.d}, "
            f"embedding_dim={model_dim}"
        ))
        sys.exit(1)
    ce     = None if args.no_rerank else load_cross_encoder()
    print(f"\n  Load time: {time.perf_counter() - t0:.1f}s")

    # ── FAISS search ──────────────────────────────────────────────────────
    _section("FAISS Semantic Search")
    t1 = time.perf_counter()
    faiss_results = faiss_search(args.query, index, model, args.k)
    print(f"  FAISS returned {len(faiss_results)} candidates in {time.perf_counter() - t1:.2f}s")
    print_top_n(
        [{"chunk_id": idx, "faiss_rank": rank + 1, "faiss_dist": dist}
         for rank, (idx, dist) in enumerate(faiss_results)],
        chunks, "FAISS",
    )

    # ── BM25 search ───────────────────────────────────────────────────────
    _section("BM25 Lexical Search")
    t2 = time.perf_counter()
    bm25_results = bm25_search(args.query, bm25, args.k)
    print(f"  BM25 returned {len(bm25_results)} candidates in {time.perf_counter() - t2:.2f}s")
    print_top_n(
        [{"chunk_id": idx, "bm25_rank": rank + 1, "bm25_score": score}
         for rank, (idx, score) in enumerate(bm25_results)],
        chunks, "BM25",
    )

    # ── RRF fusion ────────────────────────────────────────────────────────
    _section("RRF Fusion")
    fused = rrf_fuse(faiss_results, bm25_results)
    print(f"  RRF pool size: {len(fused)} unique chunks")
    print_top_n(fused, chunks, "RRF", n=5)

    # ── Cross-Encoder reranking ───────────────────────────────────────────
    ce_results: Optional[List[Dict]] = None
    if ce is not None:
        _section(f"Cross-Encoder Reranking  (pool={args.rerank_pool}, threshold={RERANK_SCORE_THRESHOLD})")
        rerank_pool = fused[: args.rerank_pool]
        pairs  = [[args.query, (chunks[c["chunk_id"]].get("text") or "")]
                  for c in rerank_pool if 0 <= c["chunk_id"] < len(chunks)]
        t3 = time.perf_counter()
        scores = ce.predict(pairs, show_progress_bar=False)
        print(f"  Reranked {len(pairs)} candidates in {time.perf_counter() - t3:.2f}s")

        for c, s in zip(rerank_pool, scores):
            c["ce_score"] = float(s)
        rerank_pool.sort(key=lambda x: x.get("ce_score", float("-inf")), reverse=True)
        ce_results = rerank_pool

        print(f"\n  {BOLD}Top 5 after Cross-Encoder (threshold={RERANK_SCORE_THRESHOLD}):{RESET}")
        _hr()
        passed = [c for c in ce_results if c.get("ce_score", float("-inf")) >= RERANK_SCORE_THRESHOLD]
        dropped = len(ce_results) - len(passed)
        for i, c in enumerate(ce_results[:5], start=1):
            chunk_meta = chunks[c["chunk_id"]] if 0 <= c["chunk_id"] < len(chunks) else {}
            print(
                f"  #{i}  chunk_id={c['chunk_id']}  "
                f"page={chunk_meta.get('page','?')}  "
                f"type={chunk_meta.get('chunk_type','?')}  "
                f"ce={_score_colour(c['ce_score'])}"
            )
        print(f"\n  {_ok(str(len(passed)))} chunks PASS threshold  |  "
              f"{_err(str(dropped))} chunks DROPPED")

    # ── Brute-force scan ──────────────────────────────────────────────────
    scan_for_term(
        args.search_term,
        chunks,
        faiss_results,
        bm25_results,
        fused,
        ce_results,
    )

    _section("Done")
    print(f"  Total runtime: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
