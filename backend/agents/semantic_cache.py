# backend/agents/semantic_cache.py
"""
SemanticCache — Embedding-based Response Cache
===============================================
Caches the final ``QueryResponse`` payload keyed by the *semantic embedding*
of the search query rather than its exact text.  This means that clinically
equivalent rephrasing of the same question (e.g. "MI management" vs
"management of myocardial infarction") is served from cache rather than
triggering a full FAISS + Groq round-trip.

Design principles (CODING_RULES.md §1, §2, §3)
-----------------------------------------------
- **Pure cache layer**: no retrieval math, no LLM calls, no HTTP logic.
  The class knows nothing about FAISS, BM25, or FastAPI.
- **Conservative threshold**: cosine similarity ≥ 0.95 is required for a hit.
  Borderline-similar but clinically distinct queries always miss the cache.
- **Crash-safe**: every public method catches all exceptions and degrades
  gracefully.  A broken cache never interrupts the main pipeline.
- **Thread-safe writes**: a ``threading.Lock`` serialises disk flushes so
  concurrent requests do not corrupt the JSON file.
- **Disk persistence**: the cache file survives server restarts.  Location:
  ``artifacts/semantic_cache.json`` relative to the project root.

Cache entry schema (stored in JSON)
-------------------------------------
Each entry is a JSON object with three keys:

.. code-block:: json

    {
        "embedding": [0.12, -0.34, ...],   // float list from the live embedding model
        "metadata": {                       // exact-match cache signature
            "mode": "qa",
            "embedding_model": "BAAI/bge-m3",
            "embedding_dim": 1024
        },
        "response":  {                      // exact QueryResponse payload
            "answer": "...",
            "confidence": "High",
            "sources": ["p.142"],
            "visual_context": [...]
        },
        "audit": {                          // non-keyed request/debug metadata
            "raw_query": "...",
            "search_query": "...",
            "returned_path": "verified"
        },
        "hits": 3                           // times this entry was served
    }

Limitations
-----------
- The cache is a flat list; lookup is O(n).  At a few hundred entries this
  is negligible (<1 ms).  If the cache grows very large (>10 000 entries)
  consider switching to FAISS or a vector DB for the lookup.
- Cache callers should pass exact-match metadata for mode, verifier state,
  model dimensions, and retrieval/prompt versions.  Semantic similarity is
  evaluated only after that signature matches.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Similarity threshold — must be met or exceeded for a cache hit.
# 0.95 is deliberately strict: only near-identical clinical queries share
# a cached response.  Lower values risk returning mismatched answers.
SIMILARITY_THRESHOLD: float = 0.95

# Maximum number of entries kept in memory and on disk.
# Oldest entries (by insertion order) are evicted when the cap is reached.
MAX_CACHE_ENTRIES: int = 500

# Absolute path to the cache file on disk.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CACHE_DIR    = _PROJECT_ROOT / "artifacts"
_CACHE_FILE   = _CACHE_DIR / "semantic_cache.json"


# ---------------------------------------------------------------------------
# Cosine similarity helper
# ---------------------------------------------------------------------------

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Compute the cosine similarity between two equal-length float lists.

    Returns a value in [-1.0, 1.0].  Returns 0.0 on any numerical error
    (e.g. zero-norm vectors) so a bad vector never triggers a false cache hit.
    """
    va = np.array(a, dtype=np.float32)
    vb = np.array(b, dtype=np.float32)

    norm_a = float(np.linalg.norm(va))
    norm_b = float(np.linalg.norm(vb))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.dot(va, vb) / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# SemanticCache
# ---------------------------------------------------------------------------

class SemanticCache:
    """
    In-process semantic response cache backed by a JSON file on disk.

    Usage
    -----
    Instantiate once at module level (so the JSON is loaded once at startup):

    .. code-block:: python

        cache = SemanticCache()

    Then inside the request handler:

    .. code-block:: python

        embedding = embed_text(search_query).flatten().tolist()

        hit = cache.check_cache(embedding, metadata=cache_signature)
        if hit:
            return QueryResponse(**hit)

        # ... run full pipeline ...

        cache.save_to_cache(embedding, response_payload, metadata=cache_signature)

    Thread safety
    -------------
    ``check_cache`` is lock-free (read-only access to the list).
    ``save_to_cache`` acquires a reentrant lock before mutating state and
    flushing to disk.
    """

    def __init__(self) -> None:
        self._entries: list[dict] = []
        self._lock = threading.Lock()
        self._load_from_disk()

    # ------------------------------------------------------------------
    # Disk I/O
    # ------------------------------------------------------------------

    def _load_from_disk(self) -> None:
        """
        Populate ``_entries`` from the JSON cache file.

        Creates the ``artifacts/`` directory and an empty cache file if
        neither exists yet.  Silently resets to an empty cache on any
        parse or I/O error so a corrupted file never prevents startup.
        """
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)

            if not _CACHE_FILE.exists():
                _CACHE_FILE.write_text("[]", encoding="utf-8")
                log.info("SemanticCache: created new cache file at %s", _CACHE_FILE)
                return

            raw = _CACHE_FILE.read_text(encoding="utf-8").strip()
            if not raw:
                self._entries = []
                return

            loaded = json.loads(raw)
            if not isinstance(loaded, list):
                log.warning(
                    "SemanticCache: cache file has unexpected structure — resetting."
                )
                self._entries = []
                return

            # Validate each entry has the minimum required keys.
            valid = []
            for entry in loaded:
                if (
                    isinstance(entry, dict)
                    and isinstance(entry.get("embedding"), list)
                    and isinstance(entry.get("response"), dict)
                ):
                    valid.append(entry)

            self._entries = valid
            log.info(
                "SemanticCache: loaded %d entries from %s", len(self._entries), _CACHE_FILE
            )

        except Exception as exc:
            log.warning(
                "SemanticCache: failed to load from disk (%s) — starting empty.", exc
            )
            self._entries = []

    def _flush_to_disk(self) -> None:
        """
        Atomically write the current ``_entries`` list to disk.

        Called while ``_lock`` is held.  Uses write-then-rename for
        atomicity so a mid-write crash leaves the previous file intact.
        """
        try:
            tmp_path = _CACHE_FILE.with_suffix(".tmp")
            # No indent: each entry carries a 1024-float embedding, and
            # pretty-printing puts every float on its own line — that alone
            # took the file from ~4 MB to ~14.6 MB at MAX_CACHE_ENTRIES, all
            # of it rewritten on every save inside the request path.
            tmp_path.write_text(
                json.dumps(self._entries, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            tmp_path.replace(_CACHE_FILE)  # atomic on POSIX; near-atomic on Windows
        except Exception as exc:
            log.warning("SemanticCache: disk flush failed (%s) — in-memory cache intact.", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def _metadata_matches(entry: dict, metadata: dict | None) -> bool:
        """Return True when ``entry`` is eligible for a request signature."""
        if metadata is None:
            return True
        return entry.get("metadata") == metadata

    def check_cache(
        self,
        query_embedding: list[float],
        metadata: dict | None = None,
    ) -> dict | None:
        """
        Search for a cached response whose embedding is semantically similar
        to ``query_embedding``.

        Parameters
        ----------
        query_embedding:
            A flat float list (e.g. produced by
            ``embed_text(search_query).flatten().tolist()``).
        metadata:
            Exact-match cache signature.  Entries with missing or different
            metadata are skipped before semantic similarity is computed.

        Returns
        -------
        dict | None
            The stored ``response`` dict (keys: ``answer``, ``confidence``,
            ``sources``, ``visual_context``) if a hit is found with
            similarity ≥ ``SIMILARITY_THRESHOLD``, otherwise ``None``.
        """
        if not self._entries or not query_embedding:
            return None

        best_sim: float = -1.0
        best_response: dict | None = None
        best_idx: int = -1

        # Score every eligible entry in one matmul.  The previous loop called
        # _cosine_similarity() per entry, and each call rebuilt a NumPy array
        # from the *query* list — 1024 floats converted once per stored entry
        # instead of once per request.
        try:
            dim = len(query_embedding)
            eligible = [
                (idx, entry)
                for idx, entry in enumerate(self._entries)
                if self._metadata_matches(entry, metadata)
                and isinstance(entry.get("embedding"), list)
                and len(entry["embedding"]) == dim
            ]
            if not eligible:
                log.debug("SemanticCache: MISS (no entry matches the request signature)")
                return None

            query_vec = np.asarray(query_embedding, dtype=np.float32)
            matrix = np.asarray([entry["embedding"] for _, entry in eligible], dtype=np.float32)

            query_norm = float(np.linalg.norm(query_vec))
            row_norms = np.linalg.norm(matrix, axis=1)
            denom = row_norms * query_norm
            with np.errstate(divide="ignore", invalid="ignore"):
                sims = np.where(denom > 0, matrix @ query_vec / denom, 0.0)

            best_row = int(np.argmax(sims))
            best_sim = float(sims[best_row])
            best_idx = eligible[best_row][0]
            best_response = eligible[best_row][1].get("response")
        except Exception as exc:
            log.warning("SemanticCache: similarity scan failed (%s) — treating as MISS.", exc)
            return None

        if best_sim >= SIMILARITY_THRESHOLD and best_response is not None:
            log.info(
                "SemanticCache: HIT (similarity=%.4f ≥ %.2f, entry=%d)",
                best_sim,
                SIMILARITY_THRESHOLD,
                best_idx,
            )
            # Increment hit counter (non-blocking best-effort).
            try:
                with self._lock:
                    self._entries[best_idx]["hits"] = (
                        self._entries[best_idx].get("hits", 0) + 1
                    )
            except Exception:
                pass
            return best_response

        log.debug(
            "SemanticCache: MISS (best_similarity=%.4f < %.2f)",
            best_sim,
            SIMILARITY_THRESHOLD,
        )
        return None

    def save_to_cache(
        self,
        query_embedding: list[float],
        response_data: dict,
        metadata: dict | None = None,
        audit_data: dict | None = None,
    ) -> None:
        """
        Persist a new cache entry to memory and disk.

        Parameters
        ----------
        query_embedding:
            Flat float list for the search query that produced ``response_data``.
        response_data:
            Dict with keys matching ``QueryResponse`` fields:
            ``answer``, ``confidence``, ``sources``, ``visual_context``.
        metadata:
            Exact-match cache signature required for future hits.
        audit_data:
            Optional non-keyed metadata for debugging cache entries.

        The method is a no-op on any error so the caller is never interrupted.
        """
        if not query_embedding or not response_data:
            return

        try:
            new_entry: dict = {
                "embedding": query_embedding,
                "metadata": metadata or {},
                # visual_context is intentionally NOT stored: it contains
                # absolute URLs built from the requesting host, and a cached
                # entry served behind a different host would hand back links
                # to the old one.  Callers rebuild it from "sources".
                "response": {
                    "answer":     response_data.get("answer", ""),
                    "confidence": response_data.get("confidence", "Low"),
                    "sources":    response_data.get("sources", []),
                },
                "audit": audit_data or {},
                "hits": 0,
            }

            with self._lock:
                self._entries.append(new_entry)

                # Evict oldest entries if cap exceeded.
                if len(self._entries) > MAX_CACHE_ENTRIES:
                    excess = len(self._entries) - MAX_CACHE_ENTRIES
                    self._entries = self._entries[excess:]
                    log.debug(
                        "SemanticCache: evicted %d oldest entries (cap=%d).",
                        excess,
                        MAX_CACHE_ENTRIES,
                    )

                self._flush_to_disk()

            log.info(
                "SemanticCache: saved new entry (total=%d).", len(self._entries)
            )

        except Exception as exc:
            log.warning("SemanticCache: save failed (%s) — skipping.", exc)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of entries currently in the cache."""
        return len(self._entries)

    def clear(self) -> None:
        """
        Wipe all in-memory entries and reset the disk file to an empty list.

        Intended for tests and administrative use only.
        """
        with self._lock:
            self._entries = []
            self._flush_to_disk()
        log.info("SemanticCache: cleared all entries.")
