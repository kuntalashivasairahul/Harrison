# backend/agents/context_router.py
"""
ContextRouter — Post-Retrieval Chunk Processor
===============================================
Sits between Cross-Encoder reranking and LLM generation.  It takes the
final scored chunk list (sorted by relevance) and returns a clean, ordered
list that is easier for the LLM to reason over:

1. **Deduplication** — removes chunks whose text is identical or is a >90%
   substring of an already-accepted chunk, preventing the LLM from seeing
   the same clinical paragraph twice and wasting context-window tokens.

2. **Chronological sort** — re-orders surviving chunks by ascending page
   number so the LLM receives textbook pages in the order they appear in
   Harrison's, producing more coherent, structurally sound summaries.

Design principles (CODING_RULES.md §1, §2, §3)
-----------------------------------------------
- **Pure function**: ``route_and_sort_context`` has no I/O, no side-effects,
  and is trivially unit-testable.
- **Zero LLM calls**: deduplication is text-overlap based — sub-millisecond.
- **Crash-safe**: any malformed chunk (missing keys, wrong types) is silently
  kept rather than dropped to avoid accidentally losing evidence.
- **No imports from** ``retrieval/``, ``llm/``, ``api/``, or ``rendering/``.
  This module depends only on the standard library.
"""

from __future__ import annotations

import logging
from typing import List, Dict

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Deduplication threshold
# ---------------------------------------------------------------------------
# A candidate chunk is dropped if the ratio of its text length to the length
# of any already-accepted chunk's text exceeds this value AND the candidate's
# text is a substring of the accepted chunk (or vice-versa).
#
# 0.90 means: "if ≥90% of this chunk's characters are contained verbatim in
# an already-accepted chunk, treat it as a duplicate."
#
# Raising this value (towards 1.0) is more permissive (fewer drops).
# Lowering it (towards 0.0) is more aggressive (more drops).
OVERLAP_THRESHOLD: float = 0.90


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_text(chunk: Dict) -> str:
    """Return the text of a chunk, normalised to a stripped string."""
    return (chunk.get("text") or "").strip()


def _get_page(chunk: Dict) -> int:
    """
    Return the page number as an integer for sorting.

    Falls back to ``0`` (sort-first) if the page key is absent or
    non-numeric so that malformed chunks are not silently dropped.
    """
    page = chunk.get("page")
    try:
        return int(page)
    except (TypeError, ValueError):
        return 0


def _is_duplicate(candidate_text: str, accepted_texts: List[str]) -> bool:
    """
    Return True if ``candidate_text`` is substantially contained within
    any already-accepted chunk text, or vice-versa.

    The check is bidirectional:
    - Candidate is a near-substring of an accepted chunk  (accepted is bigger)
    - Accepted chunk is a near-substring of the candidate (candidate is bigger)

    Both directions prevent false negatives when one chunk is a slight
    expansion of another (e.g., a neighbour-expanded chunk that includes
    a shorter chunk verbatim).

    Parameters
    ----------
    candidate_text:
        Stripped text of the chunk under consideration.
    accepted_texts:
        Stripped texts of all chunks that have already been accepted.

    Returns
    -------
    bool
        True → candidate should be dropped as a duplicate.
        False → candidate is novel enough to keep.
    """
    if not candidate_text:
        # Empty chunks are considered duplicates of "nothing" — drop them.
        return True

    cand_len = len(candidate_text)

    for accepted in accepted_texts:
        acc_len = len(accepted)
        if acc_len == 0:
            continue

        # Shorter text contained in longer text?
        if cand_len <= acc_len:
            # Candidate is shorter; check if it appears verbatim in accepted.
            if candidate_text in accepted:
                ratio = cand_len / acc_len
                if ratio >= OVERLAP_THRESHOLD:
                    return True
        else:
            # Accepted is shorter; check if it appears verbatim in candidate.
            if accepted in candidate_text:
                ratio = acc_len / cand_len
                if ratio >= OVERLAP_THRESHOLD:
                    return True

    return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def route_and_sort_context(chunks: List[Dict]) -> List[Dict]:
    """
    Deduplicate and chronologically sort a list of retrieved chunks.

    This function is the single public entry-point for the ContextRouter.
    It is designed to be called after Cross-Encoder reranking and before
    ``fuse_context()`` / ``extract_evidence()`` in the pipeline.

    Parameters
    ----------
    chunks : List[Dict]
        The scored chunk list returned by ``retrieve()``.  Each dict is
        expected to have at least the keys ``"text"`` (str) and ``"page"``
        (int or str).  Extra keys (``"score"``, ``"chunk_id"``, etc.) are
        preserved unchanged.

    Returns
    -------
    List[Dict]
        A new list with:
        - Duplicate / heavily-overlapping chunks removed.
        - Surviving chunks sorted by page number in ascending order.

        The original ``chunks`` list is **not mutated**.

    Guarantees
    ----------
    - If ``chunks`` is empty or ``None``, returns ``[]``.
    - If all chunks are duplicates, returns at least the first chunk
      (so the pipeline always has something to work with).
    - Never raises — any exception falls back to returning the original
      list so the pipeline is never interrupted.
    """
    if not chunks:
        return []

    try:
        # ── Step 1: Deduplication ──────────────────────────────────────
        accepted: List[Dict] = []
        accepted_texts: List[str] = []

        for chunk in chunks:
            text = _get_text(chunk)

            if _is_duplicate(text, accepted_texts):
                log.debug(
                    "ContextRouter: dropped duplicate chunk (page=%s, len=%d)",
                    chunk.get("page"),
                    len(text),
                )
                continue

            accepted.append(chunk)
            accepted_texts.append(text)

        # Safety net: if deduplication wiped everything, restore original.
        if not accepted:
            log.warning(
                "ContextRouter: all %d chunks were flagged as duplicates — "
                "keeping originals to avoid empty context.",
                len(chunks),
            )
            accepted = list(chunks)

        dropped = len(chunks) - len(accepted)
        if dropped:
            log.info(
                "ContextRouter: dropped %d duplicate chunk(s), %d remaining.",
                dropped,
                len(accepted),
            )

        # ── Step 2: Chronological sort ────────────────────────────────
        # Sort by page ascending so the LLM reads pages in textbook order.
        # Python's sort is stable: ties (same page) preserve relative order
        # from the reranker, keeping the highest-scoring chunk first.
        sorted_chunks = sorted(accepted, key=_get_page)

        log.debug(
            "ContextRouter: finalised %d chunk(s) spanning pages %s.",
            len(sorted_chunks),
            [_get_page(c) for c in sorted_chunks],
        )

        return sorted_chunks

    except Exception as exc:
        # Crash-safe fallback: log and return the original list unmodified.
        log.warning(
            "ContextRouter: unexpected error (%s: %s) — returning original chunks.",
            type(exc).__name__,
            exc,
        )
        return list(chunks)
