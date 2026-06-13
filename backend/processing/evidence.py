from typing import Dict, List
from backend.utils.fusion import clean_text


def extract_evidence(chunks: List[Dict]) -> List[str]:
    """
    Convert retrieved chunks into page-cited evidence statements for the LLM.

    Each statement retains the full cleaned chunk text up to MAX_WORDS words
    so that critical pathophysiological mechanisms, lab thresholds, and scoring
    criteria are NOT discarded by an overly aggressive truncation heuristic.

    Format: ``EVIDENCE: <text> [p:<page>]``

    Expected chunk keys
    -------------------
    text : str   — raw chunk text
    page : int   — Harrison page number
    """

    MAX_WORDS = 400   # ~2 400 chars per chunk; generous but bounded

    evidence: List[str] = []

    if not chunks:
        return evidence

    for ch in chunks:
        if not isinstance(ch, dict):
            continue

        raw_text = ch.get("text") or ""
        page = ch.get("page")

        if page is None:
            continue

        cleaned = clean_text(raw_text)
        if not cleaned:
            continue

        # Truncate to MAX_WORDS if the chunk is very long, but keep whole words
        words = cleaned.split()
        if len(words) > MAX_WORDS:
            cleaned = " ".join(words[:MAX_WORDS])
            # Avoid trailing partial sentence — trim to last full stop
            last_stop = max(cleaned.rfind("."), cleaned.rfind("?"), cleaned.rfind("!"))
            if last_stop > len(cleaned) // 2:   # only trim if the stop is in the second half
                cleaned = cleaned[: last_stop + 1]

        if not cleaned.endswith((".", "?", "!")):
            cleaned += "."

        evidence.append(f"EVIDENCE: {cleaned} [p:{page}]")

    return evidence



def extract_sources(chunks: List[Dict]) -> List[str]:
    """
    Return a de-duplicated, sorted list of human-readable page references from
    the retrieved chunks.

    Each chunk is expected to carry a ``"page"`` key (int or str).  Pages that
    are ``None`` or empty are silently skipped.

    Returns
    -------
    List[str]
        Sorted list of unique page labels, e.g. ``["p.142", "p.143", "p.512"]``.
        Returns an empty list when no valid pages are found.
    """
    if not chunks:
        return []

    seen: set = set()
    for ch in chunks:
        if not isinstance(ch, dict):
            continue
        page = ch.get("page")
        if page is None:
            continue
        seen.add(page)

    # Sort numerically when pages are ints/floats, lexicographically otherwise
    try:
        ordered = sorted(seen, key=lambda p: int(p))
    except (TypeError, ValueError):
        ordered = sorted(seen, key=str)

    return [f"p.{p}" for p in ordered]