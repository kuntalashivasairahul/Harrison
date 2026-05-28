from typing import Dict, List
from backend.utils.fusion import clean_text


def extract_evidence(chunks: List[Dict]) -> List[str]:
    """
    Convert retrieved chunks into concise, page-cited evidence statements.

    Expected chunk keys:
    - "text": raw chunk text
    - "page": page number in Harrison
    """

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

        # Split into sentence-like segments
        parts = [p.strip() for p in cleaned.split(".") if p.strip()]
        if not parts:
            continue

        # Use up to TWO sentences instead of one (improves reasoning quality)
        statement = ". ".join(parts[:2])

        if not statement.endswith("."):
            statement += "."

        evidence.append(f"EVIDENCE: {statement} [p:{page}]")

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