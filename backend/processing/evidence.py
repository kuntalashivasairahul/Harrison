
from backend.utils.fusion import clean_text


def extract_evidence(chunks: list[dict], exclude_chunk_ids: set | None = None) -> list[str]:
    """
    Convert retrieved chunks into page-cited evidence statements for the LLM.

    ``exclude_chunk_ids`` skips chunks the fused context already carries. The
    two blocks used to be built from the same list, so a chunk that fit the
    context budget was sent to the model twice — once as ``- text [p:N|c:M]``
    and again as ``EVIDENCE: text [p:N]``. Measured on one real query: 11
    chunks retrieved, 5 in the context, all 11 in an uncapped evidence block of
    25,337 characters, 42% of it verbatim duplication.

    Nothing is dropped, which RULE 3.2 forbids. Every retrieved chunk still
    reaches the model exactly once: in the context if it fit the budget, in
    evidence if it did not.

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

    evidence: list[str] = []

    if not chunks:
        return evidence

    excluded = exclude_chunk_ids or frozenset()

    for ch in chunks:
        if not isinstance(ch, dict):
            continue

        if ch.get("chunk_id") in excluded:
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



def extract_sources(chunks: list[dict]) -> list[str]:
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
