from typing import Dict, List
from fusion import clean_text


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