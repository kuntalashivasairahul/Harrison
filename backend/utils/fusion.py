# fusion.py
import re

def clean_text(text: str) -> str:
    text = re.sub(r"■ ■.*", "", text)
    text = re.sub(r"\(Reproduced.*?\)", "", text)
    text = re.sub(r"FIGURE\s*\d+-\d+.*", "", text)
    text = re.sub(r"TABLE\s*\d+-\d+.*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Token budget — 1 token ≈ 4 characters (lightweight heuristic, no tiktoken).
# Groq's llama-3.1-8b-instant has a 6,000 TPM on-demand limit.  We target
# ~4,500 tokens of context so that the prompt header, evidence, and question
# leave a comfortable generation headroom.
# ---------------------------------------------------------------------------
SAFE_TOKEN_LIMIT:   int = 3_000   # tokens  (~12,000 chars; leaves 3k headroom for generation)
SAFE_CHAR_LIMIT:    int = SAFE_TOKEN_LIMIT * 4   # characters  (= 12,000)


def fuse_chunks(chunks) -> str:
    """
    Fuse retrieved chunks into a single context string that is guaranteed to
    fit inside SAFE_CHAR_LIMIT characters (~SAFE_TOKEN_LIMIT tokens).

    Chunks are processed in the order supplied (highest RRF/CE score first).
    Each chunk is formatted as:

        ``- <cleaned_text> [p:<page>|c:<chunk_id>]``

    Lower-ranked chunks that would push the running total over the budget are
    silently dropped.  This prevents Groq 413 errors without removing the
    Verification, RRF, or Evidence layers.
    """
    parts: list[str] = []
    running_chars: int = 0

    for ch in chunks:
        text = ch.get("text") if isinstance(ch, dict) else None
        if not text:
            continue

        txt = clean_text(text)
        if len(txt.split()) < 5:
            continue

        page     = ch.get("page")
        chunk_id = ch.get("chunk_id")
        if page is None or chunk_id is None:
            continue

        line = f"- {txt} [p:{page}|c:{chunk_id}]"

        # Budget gate: stop adding chunks once we would exceed SAFE_CHAR_LIMIT
        if running_chars + len(line) + 1 > SAFE_CHAR_LIMIT:
            break

        parts.append(line)
        running_chars += len(line) + 1   # +1 for the joining newline

    fused_context = "\n".join(parts)
    # Absolute character hard-cap to guarantee the prompt context never exceeds budget (~2000 tokens)
    fused_context = fused_context[:8000]
    return fused_context


# ✅ ALIAS TO MATCH main.py
def fuse_context(chunks):
    return fuse_chunks(chunks)
