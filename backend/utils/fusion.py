# fusion.py
import os
import re


def clean_text(text: str) -> str:
    """Strip PDF page furniture without touching clinical content.

    Two rules used to over-reach and were removed or narrowed:

    - ``TABLE \\d+-\\d+.*`` deleted a table's title *and* everything after it
      on that line.  Table titles and bodies carry the scoring systems,
      diagnostic criteria, and dose tables the answer prompt explicitly
      demands, so a chunk like ``**TABLE 54-2 Recommendations for ...**``
      was reduced to ``**``.  Tables are no longer stripped at all.
    - ``FIGURE \\d+-\\d+.*`` is kept, because a figure caption without its
      image is genuinely low-value — but it is now anchored to the start of a
      line so it only removes a standalone caption, never the tail of a line
      that begins with prose.
    """
    text = re.sub(r"■ ■.*", "", text)
    text = re.sub(r"\(Reproduced.*?\)", "", text)
    text = re.sub(r"(?im)^[\s*_#>-]*FIGURE\s*\d+-\d+.*$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Context budget — 1 token ≈ 4 characters (lightweight heuristic, no tiktoken).
#
# The budget leaves room for the prompt header, evidence block, and question
# inside the draft model's input window.  (The old rationale here cited a Groq
# llama-3.1 TPM limit; Groq only runs the query optimizer, and that model has
# since been retired — the real consumer of this budget is Gemini.)
#
# SMART_SUMMARY_CONTEXT_CHAR_LIMIT is honoured here.  It was previously read
# into a module constant in llm.py and never used, so the documented knob had
# no effect on the fused-context budget it claimed to control.
# ---------------------------------------------------------------------------
SAFE_CHAR_LIMIT:    int = int(os.getenv("SMART_SUMMARY_CONTEXT_CHAR_LIMIT", "12000"))
SAFE_TOKEN_LIMIT:   int = SAFE_CHAR_LIMIT // 4   # retained for callers/tests


def _chunk_score(chunk: dict) -> float:
    """Cross-encoder relevance for budget selection.

    Chunks without a usable score sort last, so a caller that passes raw
    (un-reranked) chunks keeps the original supplied-order behaviour.
    """
    try:
        score = chunk.get("score")
        return float("-inf") if score is None else float(score)
    except (AttributeError, TypeError, ValueError):
        return float("-inf")


def fuse_chunks(chunks) -> str:
    """
    Fuse retrieved chunks into a single context string that is guaranteed to
    fit inside SAFE_CHAR_LIMIT characters (~SAFE_TOKEN_LIMIT tokens).

    Each chunk is formatted as:

        ``- <cleaned_text> [p:<page>|c:<chunk_id>]``

    Budget enforcement is two-pass, and the split matters:

    1. **Select** in descending cross-encoder score, so the chunks sacrificed
       to the character budget are the *least relevant* ones.
    2. **Emit** in the order supplied — page order, as set by
       ``route_and_sort_context()`` — so the model still reads Harrison
       sequentially.

    Doing the selection in the supplied order instead (as this function used
    to) means the budget drops whatever sits latest in that ordering.  With a
    page-sorted input that is the highest page number, which has nothing to do
    with relevance: the top-ranked chunk was silently discarded whenever it
    came from late in the textbook.
    """
    prepared: list[dict] = []

    for position, ch in enumerate(chunks):
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

        prepared.append(
            {
                "position": position,
                "line": f"- {txt} [p:{page}|c:{chunk_id}]",
                "score": _chunk_score(ch),
            }
        )

    # Pass 1 — select by relevance.  Ties (and unscored chunks) fall back to
    # supplied order, so this is a no-op for callers that pass no scores.
    by_relevance = sorted(prepared, key=lambda item: (-item["score"], item["position"]))

    selected: list[dict] = []
    running_chars: int = 0
    for item in by_relevance:
        projected_chars = running_chars + len(item["line"]) + (1 if selected else 0)
        if projected_chars > SAFE_CHAR_LIMIT:
            # Keep going rather than break: a shorter, lower-scored chunk may
            # still fit in the space this one could not.
            continue
        selected.append(item)
        running_chars = projected_chars

    # Pass 2 — emit in supplied (page) order.
    selected.sort(key=lambda item: item["position"])

    return "\n".join(item["line"] for item in selected)


# ✅ ALIAS TO MATCH main.py
def fuse_context(chunks):
    return fuse_chunks(chunks)
