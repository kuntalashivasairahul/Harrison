# backend/agents/confidence_scorer.py
"""
ConfidenceScorer — Deterministic RAG Response Confidence Assessment
====================================================================
Calculates a "High", "Medium", or "Low" confidence rating for the final
RAG response using purely mathematical heuristics. **No LLM calls.**

Algorithm
---------
The scorer combines two orthogonal signals:

1. **Retrieval Quality** (cross-encoder scores)
   Average the ``score`` field across all retrieved chunks.
   Cross-encoder logits from ms-marco-MiniLM-L-6-v2 span roughly
   −5.0 → +10.0; empirically:
     ≥ 3.5  → strong query–chunk alignment
     1.0–3.5 → moderate alignment
     < 1.0  → weak / noisy retrieval

2. **Verification Penalty** (draft vs verified answer divergence)
   Compares the LLM draft and the post-verification answer.
   A large edit distance indicates the verifier found significant
   unsupported claims — a strong hallucination signal.
   Penalty is measured as the character-level length ratio divergence:

       abs(len(verified) - len(original)) / max(len(original), 1)

   If this ratio exceeds VERIFICATION_PENALTY_THRESHOLD, confidence
   is capped at "Low" regardless of retrieval quality.

Design principles
-----------------
- Pure function: no I/O, no side-effects, trivially unit-testable.
- Conservative by default: ambiguous signals → downgrade, not upgrade.
- Named constants for every threshold so tuning requires no logic edits.
- Graceful degradation: missing or empty inputs return "Low".
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Thresholds — tune here, not in logic
# ---------------------------------------------------------------------------

# Average cross-encoder score thresholds (raw logit scale)
# Calibrated from observed ms-marco-MiniLM-L-6-v2 scores on Harrison text:
# Medical textbook paragraphs score in the range −3.0 → +0.1 (not the
# web-domain −5 → +10 range documented for the model).
# Derived from retrieval logs:
#   well-scoped medical queries  → avg ≈ −0.5 to −1.4, best chunk ≈ +0.05
#   moderate queries             → avg ≈ −1.5 to −2.5
#   noisy / off-domain queries   → avg < −2.5
_HIGH_AVG_SCORE: float = -0.5    # ≥ strong Harrison alignment
_MED_AVG_SCORE:  float = -2.0    # [−2.0, −0.5) → moderate evidence quality

# Verification divergence: if length ratio change exceeds this, cap at "Low".
# 0.40 = 40% character-length change between draft and verified answer.
# Minor verifier edits (< 40%) are expected and healthy; only major rewrites
# (where the verifier discards most of the draft) signal likely hallucination.
VERIFICATION_PENALTY_THRESHOLD: float = 0.40

# Minimum chunks required to even consider "High"
_MIN_CHUNKS_FOR_HIGH: int = 2


def calculate_confidence(
    chunks: list[dict],
    original_answer: str,
    verified_answer: str,
) -> str:
    """
    Return a confidence label — ``"High"``, ``"Medium"``, or ``"Low"`` —
    based on retrieval quality and verification divergence.

    Parameters
    ----------
    chunks : list[dict]
        Retrieved chunks as returned by ``retrieve()`` and
        ``route_and_sort_context()``.  Each dict may contain a ``"score"``
        key holding the Cross-Encoder relevance logit.  Chunks without a
        usable ``"score"`` are treated as insufficient evidence.
    original_answer : str
        The draft answer produced by the first LLM generation pass.
        Pass the final verified answer here when the draft is unavailable
        (the scorer will detect zero divergence and not apply a penalty).
    verified_answer : str
        The answer after the ``verify_answer()`` post-processing step.
        A large deviation from ``original_answer`` signals that the
        verifier pruned unsupported claims — a hallucination indicator.

    Returns
    -------
    str
        One of ``"High"``, ``"Medium"``, or ``"Low"``.

    Notes
    -----
    Decision matrix (evaluated top to bottom; first match wins):

    ┌──────────────────────────────────────────────────────────┬────────┐
    │ Condition                                                │ Label  │
    ├──────────────────────────────────────────────────────────┼────────┤
    │ No chunks retrieved                                      │ Low    │
    │ Verification divergence > VERIFICATION_PENALTY_THRESHOLD │ Low    │
    │ avg_score < _MED_AVG_SCORE                               │ Low    │
    │ avg_score ≥ _HIGH_AVG_SCORE AND len(chunks) ≥ 2         │ High   │
    │ Everything else                                          │ Medium │
    └──────────────────────────────────────────────────────────┴────────┘
    """
    # ── Guard: no retrieval → no confidence ──────────────────────────────
    if not chunks:
        return "Low"

    # ── Signal 1: Average cross-encoder score across all chunks ──────────
    scores = []
    for chunk in chunks:
        try:
            score = chunk.get("score")
            if score is None:
                return "Low"
            scores.append(float(score))
        except (AttributeError, TypeError, ValueError):
            return "Low"
    avg_score: float = sum(scores) / len(scores)

    # ── Signal 2: Verification divergence ────────────────────────────────
    # Measures how much the verified answer differs from the draft.
    # A ratio > VERIFICATION_PENALTY_THRESHOLD means the verifier made
    # substantial edits — strong hallucination signal → cap at "Low".
    original_len  = max(len((original_answer  or "").strip()), 1)
    verified_len  = max(len((verified_answer  or "").strip()), 1)
    length_ratio_change = abs(verified_len - original_len) / original_len

    verification_penalty: bool = length_ratio_change > VERIFICATION_PENALTY_THRESHOLD

    # ── Decision matrix ───────────────────────────────────────────────────
    if verification_penalty:
        return "Low"

    if avg_score < _MED_AVG_SCORE:
        return "Low"

    if avg_score >= _HIGH_AVG_SCORE and len(chunks) >= _MIN_CHUNKS_FOR_HIGH:
        return "High"

    return "Medium"
