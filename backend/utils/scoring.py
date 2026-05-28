# backend/utils/scoring.py
"""
Confidence Scoring — Phase 2
============================
Provides a single public function, ``calculate_confidence``, that maps retrieval
quality signals to a human-readable confidence label: "High", "Medium", or "Low".

Design principles
-----------------
- **Pure function**: no I/O, no side-effects, trivially unit-testable.
- **Conservative by default**: when signals are ambiguous, we downgrade rather
  than upgrade (medical context demands caution over false confidence).
- **Extensible**: the thresholds live in named constants so they can be tuned
  or loaded from config without touching the logic.

Inputs (all passed from the /ask pipeline in Phase 3)
------------------------------------------------------
top_reranker_score : float
    The highest Cross-Encoder relevance score returned by ``rerank()``.
    Scores are raw logits from ms-marco-MiniLM; empirically, scores above
    roughly 5.0 indicate strong query–chunk alignment.
evidence_count : int
    Number of unique evidence statements that ``extract_evidence()`` produced.
    More corroborating chunks → higher trustworthiness.
was_verified : bool
    True when the evidence-checking / self-consistency pass detected that the
    LLM answer was well-supported by retrieved context.  False when the LLM
    diverged from the retrieved evidence (e.g., hallucination signal).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Thresholds (tune here, not in logic)
# ---------------------------------------------------------------------------

# Cross-encoder score thresholds (raw logit scale)
_HIGH_SCORE_THRESHOLD: float = 5.0   # strong alignment with query
_LOW_SCORE_THRESHOLD: float = 1.0    # weak alignment — likely noisy retrieval

# Evidence count thresholds
_HIGH_EVIDENCE_MIN: int = 2          # ≥ 2 corroborating chunks → supportive
_LOW_EVIDENCE_MAX: int = 1           # ≤ 1 chunk → thin evidence base


def calculate_confidence(
    top_reranker_score: float,
    evidence_count: int,
    was_verified: bool,
) -> str:
    """
    Return a confidence label – "High", "Medium", or "Low" – based on
    retrieval-quality heuristics.

    Decision matrix (evaluated in order; first match wins)
    -------------------------------------------------------
    | Condition                                   | Label  |
    |---------------------------------------------|--------|
    | Not verified (LLM diverged from evidence)   | Low    |
    | Score < LOW_SCORE_THRESHOLD                 | Low    |
    | Score < LOW_SCORE_THRESHOLD (any evidence)  | Low    |
    | Score ≥ HIGH_SCORE_THRESHOLD AND            |        |
    |   evidence_count ≥ HIGH_EVIDENCE_MIN        | High   |
    | Everything else                             | Medium |

    Parameters
    ----------
    top_reranker_score:
        Best Cross-Encoder score across the top-k retrieved chunks.
    evidence_count:
        Number of distinct evidence statements extracted from retrieved chunks.
    was_verified:
        Whether the answer was confirmed to be grounded in retrieved evidence.

    Returns
    -------
    str
        One of "High", "Medium", or "Low".
    """
    # --- Guard: unverified answers are always Low ---
    if not was_verified:
        return "Low"

    # --- Guard: very weak retrieval signal → Low ---
    if top_reranker_score < _LOW_SCORE_THRESHOLD:
        return "Low"

    # --- Strong signal AND multiple corroborating chunks → High ---
    if (
        top_reranker_score >= _HIGH_SCORE_THRESHOLD
        and evidence_count >= _HIGH_EVIDENCE_MIN
    ):
        return "High"

    # --- Everything in between → Medium ---
    return "Medium"
