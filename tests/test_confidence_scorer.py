"""
tests/test_confidence_scorer.py
================================
Unit tests for calculate_confidence() after threshold recalibration.

Verifies the new Harrison-calibrated thresholds:
    _HIGH_AVG_SCORE = -0.5   (was 3.5 — web-domain, never reached on medical text)
    _MED_AVG_SCORE  = -2.0   (was 1.0 — never reached on Harrison text)
    VERIFICATION_PENALTY_THRESHOLD = 0.40  (was 0.10 — too strict)

Each test also documents what the OLD thresholds would have returned to act as
regression proof that the miscalibration has been fixed.
"""
from __future__ import annotations

import unittest

from backend.agents.confidence_scorer import (
    _HIGH_AVG_SCORE,
    _MED_AVG_SCORE,
    VERIFICATION_PENALTY_THRESHOLD,
    calculate_confidence,
)


def _chunks(scores: list[float]) -> list[dict]:
    """Build minimal chunk dicts from a list of cross-encoder scores."""
    return [{"chunk_id": i, "score": s} for i, s in enumerate(scores)]


class TestThresholdConstants(unittest.TestCase):
    """Guard rails: constants must be at the calibrated values."""

    def test_high_threshold_is_calibrated_for_harrison(self):
        self.assertEqual(_HIGH_AVG_SCORE, -0.5,
                         "High threshold must be -0.5 (Harrison medical text calibration).")

    def test_med_threshold_is_calibrated_for_harrison(self):
        self.assertEqual(_MED_AVG_SCORE, -2.0,
                         "Med threshold must be -2.0 (Harrison medical text calibration).")

    def test_verification_penalty_threshold_is_relaxed(self):
        self.assertEqual(VERIFICATION_PENALTY_THRESHOLD, 0.40,
                         "Verification penalty threshold must be 0.40.")


class TestConfidenceTiers(unittest.TestCase):
    """Tier decision matrix with Harrison-realistic score vectors."""

    # ------------------------------------------------------------------
    # HIGH tier
    # ------------------------------------------------------------------
    def test_high_confidence_strong_alignment(self):
        """avg = -0.3, 2 chunks → High.
        Old thresholds (3.5/1.0): would have returned 'Low' (avg < 1.0).
        """
        chunks = _chunks([-0.1, -0.5])          # avg = -0.3, max_chunk = +0.05 realistic
        result = calculate_confidence(chunks, "answer text", "answer text")
        self.assertEqual(result, "High",
                         "avg=-0.3 with 2 chunks must return High.")

    def test_high_confidence_five_chunks_above_threshold(self):
        """avg = -0.22, 5 chunks → High."""
        chunks = _chunks([0.05, -0.22, -0.10, -0.40, -0.45])   # avg = -0.224
        result = calculate_confidence(chunks, "x", "x")
        self.assertEqual(result, "High")

    def test_high_requires_at_least_two_chunks(self):
        """avg well above -0.5 but only 1 chunk → Medium (not High)."""
        chunks = _chunks([-0.1])                 # avg = -0.1, but only 1 chunk
        result = calculate_confidence(chunks, "x", "x")
        self.assertEqual(result, "Medium",
                         "Single chunk above high threshold must return Medium.")

    # ------------------------------------------------------------------
    # MEDIUM tier
    # ------------------------------------------------------------------
    def test_medium_confidence_moderate_alignment(self):
        """avg = -1.0 (realistic for well-scoped Harrison query) → Medium.
        Old thresholds: would have returned 'Low' (avg < 1.0).
        """
        chunks = _chunks([-1.03, -1.76, -0.90, -0.80, -0.51])  # avg ≈ -1.0
        result = calculate_confidence(chunks, "answer", "answer")
        self.assertEqual(result, "Medium",
                         "avg=-1.0 must return Medium after recalibration.")

    def test_medium_confidence_plaque_rupture_scenario(self):
        """Scores from actual retrieval log for 'plaque rupture in CAD' → Medium.
        Old thresholds: would have returned 'Low'.
        """
        chunks = _chunks([0.05, -0.22, -0.55, -0.67, -0.85, -2.90])  # avg = -0.857
        result = calculate_confidence(chunks, "draft", "draft")
        self.assertEqual(result, "Medium")

    # ------------------------------------------------------------------
    # LOW tier
    # ------------------------------------------------------------------
    def test_low_confidence_no_chunks(self):
        """No retrieved chunks → Low (guard: no retrieval = no confidence)."""
        result = calculate_confidence([], "answer", "answer")
        self.assertEqual(result, "Low")

    def test_low_confidence_when_any_chunk_is_unscored(self):
        """Unscored neighbor/context chunks must not be interpreted as 0.0 relevance."""
        result = calculate_confidence(
            [{"chunk_id": 1, "score": -0.1}, {"chunk_id": 2}],
            "answer",
            "answer",
        )
        self.assertEqual(result, "Low")

    def test_low_confidence_weak_retrieval(self):
        """avg = -2.5, below _MED_AVG_SCORE threshold → Low."""
        chunks = _chunks([-2.1, -2.5, -2.9])    # avg = -2.5
        result = calculate_confidence(chunks, "x", "x")
        self.assertEqual(result, "Low")

    def test_low_confidence_very_weak_retrieval(self):
        """avg = -2.9 → Low."""
        chunks = _chunks([-2.8, -2.9, -3.0])
        result = calculate_confidence(chunks, "x", "x")
        self.assertEqual(result, "Low")

    # ------------------------------------------------------------------
    # Verification penalty
    # ------------------------------------------------------------------
    def test_verification_penalty_triggers_at_40_percent(self):
        """50% length ratio change → verification penalty → Low,
        even if retrieval scores would give Medium.
        """
        chunks = _chunks([-0.5, -1.0])           # avg = -0.75, would be Medium
        original = "a" * 100
        verified = "a" * 50                      # 50% shorter → ratio = 0.50 > 0.40
        result = calculate_confidence(chunks, original, verified)
        self.assertEqual(result, "Low",
                         "50% verifier edit must trigger penalty → Low.")

    def test_verification_penalty_does_not_trigger_at_30_percent(self):
        """30% length change does NOT trigger penalty (< 0.40 threshold)."""
        chunks = _chunks([-0.3, -0.4])           # avg = -0.35 → High tier if no penalty
        original = "a" * 100
        verified = "a" * 70                      # 30% shorter → ratio = 0.30 < 0.40
        result = calculate_confidence(chunks, original, verified)
        self.assertNotEqual(result, "Low",
                            "30% verifier edit must NOT trigger penalty.")

    def test_old_penalty_threshold_would_have_triggered(self):
        """15% change: old threshold (0.10) would cap at Low; new (0.40) allows Medium."""
        chunks = _chunks([-0.8, -1.2])           # avg = -1.0 → Medium
        original = "a" * 100
        verified = "a" * 85                      # 15% shorter → ratio = 0.15
        # New threshold: 0.15 < 0.40 → no penalty → Medium
        result = calculate_confidence(chunks, original, verified)
        self.assertEqual(result, "Medium",
                         "15% verifier edit must not be penalised with new threshold.")


class TestUnscoredChunkHandling(unittest.TestCase):
    """One unscored chunk used to discard every other chunk's score."""

    def test_single_unscored_chunk_among_many_does_not_force_low(self):
        chunks = [{"chunk_id": i, "score": -0.2} for i in range(11)] + [{"chunk_id": 99}]
        self.assertEqual(calculate_confidence(chunks, "a", "a"), "Medium")

    def test_unscored_chunks_still_block_high(self):
        chunks = [{"chunk_id": i, "score": 0.5} for i in range(11)] + [{"chunk_id": 99}]
        self.assertNotEqual(calculate_confidence(chunks, "a", "a"), "High")

    def test_all_scored_can_still_reach_high(self):
        chunks = [{"chunk_id": i, "score": 0.5} for i in range(12)]
        self.assertEqual(calculate_confidence(chunks, "a", "a"), "High")

    def test_half_or_more_unscored_is_low(self):
        self.assertEqual(
            calculate_confidence(
                [{"chunk_id": 1, "score": 0.5}, {"chunk_id": 2}, {"chunk_id": 3}], "a", "a"
            ),
            "Low",
        )

    def test_all_unscored_is_low(self):
        self.assertEqual(calculate_confidence([{"chunk_id": 1}, {"chunk_id": 2}], "a", "a"), "Low")

    def test_non_numeric_score_counts_as_unusable_not_as_a_crash(self):
        chunks = [{"chunk_id": i, "score": -0.2} for i in range(11)] + [{"chunk_id": 9, "score": "oops"}]
        self.assertEqual(calculate_confidence(chunks, "a", "a"), "Medium")


if __name__ == "__main__":
    unittest.main()
