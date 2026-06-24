"""
tests/test_rag_threshold.py
===========================
Asserts the RERANK_SCORE_THRESHOLD constant is at the approved value (-3.0).

This is a canary test: any accidental revert back to the over-aggressive -2.0
value will be caught immediately.
"""
from __future__ import annotations

import unittest


class TestRerankThreshold(unittest.TestCase):
    def test_threshold_is_minus_three(self):
        # Import isolated to inside the test so we always get a fresh read.
        from backend.retrieval.rag import RERANK_SCORE_THRESHOLD
        self.assertEqual(
            RERANK_SCORE_THRESHOLD,
            -3.0,
            "RERANK_SCORE_THRESHOLD must be -3.0 (was -2.0, which over-filtered medical chunks). "
            "If you intentionally changed this value, update the test.",
        )

    def test_threshold_is_not_old_value(self):
        from backend.retrieval.rag import RERANK_SCORE_THRESHOLD
        self.assertNotEqual(
            RERANK_SCORE_THRESHOLD,
            -2.0,
            "Threshold reverted to -2.0 — this causes 10/12 clinical chunks to be dropped.",
        )


if __name__ == "__main__":
    unittest.main()
