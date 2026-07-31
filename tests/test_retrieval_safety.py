"""Focused no-network tests for deterministic, evidence-based retrieval."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from backend.retrieval import rag


class _FakeBm25:
    def get_scores(self, _tokens):
        return [0.0, 1.5, 0.0]


class TestRetrievalSafety(unittest.TestCase):
    def test_expand_query_is_deterministic_and_preserves_raw_query_first(self):
        query = "diabetic ketoacidosis"
        expected = [
            query,
            f"Harrison textbook explanation of {query}",
            f"clinical features, diagnosis and management of {query} in Harrison",
            f"What is {query}?",
        ]
        self.assertEqual(rag.expand_query(query), expected)
        self.assertEqual(rag.expand_query(query), expected)

    def test_bm25_zero_score_documents_are_not_candidates(self):
        chunks = [
            {"page": 1, "text": "alpha context"},
            {"page": 2, "text": "beta context"},
            {"page": 3, "text": "gamma context"},
        ]
        with patch.object(rag, "index", None), \
             patch.object(rag, "chunks", chunks), \
             patch.object(rag, "bm25", _FakeBm25()):
            candidates = rag._hybrid_candidates("beta", k=3, bm25_k=3)

        self.assertEqual([item["chunk_id"] for item in candidates], [1])
        self.assertEqual(candidates[0]["bm25_score"], 1.5)


if __name__ == "__main__":
    unittest.main()
