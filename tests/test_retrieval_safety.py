"""Focused no-network tests for deterministic, evidence-based retrieval."""
from __future__ import annotations

import unittest
from unittest.mock import patch

from backend.retrieval import rag


class _FakeBm25:
    def get_scores(self, _tokens):
        return [0.0, 1.5, 0.0]


class TestRetrievalSafety(unittest.TestCase):
    def test_rule_based_query_expansion_is_gone(self):
        """Expansion is the query optimizer's job; the template variants it
        used to bolt on were ungrammatical restatements of one query."""
        self.assertFalse(hasattr(rag, "expand_query"))

    def test_retrieve_issues_exactly_one_hybrid_pass(self):
        seen: list[str] = []

        def _spy(query, k, bm25_k):
            seen.append(query)
            return []

        with patch.object(rag, "chunks", [{"page": 1, "text": "x"}]), \
             patch.object(rag, "_hybrid_candidates", _spy):
            rag.retrieve("management of acute pancreatitis", k=8, final_k=2, rerank_pool=4)

        self.assertEqual(seen, ["management of acute pancreatitis"])

    def test_short_table_rows_are_not_filtered_as_low_value(self):
        """Table-aware chunking is pointless if the filter drops table rows."""
        self.assertFalse(rag.is_low_value_text("TABLE 285-1 Ranson Criteria: age >55 years, WBC >16,000/uL"))
        self.assertFalse(rag.is_low_value_text("Glucose >200 mg/dL is one Ranson criterion on admission."))

    def test_words_merely_containing_table_are_not_filtered(self):
        """Substring matching used to fire on treatable/predictable/intractable."""
        for text in (
            "Community-acquired pneumonia is a treatable cause of acute respiratory failure.",
            "The course of the disease is predictable once therapy begins in earnest.",
            "Intractable seizures may require surgical evaluation at a referral centre.",
        ):
            self.assertFalse(rag.is_low_value_text(text), text)

    def test_figure_captions_are_still_filtered(self):
        self.assertTrue(rag.is_low_value_text("FIGURE 12-3 Chest radiograph showing dense consolidation."))
        self.assertTrue(rag.is_low_value_text("References: Smith J, Jones A. N Engl J Med. 2019."))

    def test_tokenizer_strips_trailing_punctuation(self):
        """"pancreatitis," and "pancreatitis" used to be different BM25 terms."""
        self.assertEqual(rag._tokenize("acute pancreatitis."), ["acute", "pancreatitis"])
        self.assertEqual(rag._tokenize("(elevated)"), ["elevated"])
        self.assertEqual(rag._tokenize("criteria; diagnosis:"), ["criteria", "diagnosis"])

    def test_tokenizer_preserves_clinical_values(self):
        self.assertEqual(rag._tokenize("pH < 7.30"), ["ph", "7.30"])
        self.assertEqual(rag._tokenize("glucose > 250 mg/dL"), ["glucose", "250", "mg/dl"])
        self.assertEqual(rag._tokenize("BUN/creatinine ratio"), ["bun/creatinine", "ratio"])
        self.assertEqual(rag._tokenize("WBC >16,000/uL"), ["wbc", "16,000/ul"])

    def test_query_and_corpus_use_the_same_tokenizer(self):
        """A mismatch here silently zeroes the lexical half of retrieval."""
        import inspect
        source = inspect.getsource(rag._load)
        self.assertIn("_tokenize(", source)

    def test_neighbour_expansion_respects_page_adjacency(self):
        self.assertTrue(rag._is_page_adjacent(100, 100))
        self.assertTrue(rag._is_page_adjacent(100, 101))
        self.assertTrue(rag._is_page_adjacent(100, 99))
        self.assertFalse(rag._is_page_adjacent(100, 102))
        self.assertFalse(rag._is_page_adjacent(100, 2500))

    def test_unknown_pages_keep_the_neighbour_rather_than_lose_evidence(self):
        self.assertTrue(rag._is_page_adjacent(None, 100))
        self.assertTrue(rag._is_page_adjacent(100, None))
        self.assertTrue(rag._is_page_adjacent("x", "y"))

    def test_neighbour_across_a_chapter_boundary_is_not_pulled_in(self):
        chunks = [
            {"page": 100, "text": "cardiology content that is long enough to survive filtering"},
            {"page": 2500, "text": "unrelated nephrology content long enough to survive filtering"},
        ]
        candidate = [{"chunk_id": 0, "page": 100, "text": chunks[0]["text"], "rrf_score": 1.0}]
        with patch.object(rag, "chunks", chunks):
            pool = rag._pretrim_for_rerank(candidate, final_k=5, rerank_pool=5)
        self.assertEqual([c["chunk_id"] for c in pool], [0])

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
