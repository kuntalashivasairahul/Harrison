"""
tests/test_context_router.py
============================
ContextRouter — deduplication and page ordering.

This sits directly upstream of context fusion, and its page sort is half of the
ordering contract that fusion depends on: fusion selects by relevance and emits
in the order this module produces. It was the last untested backend module.
"""
from __future__ import annotations

import unittest

from backend.agents.context_router import (
    OVERLAP_THRESHOLD,
    _is_duplicate,
    route_and_sort_context,
)


def _chunk(page, text, **extra):
    return {"page": page, "text": text, **extra}


class TestDeduplication(unittest.TestCase):
    def test_identical_chunks_collapse_to_one(self):
        text = "Serum lipase three times the upper limit of normal is diagnostic."
        result = route_and_sort_context([_chunk(1, text), _chunk(2, text)])
        self.assertEqual(len(result), 1)

    def test_a_contained_shorter_chunk_is_dropped(self):
        short = "Serum lipase is elevated in acute pancreatitis."
        longer = short + " " * 2 + "x"
        result = route_and_sort_context([_chunk(1, longer), _chunk(2, short)])
        self.assertEqual(len(result), 1)

    def test_distinct_clinical_content_is_kept(self):
        result = route_and_sort_context([
            _chunk(1, "Ranson criteria assess severity on admission and at 48 hours."),
            _chunk(2, "BISAP uses BUN, impaired mental status, SIRS, effusion and age."),
        ])
        self.assertEqual(len(result), 2)

    def test_a_short_chunk_inside_a_much_longer_one_is_kept(self):
        """Below the overlap threshold the shorter chunk still adds context."""
        short = "Glucose is elevated."
        longer = "A" * 500 + short
        result = route_and_sort_context([_chunk(1, longer), _chunk(2, short)])
        self.assertEqual(len(result), 2)

    def test_empty_text_is_treated_as_a_duplicate(self):
        self.assertTrue(_is_duplicate("", ["anything"]))

    def test_threshold_is_a_named_constant(self):
        self.assertGreater(OVERLAP_THRESHOLD, 0.0)
        self.assertLessEqual(OVERLAP_THRESHOLD, 1.0)


class TestOrdering(unittest.TestCase):
    def test_survivors_come_back_in_ascending_page_order(self):
        result = route_and_sort_context([
            _chunk(900, "late chapter content that is entirely distinct here"),
            _chunk(100, "early chapter content that is entirely distinct here"),
            _chunk(500, "middle chapter content that is entirely distinct here"),
        ])
        self.assertEqual([c["page"] for c in result], [100, 500, 900])

    def test_same_page_ties_preserve_rerank_order(self):
        result = route_and_sort_context([
            _chunk(10, "highest scoring passage on this page", score=9.0),
            _chunk(10, "lower scoring passage on this same page", score=1.0),
        ])
        self.assertEqual([c["score"] for c in result], [9.0, 1.0])

    def test_non_numeric_pages_sort_first_and_are_not_dropped(self):
        result = route_and_sort_context([
            _chunk(5, "a passage with a real page number attached to it"),
            _chunk(None, "a passage whose page metadata is missing entirely"),
        ])
        self.assertEqual(len(result), 2)
        self.assertIsNone(result[0]["page"])


class TestSafety(unittest.TestCase):
    def test_empty_input_returns_empty(self):
        self.assertEqual(route_and_sort_context([]), [])
        self.assertEqual(route_and_sort_context(None), [])

    def test_input_list_is_not_mutated(self):
        original = [_chunk(900, "zzz distinct content here"), _chunk(100, "aaa distinct content here")]
        snapshot = list(original)
        route_and_sort_context(original)
        self.assertEqual(original, snapshot)

    def test_extra_keys_survive_the_round_trip(self):
        [result] = route_and_sort_context([_chunk(1, "a passage of clinical text", score=2.5, chunk_id=77)])
        self.assertEqual(result["score"], 2.5)
        self.assertEqual(result["chunk_id"], 77)

    def test_all_duplicates_still_returns_context(self):
        """Never hand an empty context to the LLM."""
        text = "identical passage repeated across the retrieved set"
        result = route_and_sort_context([_chunk(1, text), _chunk(2, text), _chunk(3, text)])
        self.assertGreaterEqual(len(result), 1)

    def test_malformed_chunks_do_not_raise(self):
        result = route_and_sort_context([{"nope": 1}, _chunk(2, "valid clinical passage of text here")])
        self.assertIsInstance(result, list)


if __name__ == "__main__":
    unittest.main()
