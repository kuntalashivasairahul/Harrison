"""
tests/test_fusion.py
====================
Regression tests for context fusion budget enforcement.
"""
from __future__ import annotations

import unittest
from unittest.mock import patch

from backend.utils import fusion


class TestFuseChunks(unittest.TestCase):
    def test_fuse_chunks_enforces_safe_char_limit(self):
        chunks = [
            {"text": "alpha beta gamma delta epsilon", "page": 1, "chunk_id": 1},
            {"text": "zeta eta theta iota kappa", "page": 2, "chunk_id": 2},
            {"text": "lambda mu nu xi omicron", "page": 3, "chunk_id": 3},
        ]

        with patch.object(fusion, "SAFE_CHAR_LIMIT", 96):
            fused = fusion.fuse_chunks(chunks)

        self.assertLessEqual(len(fused), 96)
        self.assertIn("[p:1|c:1]", fused)
        self.assertIn("[p:2|c:2]", fused)
        self.assertNotIn("[p:3|c:3]", fused)

    def test_fuse_chunks_skips_invalid_or_tiny_chunks(self):
        chunks = [
            {"text": "too short", "page": 1, "chunk_id": 1},
            {"text": "valid clinical context with enough words", "page": None, "chunk_id": 2},
            {"text": "valid clinical context with enough words", "page": 3, "chunk_id": 3},
        ]

        fused = fusion.fuse_chunks(chunks)

        self.assertNotIn("[p:1|c:1]", fused)
        self.assertNotIn("[p:None|c:2]", fused)
        self.assertIn("[p:3|c:3]", fused)


class TestCleanText(unittest.TestCase):
    """Tables carry the scoring systems and criteria; they must survive."""

    def test_table_title_is_preserved(self):
        raw = "**TABLE 54-2 Recommendations for Investigations in Patients with Suspected Interstitial Cystitis**"
        cleaned = fusion.clean_text(raw)
        self.assertIn("TABLE 54-2", cleaned)
        self.assertIn("Interstitial Cystitis", cleaned)

    def test_table_body_after_the_title_is_preserved(self):
        raw = "TABLE 285-1 Ranson Criteria\nAge >55 years\nWBC >16,000/uL\nGlucose >200 mg/dL"
        cleaned = fusion.clean_text(raw)
        for token in ("Ranson", "Age >55 years", "16,000", "200 mg/dL"):
            self.assertIn(token, cleaned)

    def test_standalone_figure_caption_is_stripped(self):
        self.assertEqual(fusion.clean_text("FIGURE 12-3 Chest radiograph showing consolidation."), "")

    def test_figure_caption_does_not_eat_surrounding_prose(self):
        raw = "The lesion is shown below.\nFIGURE 4-1 Biopsy specimen.\nDiagnosis is confirmed by culture."
        cleaned = fusion.clean_text(raw)
        self.assertIn("The lesion is shown below.", cleaned)
        self.assertIn("Diagnosis is confirmed by culture.", cleaned)
        self.assertNotIn("Biopsy specimen", cleaned)

    def test_attribution_boilerplate_still_stripped(self):
        cleaned = fusion.clean_text("Serum lipase is elevated (Reproduced with permission from X).")
        self.assertNotIn("Reproduced", cleaned)
        self.assertIn("Serum lipase is elevated", cleaned)


class TestFuseChunksRelevanceBudget(unittest.TestCase):
    """The budget must sacrifice the least relevant chunk, not the last one."""

    def _page_ordered_chunks(self):
        # Realistic pipeline input: route_and_sort_context() has already
        # page-sorted, so the best chunk (score 8.0) sits LAST.
        return [
            {"text": "weak filler passage with plenty of words " * 4,
             "page": 10, "chunk_id": 1, "score": -2.5},
            {"text": "middling passage with plenty of words here " * 4,
             "page": 20, "chunk_id": 2, "score": -1.0},
            {"text": "the single most relevant passage in the corpus " * 4,
             "page": 900, "chunk_id": 3, "score": 8.0},
        ]

    def test_budget_drops_lowest_scoring_chunk_not_highest_page(self):
        chunks = self._page_ordered_chunks()
        one_line = len(fusion.fuse_chunks([chunks[2]]))

        # Room for two of the three lines.
        with patch.object(fusion, "SAFE_CHAR_LIMIT", one_line * 2 + 1):
            fused = fusion.fuse_chunks(chunks)

        self.assertIn("[p:900|c:3]", fused)      # best chunk survives
        self.assertIn("[p:20|c:2]", fused)       # second best survives
        self.assertNotIn("[p:10|c:1]", fused)    # worst chunk is the casualty

    def test_surviving_chunks_are_emitted_in_page_order(self):
        chunks = self._page_ordered_chunks()
        fused = fusion.fuse_chunks(chunks)

        positions = [fused.index(marker) for marker in ("[p:10|c:1]", "[p:20|c:2]", "[p:900|c:3]")]
        self.assertEqual(positions, sorted(positions))

    def test_top_scoring_chunk_always_survives_a_tight_budget(self):
        chunks = self._page_ordered_chunks()
        one_line = len(fusion.fuse_chunks([chunks[2]]))

        with patch.object(fusion, "SAFE_CHAR_LIMIT", one_line):
            fused = fusion.fuse_chunks(chunks)

        self.assertIn("[p:900|c:3]", fused)
        self.assertNotIn("[p:10|c:1]", fused)
        self.assertNotIn("[p:20|c:2]", fused)

    def test_unscored_chunks_keep_supplied_order_behaviour(self):
        chunks = [
            {"text": "alpha beta gamma delta epsilon", "page": 1, "chunk_id": 1},
            {"text": "zeta eta theta iota kappa", "page": 2, "chunk_id": 2},
            {"text": "lambda mu nu xi omicron", "page": 3, "chunk_id": 3},
        ]

        with patch.object(fusion, "SAFE_CHAR_LIMIT", 96):
            fused = fusion.fuse_chunks(chunks)

        self.assertIn("[p:1|c:1]", fused)
        self.assertIn("[p:2|c:2]", fused)
        self.assertNotIn("[p:3|c:3]", fused)

    def test_shorter_lower_scored_chunk_fills_leftover_space(self):
        chunks = [
            {"text": "a very long and verbose passage that eats the budget " * 6,
             "page": 1, "chunk_id": 1, "score": 5.0},
            {"text": "another very long and verbose passage entirely " * 6,
             "page": 2, "chunk_id": 2, "score": 4.0},
            {"text": "short but still useful clinical note", "page": 3, "chunk_id": 3, "score": 0.1},
        ]
        long_line = len(fusion.fuse_chunks([chunks[0]]))
        short_line = len(fusion.fuse_chunks([chunks[2]]))

        with patch.object(fusion, "SAFE_CHAR_LIMIT", long_line + short_line + 1):
            fused = fusion.fuse_chunks(chunks)

        self.assertIn("[p:1|c:1]", fused)
        self.assertIn("[p:3|c:3]", fused)
        self.assertNotIn("[p:2|c:2]", fused)


class TestSelectedChunkIds(unittest.TestCase):
    def test_reports_exactly_what_fuse_chunks_emits(self):
        chunks = [
            {"text": "alpha beta gamma delta epsilon zeta", "page": 1, "chunk_id": 11, "score": 5.0},
            {"text": "eta theta iota kappa lambda mu", "page": 2, "chunk_id": 22, "score": 4.0},
            {"text": "nu xi omicron pi rho sigma", "page": 3, "chunk_id": 33, "score": 3.0},
        ]
        one = len(fusion.fuse_chunks([chunks[0]]))
        with patch.object(fusion, "SAFE_CHAR_LIMIT", one * 2 + 1):
            fused = fusion.fuse_chunks(chunks)
            ids = fusion.selected_chunk_ids(chunks)

        for cid in ids:
            self.assertIn(f"c:{cid}]", fused)
        self.assertEqual(len(ids), fused.count("\n") + 1)

    def test_selects_by_score_not_position(self):
        chunks = [
            {"text": "low scoring passage with enough words here", "page": 1, "chunk_id": 1, "score": -5.0},
            {"text": "high scoring passage with enough words here", "page": 2, "chunk_id": 2, "score": 9.0},
        ]
        one = len(fusion.fuse_chunks([chunks[1]]))
        with patch.object(fusion, "SAFE_CHAR_LIMIT", one):
            self.assertEqual(fusion.selected_chunk_ids(chunks), {2})

    def test_skips_chunks_fusion_would_reject(self):
        chunks = [
            {"text": "too short", "page": 1, "chunk_id": 1},
            {"text": "valid clinical context with enough words", "page": None, "chunk_id": 2},
            {"text": "valid clinical context with enough words", "page": 3, "chunk_id": 3},
        ]
        self.assertEqual(fusion.selected_chunk_ids(chunks), {3})


if __name__ == "__main__":
    unittest.main()
