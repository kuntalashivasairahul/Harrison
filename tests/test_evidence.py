"""
tests/test_evidence.py
=======================
Unit tests for extract_evidence() and extract_sources() in
backend/processing/evidence.py.

Covers:
  - Citation format standardization: sources use p:NNN (colon, not dot)
  - extract_sources() de-duplication and sorting
  - extract_evidence() page-tagged format [p:PAGE]
  - Edge cases: no chunks, missing page, empty text
"""
from __future__ import annotations

import unittest

from backend.processing.evidence import extract_evidence, extract_sources


class TestExtractSourcesFormat(unittest.TestCase):
    """extract_sources() must return p:NNN colon format (not p.NNN dot format)."""

    def test_sources_use_dot_format(self):
        """Core format check: sources values must be 'p.NNN' dot notation."""
        chunks = [{"page": 2130, "text": "some text"}]
        result = extract_sources(chunks)
        self.assertEqual(result, ["p.2130"],
                         "Sources must use dot format p.2130")

    def test_sources_do_not_use_colon_format(self):
        """Regression: the 'p:NNN' colon format must never appear in sources."""
        chunks = [{"page": 100, "text": "text"}, {"page": 200, "text": "text"}]
        result = extract_sources(chunks)
        for src in result:
            self.assertNotIn(":", src,
                             f"Source '{src}' contains a colon — must use dot format.")

    def test_multiple_pages_sorted(self):
        chunks = [
            {"page": 512, "text": "text"},
            {"page": 142, "text": "text"},
            {"page": 143, "text": "text"},
        ]
        result = extract_sources(chunks)
        self.assertEqual(result, ["p.142", "p.143", "p.512"])

    def test_deduplicated(self):
        chunks = [
            {"page": 100, "text": "a"},
            {"page": 100, "text": "b"},
            {"page": 200, "text": "c"},
        ]
        result = extract_sources(chunks)
        self.assertEqual(result, ["p.100", "p.200"])

    def test_empty_chunks_returns_empty_list(self):
        self.assertEqual(extract_sources([]), [])

    def test_none_page_skipped(self):
        chunks = [{"page": None, "text": "text"}, {"page": 99, "text": "text"}]
        result = extract_sources(chunks)
        self.assertEqual(result, ["p.99"])

    def test_missing_page_key_skipped(self):
        chunks = [{"text": "no page key"}, {"page": 50, "text": "has page"}]
        result = extract_sources(chunks)
        self.assertEqual(result, ["p.50"])

    def test_single_source(self):
        chunks = [{"page": 1, "text": "text"}]
        result = extract_sources(chunks)
        self.assertEqual(result, ["p.1"])


class TestExtractEvidence(unittest.TestCase):
    """extract_evidence() must emit EVIDENCE: <text> [p:PAGE] format."""

    def test_evidence_page_uses_colon_format(self):
        """Inline evidence markers must use [p:PAGE] colon format."""
        chunks = [{"page": 2157, "text": "Atrial fibrillation is the most common arrhythmia."}]
        result = extract_evidence(chunks)
        self.assertTrue(len(result) == 1)
        self.assertIn("[p:2157]", result[0])
        self.assertNotIn("[p.2157]", result[0])  # dot format must not appear

    def test_evidence_prefixed_with_EVIDENCE(self):
        chunks = [{"page": 100, "text": "Heart failure affects 6 million Americans."}]
        result = extract_evidence(chunks)
        self.assertTrue(result[0].startswith("EVIDENCE:"))

    def test_empty_chunks_returns_empty_list(self):
        self.assertEqual(extract_evidence([]), [])

    def test_chunk_without_page_skipped(self):
        chunks = [{"text": "text with no page"}]
        result = extract_evidence(chunks)
        self.assertEqual(result, [])

    def test_multiple_chunks_all_included(self):
        chunks = [
            {"page": 1, "text": "First chunk text."},
            {"page": 2, "text": "Second chunk text."},
        ]
        result = extract_evidence(chunks)
        self.assertEqual(len(result), 2)
        self.assertIn("[p:1]", result[0])
        self.assertIn("[p:2]", result[1])


if __name__ == "__main__":
    unittest.main()
