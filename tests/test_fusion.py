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


if __name__ == "__main__":
    unittest.main()
