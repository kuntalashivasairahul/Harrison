"""
tests/test_semantic_cache.py
============================
Regression tests for semantic cache signature isolation.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from backend.agents import semantic_cache as cache_mod


class TestSemanticCache(unittest.TestCase):
    def setUp(self):
        self._old_cache_dir = cache_mod._CACHE_DIR
        self._old_cache_file = cache_mod._CACHE_FILE
        self._tmp = tempfile.TemporaryDirectory()
        cache_mod._CACHE_DIR = Path(self._tmp.name)
        cache_mod._CACHE_FILE = cache_mod._CACHE_DIR / "semantic_cache.json"

    def tearDown(self):
        cache_mod._CACHE_DIR = self._old_cache_dir
        cache_mod._CACHE_FILE = self._old_cache_file
        self._tmp.cleanup()

    def _seed(self, entries):
        cache_mod._CACHE_FILE.write_text(json.dumps(entries), encoding="utf-8")

    def test_entries_from_a_superseded_schema_are_dropped_at_load(self):
        """A stale entry can never match a signature again, but it was still
        loaded at startup, scanned on every lookup, and rewritten by every
        flush. Six of twenty-one entries on disk were pre-v5 dead weight."""
        self._seed([
            {"embedding": [1.0], "response": {"answer": "old"},
             "metadata": {"schema": "semantic-cache-v4"}},
            {"embedding": [1.0], "response": {"answer": "new"},
             "metadata": {"schema": "semantic-cache-v5"}},
        ])
        cache = cache_mod.SemanticCache(schema_version="semantic-cache-v5")

        self.assertEqual(cache.size, 1)
        self.assertEqual(cache._entries[0]["response"]["answer"], "new")

    def test_retiring_rewrites_the_file_rather_than_only_memory(self):
        """Dropping them in memory alone leaves the next process to reload the
        same dead weight."""
        self._seed([
            {"embedding": [1.0], "response": {"answer": "old"},
             "metadata": {"schema": "semantic-cache-v4"}},
        ])
        cache_mod.SemanticCache(schema_version="semantic-cache-v5")

        on_disk = json.loads(cache_mod._CACHE_FILE.read_text(encoding="utf-8"))
        self.assertEqual(on_disk, [])

    def test_no_schema_version_retires_nothing(self):
        """The version is the caller's to own; without one the cache keeps its
        previous behaviour rather than guessing."""
        self._seed([
            {"embedding": [1.0], "response": {"answer": "old"},
             "metadata": {"schema": "semantic-cache-v4"}},
        ])
        self.assertEqual(cache_mod.SemanticCache().size, 1)

    def test_metadata_mismatch_misses_even_with_same_embedding(self):
        cache = cache_mod.SemanticCache()
        embedding = [1.0, 0.0, 0.0]
        qa_signature = {
            "schema": "semantic-cache-v2",
            "mode": "qa",
            "disable_verifier": False,
            "embedding_model": "BAAI/bge-m3",
            "embedding_dim": 1024,
        }
        summary_signature = {
            **qa_signature,
            "mode": "smart_summary",
        }

        cache.save_to_cache(
            query_embedding=embedding,
            metadata=qa_signature,
            response_data={
                "answer": "qa answer",
                "confidence": "High",
                "sources": [],
                "visual_context": [],
            },
        )

        self.assertIsNone(cache.check_cache(embedding, metadata=summary_signature))
        hit = cache.check_cache(embedding, metadata=qa_signature)
        self.assertIsNotNone(hit)
        self.assertEqual(hit["answer"], "qa answer")

    def test_embedding_dimension_mismatch_misses(self):
        cache = cache_mod.SemanticCache()
        signature = {
            "schema": "semantic-cache-v2",
            "mode": "qa",
            "disable_verifier": False,
            "embedding_model": "BAAI/bge-m3",
            "embedding_dim": 1024,
        }
        cache.save_to_cache(
            query_embedding=[1.0, 0.0, 0.0],
            metadata=signature,
            response_data={
                "answer": "cached answer",
                "confidence": "High",
                "sources": [],
                "visual_context": [],
            },
        )

        self.assertIsNone(cache.check_cache([1.0, 0.0], metadata=signature))

    def test_audit_data_is_stored_but_not_part_of_key(self):
        cache = cache_mod.SemanticCache()
        signature = {
            "schema": "semantic-cache-v2",
            "mode": "qa",
            "disable_verifier": False,
            "embedding_model": "BAAI/bge-m3",
            "embedding_dim": 1024,
        }
        cache.save_to_cache(
            query_embedding=[1.0, 0.0, 0.0],
            metadata=signature,
            audit_data={
                "raw_query": "What is DKA?",
                "search_query": "diabetic ketoacidosis",
                "returned_path": "verified",
            },
            response_data={
                "answer": "cached answer",
                "confidence": "High",
                "sources": [],
                "visual_context": [],
            },
        )

        hit = cache.check_cache([1.0, 0.0, 0.0], metadata=signature)
        self.assertIsNotNone(hit)
        self.assertEqual(hit["answer"], "cached answer")
        self.assertEqual(cache._entries[0]["audit"]["returned_path"], "verified")


if __name__ == "__main__":
    unittest.main()
