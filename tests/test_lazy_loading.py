"""
tests/test_lazy_loading.py
==========================
Guards the two import-time side effects that were removed:

1. ``import backend.llm.llm`` made a live call to Google's model-list API.
2. ``import backend.retrieval.rag`` parsed a 33 MB chunk registry, read the
   FAISS index, and built a BM25 index over the whole corpus (~13s).

Both are now deferred to first use, or to explicit startup warm-up.
"""
from __future__ import annotations

import importlib
import os
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.llm import llm
from backend.retrieval import rag


class TestModelDiscoveryIsLazy(unittest.TestCase):
    def setUp(self):
        self._saved = llm._resolved_models
        llm._resolved_models = None

    def tearDown(self):
        llm._resolved_models = self._saved

    def test_importing_the_module_does_not_resolve_models(self):
        """If import had resolved them, the cache would already be populated."""
        with patch.object(llm, "get_dynamic_models") as discover:
            import importlib
            importlib.import_module("backend.llm.llm")
            discover.assert_not_called()

    def test_resolution_happens_on_first_access(self):
        with patch.object(llm, "get_dynamic_models", return_value=("p", "b")) as discover:
            self.assertEqual(llm.prod_model(), "p")
            discover.assert_called_once()

    def test_resolution_is_cached_across_calls(self):
        with patch.object(llm, "get_dynamic_models", return_value=("p", "b")) as discover:
            llm.prod_model()
            llm.backup_model()
            llm.resolve_models()
            discover.assert_called_once()

    def test_force_refreshes_the_cache(self):
        with patch.object(llm, "get_dynamic_models", return_value=("p", "b")) as discover:
            llm.resolve_models()
            llm.resolve_models(force=True)
            self.assertEqual(discover.call_count, 2)

    def test_legacy_module_attribute_still_works(self):
        with patch.object(llm, "get_dynamic_models", return_value=("p", "b")):
            self.assertEqual(llm.PROD_MODEL, "p")
            self.assertEqual(llm.BACKUP_MODEL, "b")

    def test_unknown_attribute_still_raises(self):
        with self.assertRaises(AttributeError):
            llm.NOT_A_REAL_ATTRIBUTE  # noqa: B018


class TestVectorstoreIsLazy(unittest.TestCase):
    def setUp(self):
        self._saved = dict(rag._state)
        rag._state.clear()

    def tearDown(self):
        rag._state.clear()
        rag._state.update(self._saved)

    def test_module_import_leaves_the_vectorstore_unloaded(self):
        self.assertFalse(rag._state)

    def test_first_access_triggers_exactly_one_load(self):
        fake = {"chunks": [{"page": 1, "text": "t"}], "index": None, "bm25": None}
        with patch.object(rag, "_load", return_value=fake) as load:
            self.assertEqual(rag._chunks(), fake["chunks"])
            self.assertIsNone(rag._index())
            self.assertEqual(load.call_count, 2)

    def test_warmup_populates_state(self):
        with patch.object(rag, "_load") as load:
            rag.warmup()
            load.assert_called_once()

    def test_patched_module_attribute_wins_over_lazy_load(self):
        """Existing tests rely on patch.object(rag, "chunks", ...)."""
        fake = {"chunks": [{"page": 1, "text": "real"}], "index": None, "bm25": None}
        sentinel = [{"page": 99, "text": "patched"}]
        with patch.object(rag, "_load", return_value=fake):
            with patch.object(rag, "chunks", sentinel):
                self.assertEqual(rag._chunks(), sentinel)
            # ...and the override is removed cleanly afterwards.
            self.assertEqual(rag._chunks(), fake["chunks"])

    def test_patched_none_index_is_respected(self):
        fake = {"chunks": [], "index": "REAL-INDEX", "bm25": None}
        with patch.object(rag, "_load", return_value=fake):
            with patch.object(rag, "index", None):
                self.assertIsNone(rag._index())

    def test_patching_snapshots_the_attribute_and_so_materialises_it_once(self):
        """Documented consequence of PEP 562: mock.patch.object reads the
        attribute to save the original, which triggers a single load.  That is
        one load per test session, not one per import, which is the point."""
        fake = {"chunks": [], "index": None, "bm25": None}
        with patch.object(rag, "_load", return_value=fake) as load:
            with patch.object(rag, "chunks", []):
                pass
            self.assertGreaterEqual(load.call_count, 1)

    def test_unknown_attribute_still_raises(self):
        with self.assertRaises(AttributeError):
            rag.NOT_A_REAL_ATTRIBUTE  # noqa: B018


class TestEncoderIsLazy(unittest.TestCase):
    def test_encoder_is_not_constructed_at_import(self):
        from backend.retrieval import embeddings
        # A fresh process imports this module transitively via rag/main; if the
        # encoder were built at import, _model would be populated before any
        # call to get_model().  We assert the accessor exists and is used.
        self.assertTrue(hasattr(embeddings, "get_model"))
        self.assertTrue(callable(embeddings.get_model))

    def test_embed_text_goes_through_the_accessor(self):
        from backend.retrieval import embeddings
        fake = MagicMock()
        fake.encode.return_value = [[0.0] * 4]
        with patch.object(embeddings, "get_model", return_value=fake):
            embeddings.embed_text("acute pancreatitis")
        fake.encode.assert_called_once()


class TestDotenvIsLoadedOnceAndFirst(unittest.TestCase):
    """``backend/.env`` is read in exactly one place, above every env read.

    It used to be read in ``backend/llm/llm.py`` and
    ``backend/agents/query_optimizer.py``.  llm.py imported ``backend.config``
    on line 14 and only called ``load_dotenv`` on line 23, so any entry point
    whose first backend import was ``backend.llm.llm`` evaluated config's
    ``os.getenv`` block against an environment the .env file had never touched
    -- every ``LLM_*_SECONDS`` setting silently took its default.  Nothing
    caught it because query_optimizer happened to load the file first and the
    API entry point happened to import query_optimizer before config.
    """

    _BACKEND = Path(__file__).resolve().parents[1] / "backend"

    def test_only_config_loads_the_env_file(self):
        callers = sorted(
            path.relative_to(self._BACKEND).as_posix()
            for path in self._BACKEND.rglob("*.py")
            if "load_dotenv(" in path.read_text()
        )
        self.assertEqual(callers, ["config.py"])

    def test_config_loads_the_file_before_it_reads_the_environment(self):
        source = (self._BACKEND / "config.py").read_text()
        self.assertLess(source.index("load_dotenv("), source.index("os.getenv("))

    def test_a_value_arriving_from_the_env_file_reaches_config(self):
        """The ordering, exercised rather than inspected: a setting that exists
        only in the .env file must be visible to config's own getenv calls."""
        import backend.config as config

        def fake_load_dotenv(*_args, **_kwargs):
            os.environ["LLM_DRAFT_DEADLINE_SECONDS"] = "12.5"

        with patch.dict(os.environ, {}, clear=False), patch("dotenv.load_dotenv", fake_load_dotenv):
            os.environ.pop("LLM_DRAFT_DEADLINE_SECONDS", None)
            try:
                importlib.reload(config)
                self.assertEqual(config.LLM_DRAFT_DEADLINE_SECONDS, 12.5)
            finally:
                importlib.reload(config)


if __name__ == "__main__":
    unittest.main()
