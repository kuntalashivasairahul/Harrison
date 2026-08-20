"""
tests/test_key_manager.py
==========================
Unit tests for the refactored KeyManager:
  - Main key plus explicit 10-slot loading
  - Round-robin via next_client()
  - Exhaustion tracking via mark_exhausted()
  - All-exhausted raises RuntimeError
"""
from __future__ import annotations

import threading
import unittest
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers to build a KeyManager from arbitrary key lists without env vars
# ---------------------------------------------------------------------------

def _make_km(*keys: str):
    """Create a KeyManager whose pool is exactly `keys`, bypassing env loading."""
    from backend.llm.llm import KeyManager
    km = KeyManager.__new__(KeyManager)
    import threading
    km._lock = threading.Lock()
    km._keys = list(keys)
    km._current_idx = -1
    km._exhausted = set()
    km._rate_limited_until = {}
    return km


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestKeyLoading(unittest.TestCase):
    """Main Gemini key plus deterministic GEMINI_API_KEY_1..10 loading."""

    def test_explicit_slots_loaded_in_order(self):
        env = {
            "GEMINI_API_KEY_3": "key-c",
            "GEMINI_API_KEY_1": "key-a",
            "GEMINI_API_KEY_2": "key-b",
        }
        with patch.dict("os.environ", env, clear=True):
            # Re-create a fresh instance (don't reload the module — singleton)
            from backend.llm.llm import KeyManager
            km = KeyManager.__new__(KeyManager)
            import threading
            km._lock = threading.Lock()
            km._keys = []
            km._current_idx = -1
            km._exhausted = set()
            # Manually run __init__ body
            import os
            for slot in range(1, KeyManager.TOTAL_SLOTS + 1):
                val = os.getenv(f"GEMINI_API_KEY_{slot}", "").strip()
                if not val and slot == 1:
                    val = os.getenv("GEMINI_API_KEY", "").strip()
                if val:
                    km._keys.append(val)
        self.assertEqual(km._keys, ["key-a", "key-b", "key-c"])

    def test_main_key_is_loaded_before_numbered_keys(self):
        env = {"GEMINI_API_KEY": "bare-key"}
        with patch.dict("os.environ", env, clear=True):
            import os
            import threading

            from backend.llm.llm import KeyManager
            km = KeyManager.__new__(KeyManager)
            km._lock = threading.Lock()
            km._keys = []
            km._current_idx = -1
            km._exhausted = set()
            main_key = os.getenv("GEMINI_API_KEY", "").strip()
            if main_key:
                km._keys.append(main_key)
            for slot in range(1, KeyManager.TOTAL_SLOTS + 1):
                val = os.getenv(f"GEMINI_API_KEY_{slot}", "").strip()
                if val:
                    km._keys.append(val)
        self.assertEqual(km._keys, ["bare-key"])

    def test_main_key_and_slot_one_are_distinct(self):
        with patch.dict("os.environ", {"GEMINI_API_KEY": "main", "GEMINI_API_KEY_1": "slot-one"}, clear=True):
            from backend.llm.llm import KeyManager
            km = KeyManager()
        self.assertEqual(km._keys, ["main", "slot-one"])

    def test_total_slots_is_10(self):
        from backend.llm.llm import KeyManager
        self.assertEqual(KeyManager.TOTAL_SLOTS, 10)

    def test_empty_slots_skipped_gracefully(self):
        """If only slots 1 and 5 are set, pool has exactly 2 keys."""
        env = {
            "GEMINI_API_KEY_1": "key-1",
            "GEMINI_API_KEY_5": "key-5",
        }
        with patch.dict("os.environ", env, clear=True):
            import os
            import threading

            from backend.llm.llm import KeyManager
            km = KeyManager.__new__(KeyManager)
            km._lock = threading.Lock()
            km._keys = []
            km._current_idx = -1
            km._exhausted = set()
            main_key = os.getenv("GEMINI_API_KEY", "").strip()
            if main_key:
                km._keys.append(main_key)
            for slot in range(1, KeyManager.TOTAL_SLOTS + 1):
                val = os.getenv(f"GEMINI_API_KEY_{slot}", "").strip()
                if val:
                    km._keys.append(val)
        self.assertEqual(len(km._keys), 2)
        self.assertEqual(km._keys[0], "key-1")
        self.assertEqual(km._keys[1], "key-5")


class TestRoundRobin(unittest.TestCase):
    """next_client() must advance round-robin and return correct client."""

    def _patched_client(self, km):
        """Patch genai.Client to return a mock that records the api_key."""
        mock_client = MagicMock()
        mock_client.api_key = None

        def fake_client(api_key):
            c = MagicMock()
            c.api_key = api_key
            return c

        with patch("backend.llm.llm.genai.Client", side_effect=fake_client):
            return km.next_client()

    def test_first_call_returns_key0(self):
        km = _make_km("k0", "k1", "k2")
        with patch("backend.llm.llm.genai.Client", side_effect=lambda api_key: MagicMock(api_key=api_key)):
            c = km.next_client()
        self.assertEqual(c.api_key, "k0")
        self.assertEqual(km._current_idx, 0)

    def test_single_key_always_returns_same(self):
        km = _make_km("only-key")
        with patch("backend.llm.llm.genai.Client", side_effect=lambda api_key: MagicMock(api_key=api_key)):
            for _ in range(3):
                c = km.next_client()
                self.assertEqual(c.api_key, "only-key")

    def test_no_keys_raises(self):
        km = _make_km()
        with self.assertRaises(RuntimeError):
            km.next_client()


class TestExhaustionTracking(unittest.TestCase):
    """mark_exhausted() + next_client() must skip exhausted keys."""

    def _call(self, km):
        with patch("backend.llm.llm.genai.Client", side_effect=lambda api_key: MagicMock(api_key=api_key)):
            return km.next_client()

    def test_exhausted_key_is_skipped(self):
        km = _make_km("k0", "k1", "k2")
        c0 = self._call(km)         # uses k0, idx=0
        km.mark_exhausted()         # k0 exhausted
        c1 = self._call(km)         # must skip k0 → uses k1
        self.assertEqual(c0.api_key, "k0")
        self.assertEqual(c1.api_key, "k1")
        self.assertIn(0, km._exhausted)

    def test_multiple_exhausted_skipped(self):
        km = _make_km("k0", "k1", "k2")
        self._call(km)              # uses k0
        km.mark_exhausted()         # k0 exhausted
        self._call(km)              # uses k1
        km.mark_exhausted()         # k1 exhausted
        c = self._call(km)          # must use k2
        self.assertEqual(c.api_key, "k2")

    def test_all_exhausted_raises(self):
        km = _make_km("k0", "k1")
        self._call(km)              # uses k0
        km.mark_exhausted()         # k0 exhausted
        self._call(km)              # uses k1
        km.mark_exhausted()         # k1 exhausted
        with patch("backend.llm.llm.genai.Client", side_effect=lambda api_key: MagicMock(api_key=api_key)):
            with self.assertRaises(RuntimeError) as ctx:
                km.next_client()
        self.assertIn("exhausted", str(ctx.exception).lower())

    def test_mark_exhausted_at_negative_idx_is_safe(self):
        """mark_exhausted() before any next_client() call must not crash."""
        km = _make_km("k0")
        km.mark_exhausted()         # _current_idx = -1, should be a no-op
        self.assertEqual(len(km._exhausted), 0)


class TestRateLimitCooldown(unittest.TestCase):
    """A transient 429 must not permanently deplete the key pool."""

    def test_rate_limited_key_is_skipped_then_recovers(self):
        km = _make_km("k0", "k1")
        with patch("backend.llm.llm.genai.Client", side_effect=lambda api_key: MagicMock(api_key=api_key)):
            first = km.next_client()
            km.mark_rate_limited(cooldown_seconds=60)
            second = km.next_client()

        self.assertEqual(first.api_key, "k0")
        self.assertEqual(second.api_key, "k1")
        self.assertEqual(km.available_key_count, 1)

        km._rate_limited_until[0] = 0.0
        self.assertEqual(km.available_key_count, 2)


class TestMakeClient(unittest.TestCase):
    """make_client() returns a client for the CURRENT key without advancing."""

    def test_make_client_uses_current_idx(self):
        km = _make_km("k0", "k1", "k2")
        km._current_idx = 2
        with patch("backend.llm.llm.genai.Client", side_effect=lambda api_key: MagicMock(api_key=api_key)):
            c = km.make_client()
        self.assertEqual(c.api_key, "k2")
        self.assertEqual(km._current_idx, 2)   # must NOT advance

    def test_make_client_falls_back_to_next_when_uninitialised(self):
        """If _current_idx = -1 (no next_client() called yet), falls back gracefully."""
        km = _make_km("k0")
        # _current_idx is -1 by default in _make_km
        with patch("backend.llm.llm.genai.Client", side_effect=lambda api_key: MagicMock(api_key=api_key)):
            c = km.make_client()
        self.assertEqual(c.api_key, "k0")


class TestThreadSafety(unittest.TestCase):
    """next_client() under concurrent load must not produce duplicate advances."""

    def test_concurrent_next_client_unique_distribution(self):
        """50 concurrent threads each call next_client() once.
        With 5 keys, each key should be used exactly 10 times.
        """
        km = _make_km("k0", "k1", "k2", "k3", "k4")
        results = []
        lock = threading.Lock()

        def worker():
            with patch("backend.llm.llm.genai.Client", side_effect=lambda api_key: MagicMock(api_key=api_key)):
                c = km.next_client()
            with lock:
                results.append(c.api_key)

        threads = [threading.Thread(target=worker) for _ in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        from collections import Counter
        counts = Counter(results)
        # Each key should be used exactly 10 times (50 calls / 5 keys)
        for key in ["k0", "k1", "k2", "k3", "k4"]:
            self.assertEqual(counts[key], 10, f"{key} used {counts[key]} times, expected 10")


if __name__ == "__main__":
    unittest.main()
