"""No-network tests for QueryOptimizer key rotation and fallback behavior."""

from __future__ import annotations

import types
import unittest

from backend.agents import query_optimizer


class TestQueryOptimizerRetries(unittest.TestCase):
    def setUp(self) -> None:
        self.original_key_manager = query_optimizer.key_manager

    def tearDown(self) -> None:
        query_optimizer.key_manager = self.original_key_manager

    def test_quota_error_marks_key_then_retries_with_next_client(self) -> None:
        calls: list[str] = []

        class FakeModels:
            def __init__(self) -> None:
                self.attempts = 0

            def generate_content(self, **_kwargs):
                self.attempts += 1
                if self.attempts == 1:
                    raise RuntimeError("429 RESOURCE_EXHAUSTED: quota exceeded")
                return types.SimpleNamespace(
                    text=(
                        '{"is_medical_query": true, '
                        '"expanded_query": "diabetic ketoacidosis management", '
                        '"focus": "management", "complexity": "complex"}'
                    )
                )

        class FakeClient:
            def __init__(self, models) -> None:
                self.models = models

        class FakeKeyManager:
            def __init__(self) -> None:
                self.models = FakeModels()

            def next_client(self):
                calls.append("next_client")
                return FakeClient(self.models)

            def mark_exhausted(self) -> None:
                calls.append("mark_exhausted")

        query_optimizer.key_manager = FakeKeyManager()

        result = query_optimizer.optimize_query("DKA treatment")

        self.assertTrue(result["optimizer_used"])
        self.assertEqual(result["expanded_query"], "diabetic ketoacidosis management")
        self.assertEqual(calls, ["next_client", "mark_exhausted", "next_client"])

    def test_non_quota_error_returns_fallback_without_marking_key(self) -> None:
        calls: list[str] = []

        class FakeModels:
            def generate_content(self, **_kwargs):
                raise RuntimeError("connection reset by peer")

        class FakeClient:
            models = FakeModels()

        class FakeKeyManager:
            def next_client(self):
                calls.append("next_client")
                return FakeClient()

            def mark_exhausted(self) -> None:
                calls.append("mark_exhausted")

        query_optimizer.key_manager = FakeKeyManager()

        result = query_optimizer.optimize_query("What causes chest pain?")

        self.assertFalse(result["optimizer_used"])
        self.assertEqual(result["expanded_query"], "What causes chest pain?")
        self.assertEqual(calls, ["next_client"])


if __name__ == "__main__":
    unittest.main()
