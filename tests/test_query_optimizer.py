"""No-network tests for QueryOptimizer routing and deterministic fallback."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from backend.agents import query_optimizer
from backend.llm.contracts import LLMError, LLMErrorCategory, LLMResult


def _valid_result(provider: str = "groq") -> LLMResult:
    return LLMResult(
        text='{"is_medical_query": true, "expanded_query": "diabetic ketoacidosis diagnosis and treatment", "focus": "management", "complexity": "complex"}',
        provider=provider,
        model="test-model",
    )


class TestQueryOptimizerRouting(unittest.TestCase):
    def setUp(self) -> None:
        self.original_router = query_optimizer.llm_router
        query_optimizer.llm_router = MagicMock()

    def tearDown(self) -> None:
        query_optimizer.llm_router = self.original_router

    def test_groq_success_uses_router_result(self) -> None:
        query_optimizer.llm_router.optimize.return_value = _valid_result("groq")

        result = query_optimizer.optimize_query("DKA treatment")

        self.assertTrue(result["optimizer_used"])
        self.assertIn("diabetic ketoacidosis", result["expanded_query"])
        query_optimizer.llm_router.optimize.assert_called_once()

    def test_router_gemini_fallback_result_is_accepted(self) -> None:
        query_optimizer.llm_router.optimize.return_value = _valid_result("gemini")

        result = query_optimizer.optimize_query("DKA treatment")

        self.assertTrue(result["optimizer_used"])
        query_optimizer.llm_router.optimize.assert_called_once()

    def test_groq_and_gemini_failure_returns_deterministic_fallback(self) -> None:
        query_optimizer.llm_router.optimize.side_effect = LLMError(
            LLMErrorCategory.RATE_LIMITED,
            "all optimizer deployments cooling down",
        )

        result = query_optimizer.optimize_query("DKA treatment")

        self.assertFalse(result["optimizer_used"])
        self.assertEqual(result["expanded_query"], "DKA treatment")
        query_optimizer.llm_router.optimize.assert_called_once()

    def test_invalid_response_returns_deterministic_fallback(self) -> None:
        query_optimizer.llm_router.optimize.return_value = LLMResult(
            text="not JSON", provider="groq", model="test-model"
        )

        result = query_optimizer.optimize_query("What causes chest pain?")

        self.assertFalse(result["optimizer_used"])
        self.assertEqual(result["expanded_query"], "What causes chest pain?")


if __name__ == "__main__":
    unittest.main()
