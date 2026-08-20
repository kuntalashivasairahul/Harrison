"""
tests/test_transient_retry.py
=============================
Transient provider failures must consume the retry budget, not bypass it.

Found by live testing, not by the suite: Google returned
`503 UNAVAILABLE — this model is currently experiencing high demand` and every
clinical query failed in ~2 seconds. `ask_llm()` retried only on RATE_LIMITED,
so a transient 503 broke out of the loop on the first attempt while the router
was logging `fallback_eligible=True` for exactly that category. The mocked
suite never sees a 503, which is why this survived.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from backend.llm import llm as llm_mod
from backend.llm.contracts import LLMError, LLMErrorCategory


class TestHandleRetryable(unittest.TestCase):
    def setUp(self):
        self._sleep = patch.object(llm_mod.time, "sleep")
        self.sleep = self._sleep.start()

    def tearDown(self):
        self._sleep.stop()

    def _err(self, category, retry_after=None):
        return LLMError(category, "boom", retry_after_seconds=retry_after)

    def test_unavailable_is_retried(self):
        self.assertTrue(
            llm_mod._handle_retryable(self._err(LLMErrorCategory.UNAVAILABLE), 0, 3, "ask_llm")
        )

    def test_timeout_is_retried(self):
        self.assertTrue(
            llm_mod._handle_retryable(self._err(LLMErrorCategory.TIMEOUT), 0, 3, "ask_llm")
        )

    def test_rate_limited_is_retried(self):
        with patch.object(llm_mod, "key_manager") as km:
            self.assertTrue(
                llm_mod._handle_retryable(self._err(LLMErrorCategory.RATE_LIMITED), 0, 3, "ask_llm")
            )
            km.mark_rate_limited.assert_called_once()

    def test_auth_is_not_retried(self):
        """A bad key will not fix itself; failing fast is correct."""
        self.assertFalse(
            llm_mod._handle_retryable(self._err(LLMErrorCategory.AUTH), 0, 3, "ask_llm")
        )

    def test_invalid_request_is_not_retried(self):
        self.assertFalse(
            llm_mod._handle_retryable(self._err(LLMErrorCategory.INVALID_REQUEST), 0, 3, "ask_llm")
        )

    def test_last_attempt_does_not_retry(self):
        self.assertFalse(
            llm_mod._handle_retryable(self._err(LLMErrorCategory.UNAVAILABLE), 2, 3, "ask_llm")
        )

    def test_quota_errors_rotate_instead_of_sleeping(self):
        with patch.object(llm_mod, "key_manager"):
            llm_mod._handle_retryable(self._err(LLMErrorCategory.RATE_LIMITED), 0, 3, "ask_llm")
        self.sleep.assert_not_called()

    def test_transient_errors_back_off_exponentially(self):
        llm_mod._handle_retryable(self._err(LLMErrorCategory.UNAVAILABLE), 0, 4, "ask_llm")
        llm_mod._handle_retryable(self._err(LLMErrorCategory.UNAVAILABLE), 1, 4, "ask_llm")
        delays = [c.args[0] for c in self.sleep.call_args_list]
        self.assertEqual(len(delays), 2)
        self.assertGreater(delays[1], delays[0])

    def test_provider_retry_after_wins_over_backoff(self):
        llm_mod._handle_retryable(self._err(LLMErrorCategory.UNAVAILABLE, retry_after=7.0), 0, 3, "ask_llm")
        self.sleep.assert_called_once_with(7.0)


class TestAskLlmUsesTheWholeBudget(unittest.TestCase):
    def test_a_503_consumes_every_attempt_before_giving_up(self):
        """The live symptom: one attempt, then the generic error fallback."""
        router = MagicMock()
        router.generate_for_stage.side_effect = LLMError(
            LLMErrorCategory.UNAVAILABLE, "503 UNAVAILABLE high demand"
        )
        with patch.object(llm_mod, "llm_router", router), \
             patch.object(llm_mod.time, "sleep"), \
             patch.object(llm_mod, "prod_model", return_value="gemini-2.5-flash"), \
             patch.object(llm_mod, "DRAFT_MAX_ATTEMPTS", 3):
            answer, draft, truncated, path = llm_mod.ask_llm(
                fused_context="A" * 200, question="test"
            )

        self.assertEqual(router.generate_for_stage.call_count, 3)
        self.assertEqual(path, "error_fallback")

    def test_a_transient_failure_that_recovers_returns_a_real_answer(self):
        result = MagicMock(text="recovered answer", finish_reason="STOP")
        router = MagicMock()
        router.generate_for_stage.side_effect = [
            LLMError(LLMErrorCategory.UNAVAILABLE, "503"),
            result,
            result,
        ]
        with patch.object(llm_mod, "llm_router", router), \
             patch.object(llm_mod.time, "sleep"), \
             patch.object(llm_mod, "prod_model", return_value="gemini-2.5-flash"), \
             patch.object(llm_mod, "DRAFT_MAX_ATTEMPTS", 3):
            answer, draft, truncated, path = llm_mod.ask_llm(
                fused_context="A" * 200, question="test", disable_verifier=True
            )

        self.assertEqual(draft, "recovered answer")
        self.assertNotEqual(path, "error_fallback")


if __name__ == "__main__":
    unittest.main()
