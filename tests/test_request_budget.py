"""
tests/test_request_budget.py
============================
Nothing used to bound one /ask request end to end.  The draft and verifier
stages each get 60s and each retries, so the deadline structure alone
permitted several minutes for a single question.

``start_request_budget()`` opens a wall-clock ceiling; stage deadlines clamp
against what is left of it and retries stop once it is spent.
"""
from __future__ import annotations

import time
import unittest
from unittest.mock import MagicMock, patch

from backend.llm import llm as llm_mod
from backend.llm.contracts import LLMError, LLMErrorCategory
from backend.observability import remaining_budget, request_deadline_var, start_request_budget


def _spend_the_budget() -> None:
    """A budget goes negative by elapsing, not by being opened negative —
    start_request_budget() reads a non-positive argument as "no budget"."""
    request_deadline_var.set(time.monotonic() - 1.0)


class TestBudgetWindow(unittest.TestCase):
    def setUp(self):
        self._token = request_deadline_var.set(None)

    def tearDown(self):
        request_deadline_var.reset(self._token)

    def test_no_budget_means_no_ceiling(self):
        self.assertIsNone(remaining_budget())

    def test_zero_disables_the_budget(self):
        start_request_budget(0)
        self.assertIsNone(remaining_budget())

    def test_a_budget_counts_down(self):
        start_request_budget(30)
        remaining = remaining_budget()
        self.assertIsNotNone(remaining)
        self.assertLessEqual(remaining, 30)
        self.assertGreater(remaining, 29)


class TestStageDeadlineClamp(unittest.TestCase):
    def setUp(self):
        self._token = request_deadline_var.set(None)

    def tearDown(self):
        request_deadline_var.reset(self._token)

    def test_unbudgeted_stages_keep_their_own_deadline(self):
        self.assertEqual(llm_mod._stage_deadline(60.0), 60.0)

    def test_a_roomy_budget_does_not_shorten_the_stage(self):
        start_request_budget(300)
        self.assertEqual(llm_mod._stage_deadline(60.0), 60.0)

    def test_a_tight_budget_shortens_the_stage(self):
        start_request_budget(10)
        self.assertLessEqual(llm_mod._stage_deadline(60.0), 10.0)

    def test_a_spent_budget_still_leaves_a_usable_floor(self):
        """A sub-second deadline just burns a provider round-trip."""
        _spend_the_budget()
        self.assertEqual(llm_mod._stage_deadline(60.0), llm_mod._MIN_STAGE_DEADLINE_SECONDS)


class TestRetriesStopWhenTheBudgetIsSpent(unittest.TestCase):
    def setUp(self):
        self._token = request_deadline_var.set(None)

    def tearDown(self):
        request_deadline_var.reset(self._token)

    def _retryable(self):
        return LLMError(LLMErrorCategory.UNAVAILABLE, "503 UNAVAILABLE")

    def test_a_retryable_error_retries_with_budget_left(self):
        start_request_budget(300)
        with patch.object(llm_mod.time, "sleep"):
            self.assertTrue(llm_mod._handle_retryable(self._retryable(), 0, 3, "ask_llm"))

    def test_a_retryable_error_does_not_retry_once_the_budget_is_spent(self):
        _spend_the_budget()
        with patch.object(llm_mod.time, "sleep") as sleep:
            self.assertFalse(llm_mod._handle_retryable(self._retryable(), 0, 3, "ask_llm"))
        sleep.assert_not_called()

    def test_ask_llm_gives_up_instead_of_burning_every_attempt(self):
        router = MagicMock()
        router.generate_for_stage.side_effect = self._retryable()
        _spend_the_budget()
        with patch.object(llm_mod, "llm_router", router), \
             patch.object(llm_mod.time, "sleep"), \
             patch.object(llm_mod, "prod_model", return_value="gemini-2.5-flash"), \
             patch.object(llm_mod, "DRAFT_MAX_ATTEMPTS", 3):
            _, _, _, path = llm_mod.ask_llm(fused_context="A" * 200, question="test")

        self.assertEqual(router.generate_for_stage.call_count, 1)
        self.assertEqual(path, "provider_failure")


if __name__ == "__main__":
    unittest.main()
