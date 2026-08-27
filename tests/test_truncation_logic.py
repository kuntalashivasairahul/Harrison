"""Compatibility regression tests for LLM truncation return paths."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from backend.llm import llm


def _response(text: str, finish_reason: str) -> MagicMock:
    response = MagicMock()
    response.text = text
    response.candidates[0].finish_reason.name = finish_reason
    response.candidates[0].content.parts = [MagicMock(text=text)]
    return response


class _Config:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs)


class TestTruncationLogic(unittest.TestCase):
    def _ask(self, responses):
        client = MagicMock()
        if isinstance(responses, list):
            client.models.generate_content.side_effect = responses
        else:
            # One response reused for every call. A truncated result now
            # escalates to the next deployment, so a stage that keeps truncating
            # walks its whole chain and a fixed-length script runs dry mid-chain.
            client.models.generate_content.side_effect = lambda *_a, **_kw: responses
        with patch("backend.llm.llm.key_manager") as key_manager, \
             patch("backend.llm.llm.types") as types:
            key_manager.next_client.return_value = client
            types.GenerateContentConfig.side_effect = _Config
            return llm.ask_llm(
                fused_context="Valid medical context of length greater than twenty characters.",
                question="What is the treatment?",
                mode="qa",
            )

    def test_complete_verifier_retry_returns_verified_path(self):
        final, draft, was_truncated, path = self._ask([
            _response("This is a complete draft answer.", "STOP"),
            _response("This is a truncated verified text", "MAX_TOKENS"),
            _response("This is a fully complete verified text.", "STOP"),
        ])
        self.assertEqual(final, "This is a fully complete verified text.")
        self.assertEqual(draft, "This is a complete draft answer.")
        self.assertFalse(was_truncated)
        self.assertEqual(path, "verified")

    def test_truncated_verifier_returns_complete_draft_fallback(self):
        final, _, was_truncated, path = self._ask([
            _response("This is a complete draft answer.", "STOP"),
            _response("This is a truncated verified text", "MAX_TOKENS"),
            _response("This is a truncated verified text", "MAX_TOKENS"),
        ])
        self.assertEqual(final, "This is a complete draft answer.")
        self.assertFalse(was_truncated)
        self.assertEqual(path, "draft_fallback")

    def test_double_truncation_returns_marked_partial_answer(self):
        final, _, was_truncated, path = self._ask(
            _response("This is a truncated verified text", "MAX_TOKENS")
        )
        self.assertIn("incomplete due to length constraints", final)
        self.assertTrue(was_truncated)
        self.assertEqual(path, "graceful_fallback")


if __name__ == "__main__":
    unittest.main()
