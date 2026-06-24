"""
tests/test_verify_answer.py
===========================
Unit tests for the verify_answer fix in backend/llm/llm.py.

Verifies that:
1. system_instruction is populated with the full verify_prompt (not the old one-liner).
2. max_output_tokens uses the mode-aware limit (QA_MAX_TOKENS / SMART_SUMMARY_MAX_TOKENS),
   not the hardcoded 1024.
"""
from __future__ import annotations

import types
import unittest
from unittest.mock import MagicMock, patch, call


class TestVerifyAnswerFix(unittest.TestCase):
    """verify_answer must pass full instructions and mode-aware token limit."""

    def _make_fake_resp(self, text: str):
        resp = MagicMock()
        resp.text = text
        return resp

    def _run_verify(self, mode: str = "qa") -> dict:
        """
        Call verify_answer with a fake genai client and return the captured
        kwargs used in GenerativeModel.__init__ and generate_content.
        """
        captured = {}

        class FakeModel:
            def __init__(self, model_name, system_instruction):
                captured["system_instruction"] = system_instruction

            def generate_content(self, user_msg, generation_config=None):
                captured["max_output_tokens"] = generation_config.max_output_tokens
                resp = MagicMock()
                resp.text = "Verified medical text."
                return resp

        with patch("backend.llm.llm.genai") as mock_genai, \
             patch("backend.llm.llm.key_manager") as mock_km:
            mock_genai.GenerativeModel.side_effect = FakeModel
            mock_genai.types.GenerationConfig = types.SimpleNamespace(
                __call__=lambda self, **kw: types.SimpleNamespace(**kw)
            )
            # Make GenerationConfig a real callable
            mock_genai.types = MagicMock()
            mock_genai.types.GenerationConfig = lambda **kw: types.SimpleNamespace(**kw)

            from backend.llm.llm import verify_answer, QA_MAX_TOKENS, SMART_SUMMARY_MAX_TOKENS, verify_answer
            verify_answer("Draft answer.", "Some context.", mode=mode)

        return captured, QA_MAX_TOKENS, SMART_SUMMARY_MAX_TOKENS

    def test_system_instruction_contains_full_verify_prompt(self):
        captured, _, _ = self._run_verify(mode="qa")
        si = captured.get("system_instruction", "")
        # The verify_prompt must contain key distinctive phrases
        self.assertIn("HarrisonGPT verifying", si,
                      "system_instruction should contain full verify_prompt")
        self.assertIn("output_format", si,
                      "system_instruction should contain the <output_format> block")
        # The old one-liner should NOT be the only instruction
        self.assertNotEqual(
            si,
            "You are a silent verification filter. Output ONLY corrected medical text. "
            "Never output meta-commentary, analysis, or references to the verification process.",
            "system_instruction must be the full verify_prompt, not the old one-liner",
        )

    def test_max_output_tokens_uses_qa_limit(self):
        captured, qa_max, _ = self._run_verify(mode="qa")
        self.assertEqual(
            captured.get("max_output_tokens"), qa_max,
            f"qa mode must use QA_MAX_TOKENS={qa_max}, not hardcoded 1024",
        )
        self.assertNotEqual(captured.get("max_output_tokens"), 1024,
                            "The hardcoded 1024 must be replaced with max_tokens")

    def test_max_output_tokens_uses_smart_summary_limit(self):
        captured, _, smart_max = self._run_verify(mode="smart_summary")
        self.assertEqual(
            captured.get("max_output_tokens"), smart_max,
            f"smart_summary mode must use SMART_SUMMARY_MAX_TOKENS={smart_max}",
        )


if __name__ == "__main__":
    unittest.main()
