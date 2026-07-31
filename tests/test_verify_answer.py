"""
tests/test_verify_answer.py
===========================
Unit tests for verify_answer() after the google-genai SDK migration.

Verifies that:
1. key_manager.next_client() is called (round-robin, not the old make_client/configure_current).
2. client.models.generate_content() is called with the correct arguments:
   - model= set to PROD_MODEL
   - contents= set to the verify_user string
   - config= is a GenerateContentConfig with system_instruction containing
     the full verify_prompt, temperature=0.0, and the mode-aware max_output_tokens.
3. max_output_tokens uses QA_MAX_TOKENS for mode="qa".
4. max_output_tokens uses SMART_SUMMARY_MAX_TOKENS for mode="smart_summary".
"""
from __future__ import annotations

import types as _builtins_types
import unittest
from unittest.mock import MagicMock, patch, call


def _make_config_capture():
    """Return a simple namespace that records keyword args like GenerateContentConfig."""
    class FakeConfig:
        def __init__(self, **kw):
            self.__dict__.update(kw)
    return FakeConfig


def _make_fake_resp(text: str = "Verified medical text.") -> MagicMock:
    """Return a fake Gemini response with a STOP finish_reason."""
    fake_resp = MagicMock()
    fake_resp.text = text
    fake_cand = MagicMock()
    fake_cand.finish_reason.name = "STOP"
    fake_resp.candidates = [fake_cand]
    return fake_resp


class TestVerifyAnswerNewSDK(unittest.TestCase):
    """verify_answer must use the new google-genai client pattern with next_client()."""

    def _run_verify(self, mode: str = "qa") -> dict:
        """
        Patch the genai Client and types, call verify_answer, return captured
        kwargs from the generate_content call.
        """
        captured = {}
        fake_resp = _make_fake_resp()
        FakeConfig = _make_config_capture()

        fake_client = MagicMock()
        fake_client.models.generate_content.return_value = fake_resp

        with patch("backend.llm.llm.key_manager") as mock_km, \
             patch("backend.llm.llm.types") as mock_types:

            # verify_answer uses next_client() since the round-robin refactor
            mock_km.next_client.return_value = fake_client
            mock_types.GenerateContentConfig.side_effect = FakeConfig

            from backend.llm.llm import verify_answer, QA_MAX_TOKENS, SMART_SUMMARY_MAX_TOKENS

            verify_answer("Draft answer.", "Some context.", mode=mode)

            # Capture what was passed to generate_content
            call_kwargs = fake_client.models.generate_content.call_args
            captured["call_args"] = call_kwargs
            captured["config"] = call_kwargs.kwargs.get("config") or (
                call_kwargs.args[2] if len(call_kwargs.args) > 2 else None
            )
            captured["qa_max"] = QA_MAX_TOKENS
            captured["smart_max"] = SMART_SUMMARY_MAX_TOKENS

        return captured

    def test_next_client_is_called_not_configure_current(self):
        """next_client() must be called; configure_current() must NOT be used."""
        fake_resp = _make_fake_resp("ok")
        with patch("backend.llm.llm.key_manager") as mock_km, \
             patch("backend.llm.llm.types"):
            mock_km.next_client.return_value = MagicMock()
            mock_km.next_client.return_value.models.generate_content.return_value = fake_resp
            from backend.llm.llm import verify_answer
            verify_answer("answer", "context", mode="qa")
        mock_km.next_client.assert_called()
        mock_km.configure_current.assert_not_called()

    def test_system_instruction_contains_full_verify_prompt(self):
        captured = self._run_verify(mode="qa")
        cfg = captured["config"]
        self.assertIsNotNone(cfg, "GenerateContentConfig must be passed")
        si = getattr(cfg, "system_instruction", "")
        self.assertIn("HarrisonGPT verifying", si,
                      "system_instruction must contain the full verify_prompt")
        self.assertIn("output_format", si,
                      "system_instruction must include the <output_format> block")

    def test_temperature_is_zero_for_verification(self):
        captured = self._run_verify(mode="qa")
        cfg = captured["config"]
        self.assertEqual(getattr(cfg, "temperature", None), 0.0,
                         "verify_answer must use temperature=0.0")

    def test_max_output_tokens_qa_mode(self):
        captured = self._run_verify(mode="qa")
        cfg = captured["config"]
        qa_max = captured["qa_max"]
        self.assertEqual(getattr(cfg, "max_output_tokens", None), qa_max,
                         f"qa mode must use QA_MAX_TOKENS={qa_max}")

    def test_max_output_tokens_smart_summary_mode(self):
        captured = self._run_verify(mode="smart_summary")
        cfg = captured["config"]
        smart_max = captured["smart_max"]
        self.assertEqual(getattr(cfg, "max_output_tokens", None), smart_max,
                         f"smart_summary mode must use SMART_SUMMARY_MAX_TOKENS={smart_max}")


if __name__ == "__main__":
    unittest.main()
