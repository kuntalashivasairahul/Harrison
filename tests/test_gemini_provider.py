"""
tests/test_gemini_provider.py
=============================
Gemini adapter behaviour, notably the thinking budget.

Gemini 2.5 models run an internal reasoning pass by default and those tokens
come out of ``max_output_tokens``. A trivial verifier call measured 561
thinking tokens against 17 tokens of answer. On a real smart_summary the
verifier must reproduce a ~1,660-token draft, so thinking pushed it past the
ceiling and it returned MAX_TOKENS — every smart summary then took the
draft_fallback path and was capped at Medium confidence, never actually
verified.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from backend.llm.contracts import LLMRequest, LLMStage
from backend.llm.gemini_provider import GeminiProvider


def _request(stage: LLMStage) -> LLMRequest:
    return LLMRequest("prompt", "system", "gemini-primary", 0.0, 3000, 30.0, stage)


class TestThinkingBudget(unittest.TestCase):
    def test_verifier_disables_thinking(self):
        config = GeminiProvider._thinking_config(LLMStage.VERIFIER)
        self.assertIsNotNone(config)
        self.assertEqual(config.thinking_budget, 0)

    def test_optimizer_disables_thinking(self):
        self.assertIsNotNone(GeminiProvider._thinking_config(LLMStage.OPTIMIZER))

    def test_draft_keeps_thinking(self):
        """Synthesis is where reasoning earns its token cost."""
        self.assertIsNone(GeminiProvider._thinking_config(LLMStage.DRAFT))

    def test_older_sdk_without_thinking_config_is_tolerated(self):
        """An SDK predating the thinking budget must not break generation."""
        from google.genai import types as genai_types

        with patch.object(genai_types, "ThinkingConfig", None):
            self.assertIsNone(GeminiProvider._thinking_config(LLMStage.VERIFIER))


class TestGenerateWiring(unittest.TestCase):
    def _call(self, stage: LLMStage):
        response = MagicMock()
        response.usage_metadata.prompt_token_count = 10
        response.usage_metadata.candidates_token_count = 5
        client = MagicMock()
        client.models.generate_content.return_value = response

        key_manager = MagicMock()
        key_manager.next_client.return_value = client
        provider = GeminiProvider(lambda: key_manager, lambda r: ("text", False))

        with patch.object(GeminiProvider, "_generate_config", side_effect=lambda **kw: kw):
            provider.generate(_request(stage), "gemini-2.5-flash")
        return client.models.generate_content.call_args.kwargs["config"]

    def test_verifier_call_carries_a_thinking_config(self):
        self.assertIn("thinking_config", self._call(LLMStage.VERIFIER))

    def test_draft_call_does_not(self):
        self.assertNotIn("thinking_config", self._call(LLMStage.DRAFT))

    def test_core_generation_params_are_always_passed(self):
        config = self._call(LLMStage.DRAFT)
        self.assertEqual(config["temperature"], 0.0)
        self.assertEqual(config["max_output_tokens"], 3000)
        self.assertEqual(config["system_instruction"], "system")


class TestRegistryBudget(unittest.TestCase):
    def test_draft_ceiling_fits_a_full_smart_summary_plus_thinking(self):
        from backend.llm.llm import SMART_SUMMARY_MAX_TOKENS
        from backend.llm.router import load_registry

        cap = load_registry()["gemini-primary"].max_output_tokens
        self.assertGreaterEqual(cap, SMART_SUMMARY_MAX_TOKENS)
        self.assertGreaterEqual(SMART_SUMMARY_MAX_TOKENS, 4000)


if __name__ == "__main__":
    unittest.main()
