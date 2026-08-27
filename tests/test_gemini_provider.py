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

    def test_gemini_3_uses_thinking_level_not_budget(self):
        """Probed live: gemini-3.6-flash returns 400 INVALID_ARGUMENT for
        thinking_budget=0 -- and 400 is INVALID_REQUEST, which is not
        fallback-eligible, so it would abort the verifier stage instead of
        deferring to the next deployment."""
        config = GeminiProvider._thinking_config(LLMStage.VERIFIER, "gemini-3.6-flash")
        self.assertEqual(config.thinking_level, "MINIMAL")
        self.assertIsNone(config.thinking_budget)

    def test_gemini_3_7_gets_the_lowest_level_it_accepts(self):
        """gemini-3.7-flash rejects MINIMAL and *ignores* thinking_budget=0 --
        248 thinking tokens against a 256-token ceiling, finish_reason
        MAX_TOKENS. LOW is the floor it honours."""
        config = GeminiProvider._thinking_config(LLMStage.VERIFIER, "gemini-3.7-flash")
        self.assertEqual(config.thinking_level, "LOW")

    def test_gemini_2_still_uses_the_budget_knob(self):
        """thinking_level is rejected outright on 2.x."""
        config = GeminiProvider._thinking_config(LLMStage.VERIFIER, "gemini-2.5-flash")
        self.assertEqual(config.thinking_budget, 0)
        self.assertIsNone(config.thinking_level)

    def test_optimizer_disables_thinking(self):
        self.assertIsNotNone(GeminiProvider._thinking_config(LLMStage.OPTIMIZER))

    def test_draft_keeps_thinking_on_2_x(self):
        """Synthesis is where reasoning earns its token cost, and 2.5-flash
        spends it modestly enough to be left alone."""
        self.assertIsNone(GeminiProvider._thinking_config(LLMStage.DRAFT, "gemini-2.5-flash"))

    def test_unknown_draft_model_is_left_alone(self):
        """No model name (older call sites, tests) must behave like 2.x."""
        self.assertIsNone(GeminiProvider._thinking_config(LLMStage.DRAFT))

    def test_gemini_3_draft_is_capped_but_still_reasons(self):
        """A live qa draft on gemini-3-flash-preview returned MAX_TOKENS inside
        the 3,000-token ceiling: the 3.x default reasoning pass ate the answer's
        budget.  LOW still reasons (81 tokens probed); MINIMAL would be off."""
        for model in ("gemini-3-flash-preview", "gemini-3.6-flash", "gemini-3.7-flash"):
            with self.subTest(model=model):
                config = GeminiProvider._thinking_config(LLMStage.DRAFT, model)
                self.assertEqual(config.thinking_level, "LOW")
                self.assertIsNone(config.thinking_budget)

    def test_older_sdk_without_thinking_config_is_tolerated(self):
        """An SDK predating the thinking budget must not break generation."""
        from google.genai import types as genai_types

        with patch.object(genai_types, "ThinkingConfig", None):
            self.assertIsNone(GeminiProvider._thinking_config(LLMStage.VERIFIER))


class TestGenerateWiring(unittest.TestCase):
    def _call(self, stage: LLMStage, model: str = "gemini-2.5-flash"):
        response = MagicMock()
        response.usage_metadata.prompt_token_count = 10
        response.usage_metadata.candidates_token_count = 5
        client = MagicMock()
        client.models.generate_content.return_value = response

        key_manager = MagicMock()
        key_manager.next_client.return_value = client
        provider = GeminiProvider(lambda: key_manager, lambda r: ("text", False))

        with patch.object(GeminiProvider, "_generate_config", side_effect=lambda **kw: kw):
            provider.generate(_request(stage), model)
        return client.models.generate_content.call_args.kwargs["config"]

    def test_verifier_call_carries_a_thinking_config(self):
        self.assertIn("thinking_config", self._call(LLMStage.VERIFIER))

    def test_2_x_draft_call_does_not(self):
        self.assertNotIn("thinking_config", self._call(LLMStage.DRAFT))

    def test_3_x_draft_call_carries_the_cap(self):
        """The knob has to survive the trip into generate_content(); the two
        defects this mechanism has already had were both in the wiring, not in
        the decision."""
        config = self._call(LLMStage.DRAFT, "gemini-3.6-flash")
        self.assertEqual(config["thinking_config"].thinking_level, "LOW")

    def test_the_deadline_reaches_the_sdk_as_a_timeout(self):
        """The router clamps deadline_seconds per deployment and per request
        budget, and then nothing passed it on: for Gemini it bounded nothing.
        A live gemini-3.7-flash draft ran 200s with no result and no error,
        with the whole failover chain stuck behind it. google-genai takes
        milliseconds."""
        config = self._call(LLMStage.DRAFT)
        self.assertEqual(config["http_options"]["timeout"], 30_000)

    def test_core_generation_params_are_always_passed(self):
        config = self._call(LLMStage.DRAFT)
        self.assertEqual(config["temperature"], 0.0)
        self.assertEqual(config["max_output_tokens"], 3000)
        self.assertEqual(config["system_instruction"], "system")


class TestErrorNormalization(unittest.TestCase):
    def test_the_sdk_timeout_spelling_is_fallback_eligible(self):
        """Observed live once the deadline actually reached the SDK:
        "504 DEADLINE_EXCEEDED". The spaced marker missed the underscore and
        504 is not in the 500/502/503 rule, so it normalised to UNKNOWN --
        which is not fallback-eligible, so the stage stopped with two healthy
        deployments untried."""
        from backend.llm.contracts import LLMErrorCategory
        from backend.llm.gemini_provider import normalize_provider_error
        from backend.llm.router import FALLBACK_ELIGIBLE

        for message in ("504 DEADLINE_EXCEEDED", "deadline exceeded", "Read timed out"):
            with self.subTest(message=message):
                error = normalize_provider_error(Exception(message), "gemini")
                self.assertEqual(error.category, LLMErrorCategory.TIMEOUT)
                self.assertIn(error.category, FALLBACK_ELIGIBLE)


class TestRegistryBudget(unittest.TestCase):
    def test_draft_ceiling_fits_a_full_smart_summary_plus_thinking(self):
        from backend.llm.llm import SMART_SUMMARY_MAX_TOKENS
        from backend.llm.router import load_registry

        cap = load_registry()["gemini-primary"].max_output_tokens
        self.assertGreaterEqual(cap, SMART_SUMMARY_MAX_TOKENS)
        self.assertGreaterEqual(SMART_SUMMARY_MAX_TOKENS, 4000)


if __name__ == "__main__":
    unittest.main()


class TestModelSelection(unittest.TestCase):
    """The backup model list had gone stale: every entry was retired, so
    discovery fell through to a default that was also retired and 404'd."""

    def test_exact_match_beats_prefix_match(self):
        from backend.llm.llm import _select_model

        available = ["gemini-2.5-flash-image", "gemini-2.5-flash", "gemini-2.5-flash-lite"]
        self.assertEqual(_select_model(["gemini-2.5-flash"], available, "fallback"), "gemini-2.5-flash")

    def test_prefix_match_is_deterministic(self):
        from backend.llm.llm import _select_model

        available = ["gemini-9-flash-tts", "gemini-9-flash-image"]
        first = _select_model(["gemini-9-flash"], available, "fallback")
        second = _select_model(["gemini-9-flash"], list(reversed(available)), "fallback")
        self.assertEqual(first, second)

    def test_priority_order_is_respected(self):
        from backend.llm.llm import _select_model

        available = ["model-b", "model-a"]
        self.assertEqual(_select_model(["model-a", "model-b"], available, "x"), "model-a")

    def test_falls_back_to_default_when_nothing_matches(self):
        from backend.llm.llm import _select_model

        self.assertEqual(_select_model(["nope"], ["other"], "the-default"), "the-default")

    def test_backup_priority_does_not_name_retired_models(self):
        from backend.llm.llm import _BACKUP_PRIORITY, _DEFAULT_BACKUP

        retired = {"gemini-1.5-flash", "gemini-1.5-flash-8b", "gemini-2.0-flash-lite", "gemini-2.0-flash"}
        self.assertFalse(retired & set(_BACKUP_PRIORITY))
        self.assertNotIn(_DEFAULT_BACKUP, retired)
