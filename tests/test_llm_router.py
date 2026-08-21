"""No-network tests for registry validation, routing policy, and cooldowns."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.llm.contracts import LLMError, LLMErrorCategory, LLMRequest, LLMResult, LLMStage
from backend.llm.router import FALLBACK_ELIGIBLE, LLMRouter, load_registry


def _request() -> LLMRequest:
    return LLMRequest("prompt", "system", "optimizer", 0.0, 32, 5.0, LLMStage.OPTIMIZER)


class TestRegistry(unittest.TestCase):
    def test_default_registry_has_only_approved_stage_one_deployments(self) -> None:
        registry = load_registry()
        self.assertEqual(
            set(registry),
            {
                "gemini-primary", "gemini-draft-fallback", "gemini-optimizer-fallback",
                "gemini-flash-3.7", "gemini-flash-3.6", "gemini-flash-3",
                "groq-optimizer", "groq-draft", "mistral-draft",
            },
        )
        self.assertEqual(registry["gemini-primary"].stages, (LLMStage.DRAFT, LLMStage.VERIFIER))
        # The draft stage must have somewhere to fall back to.
        self.assertEqual(registry["gemini-draft-fallback"].stages, (LLMStage.DRAFT, LLMStage.VERIFIER))
        self.assertGreater(
            registry["gemini-draft-fallback"].priority, registry["gemini-primary"].priority
        )

    def test_each_gemini_draft_pins_a_distinct_model(self) -> None:
        """Free-tier Gemini quota is metered per project *per model*, so the
        draft chain buys capacity only if every deployment names a different
        model. Two aliases on one model would share one 20/day bucket."""
        models = [
            d.model for d in load_registry().values()
            if d.provider == "gemini" and LLMStage.DRAFT in d.stages
        ]
        self.assertEqual(len(models), len(set(models)), models)

    def test_no_verifier_deployment_pins_a_model_that_cannot_stop_thinking(self) -> None:
        """The verifier is a deterministic rewrite that must fit its output
        ceiling. gemini-3.7-flash honours no setting that reaches zero thinking
        (budget=0 is ignored, MINIMAL is rejected), so it serves draft only --
        where thinking is wanted and no ceiling is at risk."""
        for deployment in load_registry().values():
            if LLMStage.VERIFIER in deployment.stages:
                self.assertNotEqual(deployment.model, "gemini-3.7-flash", deployment.alias)

    def test_only_gemini_serves_the_verifier_stage(self) -> None:
        """A non-Gemini draft is only safe because Gemini still verifies it.
        Letting a draft provider verify its own work would leave no independent
        check on it."""
        for deployment in load_registry().values():
            if deployment.provider != "gemini":
                self.assertNotIn(LLMStage.VERIFIER, deployment.stages, deployment.alias)

    def test_mistral_draft_outranks_groq_draft(self) -> None:
        """Ordering is load-bearing, not cosmetic. groq-draft is capped at
        Groq's 8k tokens/minute free tier, so it cannot carry a full
        smart_summary prompt; mistral-draft can. Behind Groq it would only ever
        be reached for prompts Groq had already served."""
        registry = load_registry()
        self.assertLess(registry["mistral-draft"].priority, registry["groq-draft"].priority)
        self.assertGreaterEqual(
            registry["mistral-draft"].max_input_tokens,
            registry["gemini-primary"].max_input_tokens,
        )

    def test_groq_draft_fits_the_free_tier_token_per_minute_ceiling(self) -> None:
        """Groq's free tier allows 8000 tokens/minute on every model it serves.
        A registry entry claiming more gets an HTTP 413 at call time instead of
        being rejected locally, which wastes the round trip."""
        groq_draft = load_registry()["groq-draft"]
        self.assertLessEqual(
            groq_draft.max_input_tokens + groq_draft.max_output_tokens, 8000
        )

    def test_third_party_drafts_rank_behind_every_gemini_draft(self) -> None:
        registry = load_registry()
        gemini_draft = [
            deployment.priority for deployment in registry.values()
            if LLMStage.DRAFT in deployment.stages and deployment.provider == "gemini"
        ]
        for alias in ("groq-draft", "mistral-draft"):
            self.assertEqual(registry[alias].stages, (LLMStage.DRAFT,), alias)
            self.assertGreater(registry[alias].priority, max(gemini_draft), alias)

    def test_unapproved_provider_is_rejected_at_load(self) -> None:
        """_adapter() resolves an unrecognized provider to Gemini, so without
        this check a typo would quietly send context to Gemini while the
        registry and the logs both named a different provider."""
        entry = {
            "alias": "rogue", "provider": "anthropic", "model": "x", "enabled": True,
            "stages": ["draft"], "max_input_tokens": 100, "max_output_tokens": 100,
            "timeout_seconds": 5, "privacy": "configured_external_provider", "priority": 99,
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.json"
            path.write_text(json.dumps({"version": "v1", "deployments": [entry]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_registry(path)

    def test_invalid_registry_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "registry.json"
            path.write_text(json.dumps({"version": "v1", "deployments": []}), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_registry(path)


class TestRouter(unittest.TestCase):
    def _router(self) -> tuple[LLMRouter, MagicMock]:
        gemini = MagicMock()
        gemini.generate.return_value = LLMResult("{}", "gemini", "flash")
        return LLMRouter(gemini, "gemini-prod", "gemini-backup"), gemini

    def test_disabled_groq_is_not_eligible(self) -> None:
        router, _ = self._router()
        with patch.dict("os.environ", {"GROQ_ENABLED": "false"}, clear=False):
            aliases = [deployment.alias for deployment in router.deployments_for(LLMStage.OPTIMIZER)]
        self.assertEqual(aliases, ["gemini-optimizer-fallback"])

    def test_groq_rate_limit_falls_back_once_to_gemini(self) -> None:
        router, gemini = self._router()
        router._groq_provider = MagicMock()
        router._groq_provider.configured = True
        router._groq_provider.generate.side_effect = LLMError(LLMErrorCategory.RATE_LIMITED, "429", provider="groq")
        with patch.dict("os.environ", {"GROQ_ENABLED": "true"}, clear=False):
            result = router.optimize(_request())
        self.assertEqual(result.provider, "gemini")
        self.assertEqual(router._groq_provider.generate.call_count, 1)
        self.assertEqual(gemini.generate.call_count, 1)

    def test_non_retryable_groq_error_does_not_consume_gemini(self) -> None:
        router, gemini = self._router()
        router._groq_provider = MagicMock()
        router._groq_provider.configured = True
        router._groq_provider.generate.side_effect = LLMError(LLMErrorCategory.INVALID_REQUEST, "400", provider="groq")
        with patch.dict("os.environ", {"GROQ_ENABLED": "true"}, clear=False):
            with self.assertRaises(LLMError):
                router.optimize(_request())
        gemini.generate.assert_not_called()

    def test_mistral_is_dark_without_its_opt_in_flag(self) -> None:
        router, _ = self._router()
        router._mistral_provider = MagicMock(configured=True)
        with patch.dict("os.environ", {"MISTRAL_ENABLED": "false"}, clear=False):
            aliases = [d.alias for d in router.deployments_for(LLMStage.DRAFT)]
        self.assertNotIn("mistral-draft", aliases)
        with patch.dict("os.environ", {"MISTRAL_ENABLED": "true"}, clear=False):
            aliases = [d.alias for d in router.deployments_for(LLMStage.DRAFT)]
        self.assertEqual(aliases[0], "gemini-primary")
        self.assertLess(aliases.index("gemini-draft-fallback"), aliases.index("mistral-draft"))

    def test_prompt_too_large_for_one_deployment_still_tries_the_next(self) -> None:
        """The capacity pre-check rejects the deployment, not the request. As
        INVALID_REQUEST it was not fallback-eligible, so a small deployment
        aborted the stage instead of deferring to a larger one behind it."""
        router, _ = self._router()
        oversized = LLMRequest("x" * 40_000, "system", "draft", 0.0, 512, 30.0, LLMStage.DRAFT)
        with self.assertRaises(LLMError) as caught:
            router.generate(oversized, router._deployments["groq-draft"])
        self.assertIn(caught.exception.category, FALLBACK_ELIGIBLE)

    def test_rate_limited_groq_cools_down_then_recovers(self) -> None:
        router, _ = self._router()
        deployment = router._deployments["groq-optimizer"]
        router._cooldown(deployment, LLMError(LLMErrorCategory.RATE_LIMITED, "429", retry_after_seconds=60))
        self.assertGreater(router._cooldowns[deployment.alias], 0.0)
        router._cooldowns[deployment.alias] = 0.0
        self.assertEqual(router._cooldowns[deployment.alias], 0.0)


if __name__ == "__main__":
    unittest.main()
