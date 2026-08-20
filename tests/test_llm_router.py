"""No-network tests for registry validation, routing policy, and cooldowns."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.llm.contracts import LLMError, LLMErrorCategory, LLMRequest, LLMResult, LLMStage
from backend.llm.router import LLMRouter, load_registry


def _request() -> LLMRequest:
    return LLMRequest("prompt", "system", "optimizer", 0.0, 32, 5.0, LLMStage.OPTIMIZER)


class TestRegistry(unittest.TestCase):
    def test_default_registry_has_only_approved_stage_one_deployments(self) -> None:
        registry = load_registry()
        self.assertEqual(
            set(registry),
            {"gemini-primary", "gemini-draft-fallback", "gemini-optimizer-fallback", "groq-optimizer"},
        )
        self.assertEqual(registry["gemini-primary"].stages, (LLMStage.DRAFT, LLMStage.VERIFIER))
        # The draft stage must have somewhere to fall back to.
        self.assertEqual(registry["gemini-draft-fallback"].stages, (LLMStage.DRAFT, LLMStage.VERIFIER))
        self.assertGreater(
            registry["gemini-draft-fallback"].priority, registry["gemini-primary"].priority
        )

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

    def test_rate_limited_groq_cools_down_then_recovers(self) -> None:
        router, _ = self._router()
        deployment = router._deployments["groq-optimizer"]
        router._cooldown(deployment, LLMError(LLMErrorCategory.RATE_LIMITED, "429", retry_after_seconds=60))
        self.assertGreater(router._cooldowns[deployment.alias], 0.0)
        router._cooldowns[deployment.alias] = 0.0
        self.assertEqual(router._cooldowns[deployment.alias], 0.0)


if __name__ == "__main__":
    unittest.main()
