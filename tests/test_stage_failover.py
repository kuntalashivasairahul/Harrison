"""
tests/test_stage_failover.py
============================
Draft/verifier failover across deployments.

The outage this exists for: gemini-2.5-flash returned 503 UNAVAILABLE across
all nine keys while gemini-3.5-flash answered normally. The draft stage called
generate_named() against a single deployment, so every clinical query failed
even though a healthy sibling model was one config entry away. The optimizer
stage had had this failover all along.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from backend.llm.contracts import LLMError, LLMErrorCategory, LLMRequest, LLMResult, LLMStage
from backend.llm.router import LLMRouter, load_registry


def _request(stage: LLMStage) -> LLMRequest:
    return LLMRequest("prompt", "system", "gemini-primary", 0.2, 3000, 60.0, stage)


def _result(model: str) -> LLMResult:
    return LLMResult(text="answer", provider="gemini", model=model, finish_reason="STOP")


class _Router(LLMRouter):
    """Router with a scripted provider, so no network is involved."""

    def __init__(self, outcomes):
        super().__init__(MagicMock(), "prod-model", "backup-model", "draft-fallback-model")
        self._outcomes = list(outcomes)
        self.attempted: list[str] = []

    def generate(self, request, deployment, model_override=None):
        self.attempted.append(deployment.alias)
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class TestRegistryShape(unittest.TestCase):
    def test_draft_stage_has_more_than_one_deployment(self):
        registry = load_registry()
        draft = [d for d in registry.values() if LLMStage.DRAFT in d.stages]
        self.assertGreaterEqual(len(draft), 2)

    def test_the_fallback_is_a_different_model_from_primary(self):
        registry = load_registry()
        self.assertNotEqual(
            registry["gemini-primary"].model, registry["gemini-draft-fallback"].model
        )


class TestDraftFailover(unittest.TestCase):
    def test_a_503_on_the_primary_falls_through_to_the_fallback(self):
        """The exact live failure."""
        router = _Router([
            LLMError(LLMErrorCategory.UNAVAILABLE, "503 high demand"),
            _result("draft-fallback-model"),
        ])
        result = router.generate_for_stage(_request(LLMStage.DRAFT), LLMStage.DRAFT)

        self.assertEqual(result.model, "draft-fallback-model")
        self.assertEqual(router.attempted, ["gemini-primary", "gemini-draft-fallback"])

    def test_primary_success_never_touches_the_fallback(self):
        router = _Router([_result("prod-model")])
        result = router.generate_for_stage(_request(LLMStage.DRAFT), LLMStage.DRAFT)

        self.assertEqual(result.model, "prod-model")
        self.assertEqual(router.attempted, ["gemini-primary"])

    def test_deployments_are_tried_in_priority_order(self):
        router = _Router([
            LLMError(LLMErrorCategory.TIMEOUT, "timeout"),
            _result("draft-fallback-model"),
        ])
        router.generate_for_stage(_request(LLMStage.DRAFT), LLMStage.DRAFT)
        priorities = [router._deployments[a].priority for a in router.attempted]
        self.assertEqual(priorities, sorted(priorities))

    def test_a_bad_request_does_not_waste_the_fallback(self):
        """INVALID_REQUEST fails identically everywhere; stop immediately."""
        router = _Router([LLMError(LLMErrorCategory.INVALID_REQUEST, "prompt too long")])
        with self.assertRaises(LLMError):
            router.generate_for_stage(_request(LLMStage.DRAFT), LLMStage.DRAFT)
        self.assertEqual(router.attempted, ["gemini-primary"])

    def test_an_auth_error_does_not_waste_the_fallback(self):
        router = _Router([LLMError(LLMErrorCategory.AUTH, "bad key")])
        with self.assertRaises(LLMError):
            router.generate_for_stage(_request(LLMStage.DRAFT), LLMStage.DRAFT)
        self.assertEqual(router.attempted, ["gemini-primary"])

    def test_all_deployments_down_raises_the_last_error(self):
        router = _Router([
            LLMError(LLMErrorCategory.UNAVAILABLE, "first"),
            LLMError(LLMErrorCategory.UNAVAILABLE, "second"),
        ])
        with self.assertRaises(LLMError) as ctx:
            router.generate_for_stage(_request(LLMStage.DRAFT), LLMStage.DRAFT)
        self.assertEqual(ctx.exception.category, LLMErrorCategory.UNAVAILABLE)
        self.assertEqual(len(router.attempted), 2)

    def test_the_verifier_stage_fails_over_too(self):
        router = _Router([
            LLMError(LLMErrorCategory.UNAVAILABLE, "503"),
            _result("draft-fallback-model"),
        ])
        result = router.generate_for_stage(_request(LLMStage.VERIFIER), LLMStage.VERIFIER)
        self.assertEqual(result.model, "draft-fallback-model")

    def test_optimize_still_uses_the_same_machinery(self):
        router = _Router([_result("prod-model")])
        router.generate_for_stage = MagicMock(return_value=_result("prod-model"))
        router.optimize(_request(LLMStage.OPTIMIZER))
        router.generate_for_stage.assert_called_once()


class TestRetiredModelFailover(unittest.TestCase):
    """A retired model used to normalize to UNKNOWN, which is not
    fallback-eligible — so pointing a deployment at a decommissioned model
    failed the request outright. Three outages in this project traced back to
    a model being retired, so 404 has its own category."""

    def test_404_normalizes_to_not_found(self):
        from backend.llm.gemini_provider import normalize_provider_error

        for text in (
            "404 NOT_FOUND. {'error': {'code': 404}}",
            "models/gemini-1.5-flash is not found for API version v1beta",
            "This model does not exist or you do not have access to it",
        ):
            self.assertEqual(
                normalize_provider_error(RuntimeError(text), "gemini").category,
                LLMErrorCategory.NOT_FOUND,
                text,
            )

    def test_not_found_is_fallback_eligible(self):
        from backend.llm.router import FALLBACK_ELIGIBLE

        self.assertIn(LLMErrorCategory.NOT_FOUND, FALLBACK_ELIGIBLE)

    def test_not_found_is_not_retry_eligible(self):
        """Retrying the same dead model achieves nothing; only failover helps."""
        from backend.llm.llm import _RETRYABLE

        self.assertNotIn(LLMErrorCategory.NOT_FOUND, _RETRYABLE)

    def test_a_retired_primary_falls_through_to_the_fallback(self):
        router = _Router([
            LLMError(LLMErrorCategory.NOT_FOUND, "404 model retired"),
            _result("draft-fallback-model"),
        ])
        result = router.generate_for_stage(_request(LLMStage.DRAFT), LLMStage.DRAFT)
        self.assertEqual(result.model, "draft-fallback-model")
        self.assertEqual(router.attempted, ["gemini-primary", "gemini-draft-fallback"])


class TestModelSlots(unittest.TestCase):
    def test_draft_fallback_resolves_to_its_own_model(self):
        router = _Router([])
        deployment = router._deployments["gemini-draft-fallback"]
        self.assertEqual(router._resolve_model(deployment), "draft-fallback-model")

    def test_primary_still_resolves_to_prod(self):
        router = _Router([])
        self.assertEqual(router._resolve_model(router._deployments["gemini-primary"]), "prod-model")


if __name__ == "__main__":
    unittest.main()
