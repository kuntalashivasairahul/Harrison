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

import os
import unittest
from unittest.mock import MagicMock, patch

from backend.llm.contracts import LLMError, LLMErrorCategory, LLMRequest, LLMResult, LLMStage
from backend.llm.router import LLMRouter, load_registry


def _request(stage: LLMStage) -> LLMRequest:
    return LLMRequest("prompt", "system", "gemini-primary", 0.2, 3000, 60.0, stage)


def _result(model: str) -> LLMResult:
    return LLMResult(text="answer", provider="gemini", model=model, finish_reason="STOP")


#: The Gemini draft chain is one deployment per model, because free-tier quota
#: is metered per model. Derive it from the registry rather than hardcoding a
#: length: adding a model must not silently invalidate these tests.
#: Enabled only -- deployments_for() skips disabled entries, so including one
#: here asserts an attempt the router will never make.
_GEMINI_DRAFTS = [
    d.alias
    for d in sorted(load_registry().values(), key=lambda d: d.priority)
    if LLMStage.DRAFT in d.stages and d.provider == "gemini" and d.enabled
]


def _gemini_429s() -> list[LLMError]:
    return [LLMError(LLMErrorCategory.RATE_LIMITED, "429 quota") for _ in _GEMINI_DRAFTS]


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


class TestMistralDraftFailover(unittest.TestCase):
    """Mistral sits between the Gemini drafts and Groq. It is the only draft
    failover with the token budget to carry a full smart_summary: Groq's free
    tier caps every model at 8000 tokens/minute."""

    def _router(self, outcomes):
        router = _Router(outcomes)
        router._mistral_provider = MagicMock()
        router._mistral_provider.configured = True
        router._groq_provider = MagicMock()
        router._groq_provider.configured = True
        return router

    def test_every_gemini_draft_rate_limited_falls_through_to_mistral(self):
        # Mistral is now promoted to mid-tier (priority 15, between primary and fallback).
        # It attempts after primary fails but before fallback and other Gemini models.
        router = self._router([
            LLMError(LLMErrorCategory.RATE_LIMITED, "429 quota"),  # gemini-primary fails
            LLMResult(text="answer", provider="mistral", model="mistral-large-latest"),
        ])
        with patch.dict(os.environ, {"MISTRAL_ENABLED": "true", "GROQ_ENABLED": "true"}, clear=False):
            result = router.generate_for_stage(_request(LLMStage.DRAFT), LLMStage.DRAFT)

        self.assertEqual(result.provider, "mistral")
        self.assertEqual(router.attempted, ["gemini-primary", "mistral-draft"])

    def test_groq_still_catches_a_mistral_outage(self):
        """Mistral is now mid-tier (priority 15, tried early), so Groq is only
        reachable if the chain continues past Mistral and all Gemini fallbacks."""
        router = self._router([
            LLMError(LLMErrorCategory.RATE_LIMITED, "429"),  # gemini-primary
            LLMError(LLMErrorCategory.UNAVAILABLE, "503"),   # mistral-draft
            LLMError(LLMErrorCategory.RATE_LIMITED, "429"),  # gemini-draft-fallback
            LLMError(LLMErrorCategory.RATE_LIMITED, "429"),  # gemini-flash-3.6
            LLMError(LLMErrorCategory.RATE_LIMITED, "429"),  # gemini-flash-3
            LLMResult(text="answer", provider="groq", model="openai/gpt-oss-120b"),
        ])
        with patch.dict(os.environ, {"MISTRAL_ENABLED": "true", "GROQ_ENABLED": "true"}, clear=False):
            result = router.generate_for_stage(_request(LLMStage.DRAFT), LLMStage.DRAFT)

        self.assertEqual(result.provider, "groq")
        # Mistral gets tried early (priority 15), so Groq comes later after all Gemini options exhaust.
        self.assertIn("mistral-draft", router.attempted)
        self.assertEqual(router.attempted[-1], "groq-draft")

    def test_mistral_is_skipped_when_disabled(self):
        router = self._router([
            *_gemini_429s(),
            LLMResult(text="answer", provider="groq", model="openai/gpt-oss-120b"),
        ])
        with patch.dict(os.environ, {"MISTRAL_ENABLED": "false", "GROQ_ENABLED": "true"}, clear=False):
            router.generate_for_stage(_request(LLMStage.DRAFT), LLMStage.DRAFT)

        self.assertNotIn("mistral-draft", router.attempted)

    def test_mistral_now_serves_the_verifier_stage(self):
        # Mistral is now promoted to serve both DRAFT and VERIFIER as a hedge
        # against Gemini quota exhaustion (gemini rate limits are unstable).
        router = self._router([_result("prod-model")])
        with patch.dict(os.environ, {"MISTRAL_ENABLED": "true"}, clear=False):
            aliases = [d.alias for d in router.deployments_for(LLMStage.VERIFIER)]

        self.assertIn("mistral-draft", aliases)


class TestGroqDraftFailover(unittest.TestCase):
    """Groq is the third line of defence for the draft stage: it exists so a
    Gemini quota exhaustion returns an answer instead of a provider_failure."""

    def _router(self, outcomes):
        router = _Router(outcomes)
        router._groq_provider = MagicMock()
        router._groq_provider.configured = True
        return router

    def test_every_gemini_draft_rate_limited_falls_through_to_groq(self):
        router = self._router([
            *_gemini_429s(),
            LLMResult(text="answer", provider="groq", model="openai/gpt-oss-120b"),
        ])
        with patch.dict(os.environ, {"GROQ_ENABLED": "true"}, clear=False):
            result = router.generate_for_stage(_request(LLMStage.DRAFT), LLMStage.DRAFT)

        self.assertEqual(result.provider, "groq")
        self.assertEqual(router.attempted, [*_GEMINI_DRAFTS, "groq-draft"])

    def test_groq_is_skipped_when_disabled(self):
        router = self._router(_gemini_429s())
        with patch.dict(os.environ, {"GROQ_ENABLED": "false"}, clear=False):
            with self.assertRaises(LLMError):
                router.generate_for_stage(_request(LLMStage.DRAFT), LLMStage.DRAFT)

        self.assertEqual(router.attempted, _GEMINI_DRAFTS)

    def test_groq_is_never_offered_the_verifier_stage(self):
        router = self._router([_result("prod-model")])
        with patch.dict(os.environ, {"GROQ_ENABLED": "true"}, clear=False):
            aliases = [d.alias for d in router.deployments_for(LLMStage.VERIFIER)]

        self.assertNotIn("groq-draft", aliases)

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
            LLMError(LLMErrorCategory.UNAVAILABLE, f"down {i}")
            for i in range(len(_GEMINI_DRAFTS))
        ])
        with self.assertRaises(LLMError) as ctx:
            router.generate_for_stage(_request(LLMStage.DRAFT), LLMStage.DRAFT)
        self.assertEqual(ctx.exception.category, LLMErrorCategory.UNAVAILABLE)
        self.assertEqual(router.attempted, _GEMINI_DRAFTS)

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


class TestTruncationEscalation(unittest.TestCase):
    """A truncated answer is a successful call, so it used to end the stage.

    The live symptom: a draft came back cut mid-sentence with six healthy
    deployments untried, and ask_llm() stapled a truncation notice to the stump
    and returned it.
    """

    @staticmethod
    def _truncated(model: str, text: str = "half an ans") -> LLMResult:
        return LLMResult(text=text, provider="gemini", model=model, finish_reason="MAX_TOKENS")

    def test_a_truncated_draft_escalates_to_the_next_deployment(self):
        router = _Router([self._truncated("prod-model"), _result("draft-fallback-model")])
        result = router.generate_for_stage(_request(LLMStage.DRAFT), LLMStage.DRAFT)
        self.assertEqual(result.model, "draft-fallback-model")
        self.assertFalse(result.truncated)
        self.assertEqual(router.attempted, ["gemini-primary", "gemini-draft-fallback"])

    def test_openai_style_length_escalates_too(self):
        """Mistral and Groq say "length"; only Gemini says MAX_TOKENS."""
        router = _Router([
            LLMResult(text="half", provider="gemini", model="prod-model", finish_reason="LENGTH"),
            _result("draft-fallback-model"),
        ])
        result = router.generate_for_stage(_request(LLMStage.DRAFT), LLMStage.DRAFT)
        self.assertEqual(result.model, "draft-fallback-model")

    def test_all_truncated_returns_the_longest_partial(self):
        """Half an answer still beats a hard failure — but return the best half."""
        # One length per enabled deployment, longest deliberately in the middle
        # so the winner is chosen by length rather than by position. Sized off
        # the chain: a fixed tuple broke the moment a deployment was disabled.
        lengths = [10 * (i + 1) for i in range(len(_GEMINI_DRAFTS))]
        lengths[len(lengths) // 2] = 400
        router = _Router([
            self._truncated(alias, text="x" * length)
            for alias, length in zip(_GEMINI_DRAFTS, lengths, strict=True)
        ])
        with patch.dict(os.environ, {"MISTRAL_ENABLED": "false", "GROQ_ENABLED": "false"}, clear=False):
            result = router.generate_for_stage(_request(LLMStage.DRAFT), LLMStage.DRAFT)

        self.assertEqual(len(result.text), max(lengths))
        self.assertTrue(result.truncated)
        self.assertEqual(router.attempted, _GEMINI_DRAFTS)

    def test_a_partial_answer_wins_over_a_later_hard_failure(self):
        """Truncated first, then a non-fallback-eligible error that breaks the
        loop. The partial is still worth more to the caller than the exception."""
        router = _Router([
            self._truncated("prod-model"),
            LLMError(LLMErrorCategory.INVALID_REQUEST, "400 bad request"),
        ])
        result = router.generate_for_stage(_request(LLMStage.DRAFT), LLMStage.DRAFT)
        self.assertEqual(result.model, "prod-model")
        self.assertTrue(result.truncated)


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
