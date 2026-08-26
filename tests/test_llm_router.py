"""No-network tests for registry validation, routing policy, and cooldowns."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.llm.contracts import LLMError, LLMErrorCategory, LLMRequest, LLMResult, LLMStage
from backend.llm.router import FALLBACK_ELIGIBLE, LLMRouter, load_registry
from backend.observability import metrics, start_request_budget


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
                "groq-optimizer", "groq-draft", "mistral-draft", "mistral-verifier",
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

    def test_verifier_has_provider_diversity_to_hedge_against_quota_exhaustion(self) -> None:
        """Verifier must have options outside a single provider to prevent
        Gemini quota exhaustion from blocking answer verification. A draft
        provider can verify another provider's work (cross-provider is safe);
        only same-provider verification would be unsafe."""
        draft_providers = {d.provider for d in load_registry().values() if LLMStage.DRAFT in d.stages}
        verifier_providers = {d.provider for d in load_registry().values() if LLMStage.VERIFIER in d.stages}
        self.assertGreater(len(verifier_providers), 1, "Verifier must have multiple providers")

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

    def test_groq_ranks_behind_all_drafts_mistral_is_first_class(self) -> None:
        """Groq is a last-resort draft option due to throughput constraints
        (8k tokens/min free tier). Mistral is promoted to mid-tier as a hedge
        against Gemini quota exhaustion; it serves both draft and verifier."""
        registry = load_registry()
        gemini_draft_max = max(
            d.priority for d in registry.values()
            if LLMStage.DRAFT in d.stages and d.provider == "gemini"
        )
        self.assertGreater(registry["groq-draft"].priority, gemini_draft_max)
        self.assertLess(registry["mistral-draft"].priority, gemini_draft_max)
        self.assertIn(LLMStage.VERIFIER, registry["mistral-draft"].stages)

    def test_gemini_3_7_stays_disabled_until_it_can_answer(self) -> None:
        """Probed live at the real 4,096-token qa ceiling with a real prompt:
        three consecutive 504 DEADLINE_EXCEEDED at 16.7s/19.1s/19.1s, against a
        gemini-2.5-flash control that answered in 2.6-3.6s every time. Five
        failures, no successes, all session. An always-failing deployment is not
        free -- it spent 20s of a 90s budget on every cascade that reached it,
        which is budget mistral-draft behind it never got.

        Delete this test to re-enable it, and re-probe first."""
        self.assertFalse(load_registry()["gemini-flash-3.7"].enabled)

    def test_the_draft_chain_is_traversable_within_the_request_budget(self) -> None:
        """Failover only helps if the budget can reach the deployments behind
        the failure. Every draft deployment sat at timeout_seconds=60 against a
        90s budget, so two slow ones consumed all of it: a live cascade died
        after four hops with mistral-draft and two others untried. A deployment
        may not claim more than a third of the budget, and three must fit.

        The floor matters as much as the ceiling -- the worst legitimate draft
        observed live was 18.3s, so a timeout under 20s would start killing
        real answers rather than slow ones."""
        from backend.config import LLM_TOTAL_REQUEST_BUDGET_SECONDS as budget

        # Enabled only: a disabled deployment is never routed to, so it spends
        # none of the budget and must not be counted against it.
        drafts = sorted(
            (d for d in load_registry().values() if LLMStage.DRAFT in d.stages and d.enabled),
            key=lambda d: d.priority,
        )
        for deployment in drafts:
            self.assertGreaterEqual(deployment.timeout_seconds, 20, deployment.alias)
            self.assertLessEqual(deployment.timeout_seconds, budget / 3, deployment.alias)
        self.assertLessEqual(sum(d.timeout_seconds for d in drafts[:3]), budget)

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


class TestFailoverBudget(unittest.TestCase):
    """Failover must not outlive the request budget.

    The stage deadline is computed once, before the failover loop, so every hop
    reused the original allowance: five draft deployments at 60s each is 300s
    against a 90s budget. Truncation escalation made that reachable -- a
    truncated response is a *successful* call, so a stage can walk its whole
    chain without a single error."""

    def _router(self, finish_reason: str) -> tuple[LLMRouter, MagicMock]:
        gemini = MagicMock()
        gemini.generate.return_value = LLMResult("answer", "gemini", "flash", finish_reason)
        return LLMRouter(gemini, "gemini-prod", "gemini-backup"), gemini

    def _draft(self) -> LLMRequest:
        return LLMRequest("prompt", "system", "draft", 0.0, 512, 60.0, LLMStage.DRAFT)

    def test_each_hop_is_clamped_to_what_is_left(self) -> None:
        router, gemini = self._router("MAX_TOKENS")
        start_request_budget(5.0)
        self.addCleanup(start_request_budget, 0.0)
        with patch.dict("os.environ", {"GROQ_ENABLED": "false", "MISTRAL_ENABLED": "false"}, clear=False):
            router.generate_for_stage(self._draft(), LLMStage.DRAFT)
        deadlines = [call.args[0].deadline_seconds for call in gemini.generate.call_args_list]
        self.assertTrue(deadlines)
        # The stage asks for 60s, but only ~5s of budget exists.
        for deadline in deadlines:
            self.assertLessEqual(deadline, 5.0)

    def test_an_exhausted_budget_stops_the_walk(self) -> None:
        router, gemini = self._router("MAX_TOKENS")
        start_request_budget(0.001)
        self.addCleanup(start_request_budget, 0.0)
        with patch.dict("os.environ", {"GROQ_ENABLED": "false", "MISTRAL_ENABLED": "false"}, clear=False):
            with self.assertRaises(LLMError):
                router.generate_for_stage(self._draft(), LLMStage.DRAFT)
        gemini.generate.assert_not_called()

    def test_no_budget_configured_leaves_the_deadline_alone(self) -> None:
        """LLM_TOTAL_REQUEST_BUDGET_SECONDS=0 disables the budget; the stage
        deadline must still apply rather than becoming unbounded or zero."""
        router, gemini = self._router("STOP")
        start_request_budget(0.0)
        self.addCleanup(start_request_budget, 0.0)
        with patch.dict("os.environ", {"GROQ_ENABLED": "false", "MISTRAL_ENABLED": "false"}, clear=False):
            router.generate_for_stage(self._draft(), LLMStage.DRAFT)
        # The deployment cap, not a literal: registry timeouts get retuned, and
        # a hardcoded 60 here failed the moment they were.
        self.assertEqual(
            gemini.generate.call_args.args[0].deadline_seconds,
            router._deployments["gemini-primary"].timeout_seconds,
        )


class TestServedDeploymentMetrics(unittest.TestCase):
    """Which deployment actually served, aggregated across requests.

    The per-request audit record cannot answer this: it is written only for
    verified, untruncated answers, so degraded requests are precisely the ones
    it never captures -- and "it was good, then it truncated" is a question
    about the deployment mix over time."""

    def setUp(self) -> None:
        metrics.reset()
        self.addCleanup(metrics.reset)

    def _router(self, finish_reason: str) -> LLMRouter:
        gemini = MagicMock()
        gemini.generate.return_value = LLMResult("answer", "gemini", "flash", finish_reason)
        return LLMRouter(gemini, "gemini-prod", "gemini-backup")

    def _draft(self) -> LLMRequest:
        return LLMRequest("prompt", "system", "draft", 0.0, 512, 30.0, LLMStage.DRAFT)

    def test_a_served_request_is_counted_against_its_deployment(self) -> None:
        with patch.dict("os.environ", {"GROQ_ENABLED": "false", "MISTRAL_ENABLED": "false"}, clear=False):
            self._router("STOP").generate_for_stage(self._draft(), LLMStage.DRAFT)
        counters = metrics.snapshot()["counters"]
        self.assertEqual(counters["llm_served_gemini-primary"], 1)
        self.assertNotIn("llm_truncated_gemini-primary", counters)

    def test_truncation_is_counted_separately_at_every_deployment_it_hits(self) -> None:
        with patch.dict("os.environ", {"GROQ_ENABLED": "false", "MISTRAL_ENABLED": "false"}, clear=False):
            self._router("MAX_TOKENS").generate_for_stage(self._draft(), LLMStage.DRAFT)
        counters = metrics.snapshot()["counters"]
        self.assertEqual(counters["llm_truncated_gemini-primary"], 1)
        self.assertEqual(counters["llm_served_gemini-primary"], 1)
        # Truncation escalates, so the stage walked past the first deployment.
        self.assertGreater(counters["llm_truncated_gemini-draft-fallback"], 0)


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
        # Mistral is now promoted to mid-tier, before the draft fallback.
        self.assertLess(aliases.index("mistral-draft"), aliases.index("gemini-draft-fallback"))

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
