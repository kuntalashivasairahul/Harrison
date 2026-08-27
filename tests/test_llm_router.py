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
        """CODING_RULES §6.1, amended: the verifier stage must survive a
        provider-wide outage.  A Gemini-only verifier meant a Gemini outage left
        only "return the draft unverified" or "fail the request", and a
        gemini-2.5-flash 503 on both stages of one request was observed live."""
        verifier_providers = {d.provider for d in load_registry().values() if LLMStage.VERIFIER in d.stages}
        self.assertGreater(len(verifier_providers), 1, "Verifier must have multiple providers")

    def test_groq_never_serves_the_verifier_stage(self) -> None:
        """The one provider restriction the amendment did NOT relax.  Groq's
        free tier is 8k tokens/minute, so it cannot hold a full smart_summary
        draft plus its context and would verify against a truncated view of the
        evidence.  Capacity, not availability, is the objection."""
        for deployment in load_registry().values():
            if LLMStage.VERIFIER in deployment.stages:
                self.assertNotEqual(deployment.provider, "groq", deployment.alias)

    def test_every_draft_model_has_an_independent_verifier_available(self) -> None:
        """Self-verification is forbidden, so excluding the drafter must never
        empty the verifier stage -- otherwise a model that drafts can only be
        checked by itself, and the request silently degrades to unverified."""
        registry = load_registry()
        verifiers = [d for d in registry.values() if LLMStage.VERIFIER in d.stages]
        for drafter in {d.model for d in registry.values() if LLMStage.DRAFT in d.stages}:
            independent = [d for d in verifiers if d.model != drafter]
            self.assertTrue(independent, f"{drafter} draft has no independent verifier")

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
        against Gemini quota exhaustion, on both the draft and verifier stages
        -- though not with the same model on each."""
        registry = load_registry()
        gemini_draft_max = max(
            d.priority for d in registry.values()
            if LLMStage.DRAFT in d.stages and d.provider == "gemini"
        )
        self.assertGreater(registry["groq-draft"].priority, gemini_draft_max)
        self.assertLess(registry["mistral-draft"].priority, gemini_draft_max)
        self.assertIn(LLMStage.VERIFIER, registry["mistral-verifier"].stages)

    def test_the_slow_mistral_model_drafts_but_never_verifies(self) -> None:
        """mistral-large-latest was measured at 23-30s live and hit its own 30s
        ceiling once, and that ceiling cannot be raised -- the budget invariant
        below caps a draft at a third of the request budget. It stays on the
        draft stage, where a timeout is cheap because failover continues past
        it, and off the verifier stage, where mistral-medium-latest does the
        same job inside 20s."""
        registry = load_registry()
        self.assertNotIn(LLMStage.VERIFIER, registry["mistral-draft"].stages)
        self.assertLess(
            registry["mistral-verifier"].timeout_seconds,
            registry["mistral-draft"].timeout_seconds,
        )

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
        # Mistral sits behind gemini-draft-fallback, not in front of it.  In
        # front, a gemini-primary 503 bought a 23-30s mistral-large call before
        # anything tried gemini-3.5-flash, which serves the same draft in ~10s;
        # one live smart_summary spent 53s of a 62s request that way, 30s of it
        # on a mistral-large call that then hit its own 30s timeout.  It stays
        # ahead of the remaining Gemini drafts, so it is still a real hedge.
        self.assertLess(aliases.index("gemini-draft-fallback"), aliases.index("mistral-draft"))
        self.assertLess(aliases.index("mistral-draft"), aliases.index("gemini-flash-3.6"))

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

    def test_gemini_quota_still_does_not_bench_the_deployment(self) -> None:
        """KeyManager cools the exhausted *key* and rotates to the next project.
        Benching the deployment on top of that would hide the healthy ones."""
        router, _ = self._router()
        deployment = router._deployments["gemini-primary"]
        router._cooldown(deployment, LLMError(LLMErrorCategory.RATE_LIMITED, "429", retry_after_seconds=60))
        self.assertNotIn(deployment.alias, router._cooldowns)

    def test_outage_cools_the_deployment_so_the_next_stage_skips_it(self) -> None:
        """A 503 is a property of the model, not of the key, so key rotation
        cannot clear it.  It used to cool nothing: gemini-primary 503'd on the
        draft stage and the verifier stage then paid for the same 503 again,
        ~26s of a single 79s request.  Both outage categories, both providers --
        the old guard let UNAVAILABLE and TIMEOUT through for every one."""
        for category in (LLMErrorCategory.UNAVAILABLE, LLMErrorCategory.TIMEOUT):
            for alias in ("gemini-primary", "mistral-draft"):
                with self.subTest(category=category, alias=alias):
                    router, _ = self._router()
                    router._cooldown(router._deployments[alias], LLMError(category, "503"))
                    self.assertTrue(router._cooling_down(alias))
                    self.assertEqual(router._last_errors[alias], category.value)

    def test_outage_cooldown_is_short_enough_to_rediscover_recovery(self) -> None:
        """The 503'd deployment was serving again two requests later.  Reusing
        the 60s quota cooldown would bench it for the rest of the outage *and*
        for most of the recovery."""
        self.assertLessEqual(LLMRouter.OUTAGE_COOLDOWN_SECONDS, 20.0)

    def test_a_cooling_deployment_is_dropped_from_the_stage_order(self) -> None:
        """The cooldown only buys anything if deployments_for() honours it."""
        router, _ = self._router()
        router._mistral_provider = MagicMock(configured=True)
        with patch.dict("os.environ", {"MISTRAL_ENABLED": "true"}, clear=False):
            before = [d.alias for d in router.deployments_for(LLMStage.DRAFT)]
            router._cooldown(router._deployments["gemini-primary"], LLMError(LLMErrorCategory.UNAVAILABLE, "503"))
            after = [d.alias for d in router.deployments_for(LLMStage.DRAFT)]
        self.assertIn("gemini-primary", before)
        self.assertNotIn("gemini-primary", after)
        self.assertEqual(after, [alias for alias in before if alias != "gemini-primary"])


class TestFailureLogging(unittest.TestCase):
    """A failed hop is the expensive one, so it is the one worth measuring."""

    def test_a_failed_hop_logs_how_long_it_cost(self) -> None:
        """The success line has always carried latency=; the failure line did
        not, so a 30s timeout and an instant 429 read identically -- and the
        difference between them is the entire question when a request takes
        79s."""
        gemini = MagicMock()
        gemini.generate.side_effect = LLMError(LLMErrorCategory.UNAVAILABLE, "503 high demand")
        router = LLMRouter(gemini, "gemini-prod", "gemini-backup")

        with self.assertLogs("backend.llm.router", level="WARNING") as captured:
            with self.assertRaises(LLMError):
                router.generate(
                    LLMRequest("prompt", "system", "draft", 0.0, 512, 30.0, LLMStage.DRAFT),
                    router._deployments["gemini-primary"],
                )

        line = "\n".join(captured.output)
        self.assertIn("latency=", line)
        self.assertIn("error=unavailable", line)


if __name__ == "__main__":
    unittest.main()
