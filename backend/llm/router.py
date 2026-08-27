"""Approved-deployment router for Harrison's LLM stages."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from backend.config import LLM_PROVIDER_COOLDOWN_SECONDS
from backend.llm.contracts import LLMError, LLMErrorCategory, LLMRequest, LLMResult, LLMStage
from backend.llm.groq_provider import GroqProvider
from backend.llm.mistral_provider import MistralProvider
from backend.observability import metrics, remaining_budget

log = logging.getLogger(__name__)

#: Failures worth trying the next deployment for. A bad request or a bad key
#: will fail identically everywhere, so those stop the loop immediately.
FALLBACK_ELIGIBLE = frozenset({
    LLMErrorCategory.RATE_LIMITED,
    LLMErrorCategory.TIMEOUT,
    LLMErrorCategory.UNAVAILABLE,
    LLMErrorCategory.NOT_FOUND,
})
_REGISTRY_PATH = Path(__file__).with_name("model_registry.json")

#: Providers that stay dark unless the operator opts in *and* a key is present.
#: Gemini is deliberately absent: it is the baseline, not an opt-in.
_OPT_IN_PROVIDER_FLAGS = {"groq": "GROQ_ENABLED", "mistral": "MISTRAL_ENABLED"}

#: Providers with an adapter behind them. A registry entry naming anything else
#: is rejected at load rather than dispatched: _adapter() resolves unknown names
#: to Gemini, so a typo'd or unapproved provider would otherwise send Harrison's
#: context to Gemini while the logs and the registry both claimed otherwise.
_KNOWN_PROVIDERS = frozenset({"gemini", "groq", "mistral"})


@dataclass(frozen=True)
class Deployment:
    alias: str
    provider: str
    model: str
    enabled: bool
    stages: tuple[LLMStage, ...]
    max_input_tokens: int
    max_output_tokens: int
    timeout_seconds: float
    privacy: str
    priority: int


def load_registry(path: Path = _REGISTRY_PATH) -> dict[str, Deployment]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != "v1":
        raise ValueError("Unsupported LLM model registry version.")
    deployments: dict[str, Deployment] = {}
    for item in payload.get("deployments", []):
        required = {"alias", "provider", "model", "enabled", "stages", "max_input_tokens", "max_output_tokens", "timeout_seconds", "privacy", "priority"}
        if set(item) != required or item["alias"] in deployments:
            raise ValueError("Invalid or duplicate LLM deployment registry entry.")
        if item["provider"] not in _KNOWN_PROVIDERS:
            raise ValueError(f"Unapproved LLM provider {item['provider']!r} for {item['alias']}.")
        stages = tuple(LLMStage(value) for value in item["stages"])
        if item["max_input_tokens"] <= 0 or item["max_output_tokens"] <= 0 or item["timeout_seconds"] <= 0:
            raise ValueError(f"Invalid capability limits for {item['alias']}.")
        deployments[item["alias"]] = Deployment(stages=stages, **{key: value for key, value in item.items() if key != "stages"})
    if not deployments:
        raise ValueError("LLM model registry has no deployments.")
    return deployments


class LLMRouter:
    """Routes only approved deployments and records temporary cooldowns."""

    def __init__(self, gemini_provider, prod_model: str, backup_model: str,
                 draft_fallback_model: str | None = None) -> None:
        self._gemini_provider = gemini_provider
        self._groq_provider = GroqProvider()
        self._mistral_provider = MistralProvider()
        self._prod_model = prod_model
        self._backup_model = backup_model
        self._draft_fallback_model = draft_fallback_model or backup_model
        self._deployments = load_registry()
        self._lock = threading.Lock()
        self._cooldowns: dict[str, float] = {}
        self._last_errors: dict[str, str] = {}
        self._default_cooldown = LLM_PROVIDER_COOLDOWN_SECONDS

    def _cooling_down(self, alias: str) -> bool:
        """Read cooldown state under the lock.

        The previous inline check was an unlocked ``alias in self._cooldowns``
        followed by ``self._cooldowns[alias]`` — a check-then-get against a
        dict another thread writes to in ``_cooldown()``.
        """
        with self._lock:
            return self._cooldowns.get(alias, 0.0) > time.monotonic()

    def _adapter(self, provider: str):
        return {
            "groq": self._groq_provider,
            "mistral": self._mistral_provider,
        }.get(provider, self._gemini_provider)

    def _enabled(self, deployment: Deployment) -> bool:
        if not deployment.enabled or self._cooling_down(deployment.alias):
            return False
        flag = _OPT_IN_PROVIDER_FLAGS.get(deployment.provider)
        if flag is None:
            return True
        return (
            os.getenv(flag, "false").strip().lower() in {"1", "true", "yes", "on"}
            and self._adapter(deployment.provider).configured
        )

    def deployments_for(self, stage: LLMStage) -> list[Deployment]:
        return sorted(
            (deployment for deployment in self._deployments.values() if stage in deployment.stages and self._enabled(deployment)),
            key=lambda deployment: deployment.priority,
        )

    @staticmethod
    def _value(model: str | Callable[[], str]) -> str:
        """Deployments may be wired with a callable so that model discovery
        happens on first use rather than at import time."""
        return model() if callable(model) else model

    def _resolve_model(self, deployment: Deployment) -> str:
        if deployment.model == "dynamic-prod":
            return self._value(self._prod_model)
        if deployment.model == "dynamic-backup":
            return self._value(self._backup_model)
        if deployment.model == "dynamic-draft-fallback":
            return self._value(self._draft_fallback_model)
        if deployment.alias == "groq-optimizer":
            return os.getenv("GROQ_OPTIMIZER_MODEL", deployment.model).strip() or deployment.model
        if deployment.alias == "mistral-draft":
            return os.getenv("MISTRAL_DRAFT_MODEL", deployment.model).strip() or deployment.model
        return deployment.model

    #: A model that is overloaded or hanging is unwell for seconds, not minutes,
    #: and re-probing it is how failover discovers it recovered.  Deliberately
    #: much shorter than the quota cooldown: a 503'd deployment was serving again
    #: two requests later, and LLM_PROVIDER_COOLDOWN_SECONDS (60s) would have
    #: benched the primary across that whole window for a transient spike.
    OUTAGE_COOLDOWN_SECONDS = 15.0

    #: Health failures that say something about the *deployment* rather than the
    #: key holding it, so rotating keys cannot clear them and the next stage of
    #: the same request should not pay to rediscover them.
    _OUTAGE_CATEGORIES = frozenset({
        LLMErrorCategory.UNAVAILABLE,
        LLMErrorCategory.TIMEOUT,
    })

    def _cooldown(self, deployment: Deployment, error: LLMError) -> None:
        # Two different failures with two different owners.
        #
        # Quota is per project/key and KeyManager already cools the key, so
        # benching a whole Gemini deployment would hide projects that are fine.
        #
        # An outage is not.  "This model is currently experiencing high demand"
        # is a property of the model, and no amount of key rotation clears it --
        # but this returned early for every Gemini error *and* for UNAVAILABLE
        # and TIMEOUT on every other provider, so nothing was ever cooled for
        # one.  A live smart_summary paid for that twice inside one request:
        # gemini-primary 503'd on the draft stage, then the verifier stage tried
        # the same deployment and took the same 503 -- ~26s of a 79s request
        # spent rediscovering an outage the previous stage had already found.
        if error.category in self._OUTAGE_CATEGORIES:
            wait = self.OUTAGE_COOLDOWN_SECONDS
        elif error.category == LLMErrorCategory.RATE_LIMITED and deployment.provider != "gemini":
            wait = error.retry_after_seconds or self._default_cooldown
        else:
            return
        with self._lock:
            self._cooldowns[deployment.alias] = time.monotonic() + max(wait, 1.0)
            self._last_errors[deployment.alias] = error.category.value

    def generate(self, request: LLMRequest, deployment: Deployment, model_override: str | None = None) -> LLMResult:
        model = model_override or self._resolve_model(deployment)
        if len(request.prompt) > deployment.max_input_tokens * 4:
            # UNAVAILABLE, not INVALID_REQUEST: the prompt is fine, this one
            # deployment is simply too small for it.  As INVALID_REQUEST it was
            # not fallback-eligible, so a small deployment early in the priority
            # order aborted the whole stage instead of deferring to a larger one
            # behind it — which is exactly the shape of the draft stage now that
            # groq-draft is capped at Groq's 8k-token-per-minute free tier.
            raise LLMError(
                LLMErrorCategory.UNAVAILABLE,
                f"Prompt exceeds configured input capacity for {deployment.alias}.",
                provider=deployment.provider,
            )
        # The registry is a hard capability ceiling, so clamping is correct —
        # but clamping *silently* meant that raising SMART_SUMMARY_MAX_TOKENS
        # past the registry cap appeared to work and changed nothing.  Say so.
        if request.max_output_tokens > deployment.max_output_tokens:
            log.warning(
                "llm_route: requested max_output_tokens=%d exceeds the registry cap for %s (%d) — clamping. "
                "Raise max_output_tokens in backend/llm/model_registry.json to lift this ceiling.",
                request.max_output_tokens,
                deployment.alias,
                deployment.max_output_tokens,
            )

        bounded = LLMRequest(
            prompt=request.prompt,
            system_instruction=request.system_instruction,
            model_alias=deployment.alias,
            temperature=request.temperature,
            max_output_tokens=min(request.max_output_tokens, deployment.max_output_tokens),
            deadline_seconds=min(request.deadline_seconds, deployment.timeout_seconds),
            stage=request.stage,
        )
        started = time.monotonic()
        try:
            result = self._adapter(deployment.provider).generate(bounded, model)
            log.info("llm_route: stage=%s provider=%s deployment=%s model=%s latency=%.3fs finish_reason=%s", bounded.stage.value, result.provider, deployment.alias, result.model, result.latency_seconds, result.finish_reason)
            # Which deployment actually served, aggregated across every request.
            # The per-request audit record cannot answer this: it is written only
            # for verified, untruncated answers, so the degraded requests are
            # precisely the ones it never captures. "It was fine, then it got
            # worse" is a question about the deployment mix over time.
            metrics.increment(f"llm_served_{deployment.alias}")
            if result.truncated:
                metrics.increment(f"llm_truncated_{deployment.alias}")
            return result
        except LLMError as error:
            self._cooldown(deployment, error)
            with self._lock:
                self._last_errors[deployment.alias] = error.category.value
            # The provider message is the only place the quota *metric* appears
            # ("...PerDay..." vs "...PerMinute..."), and without it a 429 that
            # clears in a minute is indistinguishable in the logs from one that
            # clears at midnight Pacific. Log the model too, so a failing
            # (key, model) pair can be read straight off the line above it.
            # latency on the failure line too: a 30s timeout and an instant 429
            # are the same line without it, and the difference is the whole
            # question when a request takes 79s. The success line has always
            # carried it, so the hops that cost the most were the only ones you
            # could not measure.
            log.warning(
                "llm_route: stage=%s provider=%s deployment=%s model=%s latency=%.3fs "
                "error=%s fallback_eligible=%s detail=%s",
                bounded.stage.value, deployment.provider, deployment.alias, model,
                time.monotonic() - started,
                error.category.value, error.category in FALLBACK_ELIGIBLE,
                str(error)[:400].replace("\n", " "),
            )
            raise

    def generate_named(self, request: LLMRequest, alias: str, model_override: str | None = None) -> LLMResult:
        deployment = self._deployments.get(alias)
        if deployment is None or request.stage not in deployment.stages or not self._enabled(deployment):
            raise LLMError(LLMErrorCategory.UNAVAILABLE, f"Deployment {alias!r} is not eligible.")
        return self.generate(request, deployment, model_override=model_override)

    #: Below this a call cannot complete anyway and the round trip is wasted.
    _MIN_HOP_SECONDS = 1.0

    def _within_budget(self, request: LLMRequest) -> LLMRequest | None:
        """Shrink a request's deadline to what is left of the request budget.

        None means there is nothing left to spend and the caller should stop.
        """
        remaining = remaining_budget()
        if remaining is None:
            return request
        if remaining < self._MIN_HOP_SECONDS:
            return None
        return replace(request, deadline_seconds=min(request.deadline_seconds, remaining))

    def generate_for_stage(
        self,
        request: LLMRequest,
        stage: LLMStage,
        exclude_model: str | None = None,
    ) -> LLMResult:
        """Try every eligible deployment for a stage, in priority order.

        The draft and verifier stages previously called generate_named() against
        a single deployment, so a provider-side outage on that one model failed
        the whole request even though a second Gemini deployment was configured
        and healthy. This is the failover the optimizer stage has always had.

        ``exclude_model`` drops one model from the order. It exists for the
        verifier stage, which must not be served by the model that wrote the
        draft: a model asked to grade its own work is the least likely to catch
        its own ungrounded claim, and that is the entire job of verify_answer().
        Without it, mistral-draft sitting on both stages meant a Gemini outage
        produced a mistral-large draft verified by mistral-large -- observed live
        on 2026-08-26, and no rule in the registry could express the constraint
        because it depends on which deployment actually served the draft.

        The exclusion is per model, not per provider. Excluding the provider
        would empty the verifier order outright whenever Mistral is dark, since
        every remaining verifier deployment is Gemini -- trading a self-verified
        answer for an unverified one, which is not an improvement.
        """
        last_error: LLMError | None = None
        longest_truncated: LLMResult | None = None
        reason = "unknown"
        deployments = self.deployments_for(stage)
        if exclude_model is not None:
            # _resolve_model(), not deployment.model: the Gemini entries carry
            # the sentinels "dynamic-prod"/"dynamic-backup" and only resolve to
            # a real model name at call time, so comparing the raw field would
            # never match the drafter and the exclusion would silently no-op.
            independent = [
                d for d in deployments if self._resolve_model(d) != exclude_model
            ]
            if not independent:
                # Refusing is the safe end of this branch: the caller treats a
                # failed verifier stage as "not verified", which caps confidence
                # and takes the draft_fallback path. Verifying with the drafter
                # would instead return a self-approved answer labelled verified.
                raise LLMError(
                    LLMErrorCategory.UNAVAILABLE,
                    f"No {stage.value} deployment is independent of model "
                    f"{exclude_model!r}.",
                )
            if len(independent) != len(deployments):
                log.info(
                    "llm_route: stage=%s excluding model=%s (wrote the draft) — "
                    "%d independent deployment(s) remain.",
                    stage.value, exclude_model, len(independent),
                )
            deployments = independent
        for index, deployment in enumerate(deployments):
            try:
                if index:
                    log.warning(
                        "llm_route: stage=%s falling back to deployment=%s after %s",
                        stage.value, deployment.alias, reason,
                    )
                # The stage deadline is computed once, before this loop, so every
                # hop reused the *original* allowance: five draft deployments at
                # 60s each is 300s against a 90s request budget. Re-clamp per hop
                # and stop when the budget is gone, so failover cannot outlive it.
                hop = self._within_budget(request)
                if hop is None:
                    log.warning(
                        "llm_route: stage=%s request budget exhausted after %s — "
                        "%d deployment(s) left untried.",
                        stage.value, reason, len(deployments) - index,
                    )
                    break
                result = self.generate(hop, deployment)
                if not result.truncated:
                    return result
                # A response cut at the token ceiling is a *successful* call, so
                # it never reached the except branch below: the stage handed back
                # the stump with every remaining deployment untried, and the user
                # got half an answer plus a truncation notice. Escalate instead —
                # the deployments further down have their own output ceilings and
                # Mistral's is the largest. Keep the longest partial in case they
                # all truncate. Bounded by _within_budget() above.
                reason = "truncated_output"
                if longest_truncated is None or len(result.text) > len(longest_truncated.text):
                    longest_truncated = result
            except LLMError as error:
                last_error = error
                reason = error.category.value
                if error.category not in FALLBACK_ELIGIBLE:
                    break
        if longest_truncated is not None:
            log.warning(
                "llm_route: stage=%s every deployment truncated — returning the longest "
                "(model=%s, %d chars).",
                stage.value, longest_truncated.model, len(longest_truncated.text),
            )
            return longest_truncated
        raise last_error or LLMError(
            LLMErrorCategory.UNAVAILABLE, f"No {stage.value} deployment is eligible."
        )

    def optimize(self, request: LLMRequest) -> LLMResult:
        return self.generate_for_stage(request, LLMStage.OPTIMIZER)

    def status(self) -> list[dict[str, object]]:
        now = time.monotonic()
        # Snapshot under the lock, then build the report outside it: _enabled()
        # takes the same non-reentrant lock, so calling it from inside would
        # deadlock.
        with self._lock:
            cooldowns = dict(self._cooldowns)
            last_errors = dict(self._last_errors)
        return [
            {
                "alias": deployment.alias,
                "provider": deployment.provider,
                "enabled": self._enabled(deployment),
                "stages": [stage.value for stage in deployment.stages],
                "cooling_down": cooldowns.get(deployment.alias, 0.0) > now,
                "last_error": last_errors.get(deployment.alias),
            }
            for deployment in sorted(self._deployments.values(), key=lambda item: item.priority)
        ]
