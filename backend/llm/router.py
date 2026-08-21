"""Approved-deployment router for Harrison's LLM stages."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from backend.config import LLM_PROVIDER_COOLDOWN_SECONDS
from backend.llm.contracts import LLMError, LLMErrorCategory, LLMRequest, LLMResult, LLMStage
from backend.llm.groq_provider import GroqProvider
from backend.llm.mistral_provider import MistralProvider

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

    def _cooldown(self, deployment: Deployment, error: LLMError) -> None:
        # Gemini quota is managed per project/key by KeyManager.  Cooling the
        # whole Gemini deployment would incorrectly hide healthy projects.
        if error.category != LLMErrorCategory.RATE_LIMITED or deployment.provider == "gemini":
            return
        wait = error.retry_after_seconds or self._default_cooldown
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
        try:
            result = self._adapter(deployment.provider).generate(bounded, model)
            log.info("llm_route: stage=%s provider=%s deployment=%s model=%s latency=%.3fs finish_reason=%s", bounded.stage.value, result.provider, deployment.alias, result.model, result.latency_seconds, result.finish_reason)
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
            log.warning(
                "llm_route: stage=%s provider=%s deployment=%s model=%s error=%s "
                "fallback_eligible=%s detail=%s",
                bounded.stage.value, deployment.provider, deployment.alias, model,
                error.category.value, error.category in FALLBACK_ELIGIBLE,
                str(error)[:400].replace("\n", " "),
            )
            raise

    def generate_named(self, request: LLMRequest, alias: str, model_override: str | None = None) -> LLMResult:
        deployment = self._deployments.get(alias)
        if deployment is None or request.stage not in deployment.stages or not self._enabled(deployment):
            raise LLMError(LLMErrorCategory.UNAVAILABLE, f"Deployment {alias!r} is not eligible.")
        return self.generate(request, deployment, model_override=model_override)

    def generate_for_stage(self, request: LLMRequest, stage: LLMStage) -> LLMResult:
        """Try every eligible deployment for a stage, in priority order.

        The draft and verifier stages previously called generate_named() against
        a single deployment, so a provider-side outage on that one model failed
        the whole request even though a second Gemini deployment was configured
        and healthy. This is the failover the optimizer stage has always had.
        """
        last_error: LLMError | None = None
        deployments = self.deployments_for(stage)
        for index, deployment in enumerate(deployments):
            try:
                if index:
                    log.warning(
                        "llm_route: stage=%s falling back to deployment=%s after %s",
                        stage.value, deployment.alias,
                        last_error.category.value if last_error else "unknown",
                    )
                return self.generate(request, deployment)
            except LLMError as error:
                last_error = error
                if error.category not in FALLBACK_ELIGIBLE:
                    break
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
