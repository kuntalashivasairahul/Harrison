"""Approved-deployment router for Harrison's LLM stages."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from backend.llm.contracts import LLMError, LLMErrorCategory, LLMRequest, LLMResult, LLMStage
from backend.llm.groq_provider import GroqProvider

# Use Uvicorn's configured handler so route diagnostics appear beside the
# request completion log in the local server console.
log = logging.getLogger("uvicorn.error")
_REGISTRY_PATH = Path(__file__).with_name("model_registry.json")


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
        stages = tuple(LLMStage(value) for value in item["stages"])
        if item["max_input_tokens"] <= 0 or item["max_output_tokens"] <= 0 or item["timeout_seconds"] <= 0:
            raise ValueError(f"Invalid capability limits for {item['alias']}.")
        deployments[item["alias"]] = Deployment(stages=stages, **{key: value for key, value in item.items() if key != "stages"})
    if not deployments:
        raise ValueError("LLM model registry has no deployments.")
    return deployments


class LLMRouter:
    """Routes only approved deployments and records temporary cooldowns."""

    def __init__(self, gemini_provider, prod_model: str, backup_model: str) -> None:
        self._gemini_provider = gemini_provider
        self._groq_provider = GroqProvider()
        self._prod_model = prod_model
        self._backup_model = backup_model
        self._deployments = load_registry()
        self._lock = threading.Lock()
        self._cooldowns: dict[str, float] = {}
        self._last_errors: dict[str, str] = {}
        self._default_cooldown = float(os.getenv("LLM_PROVIDER_COOLDOWN_SECONDS", "60"))

    def _enabled(self, deployment: Deployment) -> bool:
        if not deployment.enabled or deployment.alias in self._cooldowns and self._cooldowns[deployment.alias] > time.monotonic():
            return False
        if deployment.provider == "groq":
            return os.getenv("GROQ_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"} and self._groq_provider.configured
        return True

    def deployments_for(self, stage: LLMStage) -> list[Deployment]:
        return sorted(
            (deployment for deployment in self._deployments.values() if stage in deployment.stages and self._enabled(deployment)),
            key=lambda deployment: deployment.priority,
        )

    def _resolve_model(self, deployment: Deployment) -> str:
        if deployment.model == "dynamic-prod":
            return self._prod_model
        if deployment.model == "dynamic-backup":
            return self._backup_model
        if deployment.alias == "groq-optimizer":
            return os.getenv("GROQ_OPTIMIZER_MODEL", deployment.model).strip() or deployment.model
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
            raise LLMError(
                LLMErrorCategory.INVALID_REQUEST,
                f"Prompt exceeds configured input capacity for {deployment.alias}.",
                provider=deployment.provider,
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
            provider = self._gemini_provider if deployment.provider == "gemini" else self._groq_provider
            result = provider.generate(bounded, model)
            log.info("llm_route: stage=%s provider=%s deployment=%s model=%s latency=%.3fs finish_reason=%s", bounded.stage.value, result.provider, deployment.alias, result.model, result.latency_seconds, result.finish_reason)
            return result
        except LLMError as error:
            self._cooldown(deployment, error)
            with self._lock:
                self._last_errors[deployment.alias] = error.category.value
            log.warning("llm_route: stage=%s provider=%s deployment=%s error=%s fallback_eligible=%s", bounded.stage.value, deployment.provider, deployment.alias, error.category.value, error.category in {LLMErrorCategory.RATE_LIMITED, LLMErrorCategory.TIMEOUT, LLMErrorCategory.UNAVAILABLE})
            raise

    def generate_named(self, request: LLMRequest, alias: str, model_override: str | None = None) -> LLMResult:
        deployment = self._deployments.get(alias)
        if deployment is None or request.stage not in deployment.stages or not self._enabled(deployment):
            raise LLMError(LLMErrorCategory.UNAVAILABLE, f"Deployment {alias!r} is not eligible.")
        return self.generate(request, deployment, model_override=model_override)

    def optimize(self, request: LLMRequest) -> LLMResult:
        last_error: LLMError | None = None
        for deployment in self.deployments_for(LLMStage.OPTIMIZER):
            try:
                return self.generate(request, deployment)
            except LLMError as error:
                last_error = error
                if error.category not in {LLMErrorCategory.RATE_LIMITED, LLMErrorCategory.TIMEOUT, LLMErrorCategory.UNAVAILABLE}:
                    break
        raise last_error or LLMError(LLMErrorCategory.UNAVAILABLE, "No optimizer deployment is eligible.")

    def status(self) -> list[dict[str, object]]:
        now = time.monotonic()
        with self._lock:
            return [
                {
                    "alias": deployment.alias,
                    "provider": deployment.provider,
                    "enabled": self._enabled(deployment),
                    "stages": [stage.value for stage in deployment.stages],
                    "cooling_down": self._cooldowns.get(deployment.alias, 0.0) > now,
                    "last_error": self._last_errors.get(deployment.alias),
                }
                for deployment in sorted(self._deployments.values(), key=lambda item: item.priority)
            ]
