"""Gemini adapter used by the stage-aware router."""

from __future__ import annotations

import time
from typing import Callable

from google.genai import types

from backend.llm.contracts import LLMError, LLMErrorCategory, LLMRequest, LLMResult


def normalize_provider_error(exc: Exception, provider: str) -> LLMError:
    text = str(exc).lower()
    if any(marker in text for marker in ("429", "quota", "rate limit", "resource exhausted", "resourceexhausted")):
        return LLMError(LLMErrorCategory.RATE_LIMITED, str(exc), provider=provider)
    if any(marker in text for marker in ("timeout", "timed out", "deadline exceeded")):
        return LLMError(LLMErrorCategory.TIMEOUT, str(exc), provider=provider)
    if any(marker in text for marker in ("401", "403", "api key", "authentication", "permission denied")):
        return LLMError(LLMErrorCategory.AUTH, str(exc), provider=provider)
    if any(marker in text for marker in ("400", "invalid argument", "context length", "max tokens")):
        return LLMError(LLMErrorCategory.INVALID_REQUEST, str(exc), provider=provider)
    if any(marker in text for marker in ("500", "502", "503", "unavailable", "connection reset")):
        return LLMError(LLMErrorCategory.UNAVAILABLE, str(exc), provider=provider)
    return LLMError(LLMErrorCategory.UNKNOWN, str(exc), provider=provider)


class GeminiProvider:
    name = "gemini"

    def __init__(self, key_manager_getter: Callable[[], object], extract_response: Callable[[object], tuple[str, bool]]) -> None:
        self._key_manager_getter = key_manager_getter
        self._extract_response = extract_response

    def generate(self, request: LLMRequest, model: str) -> LLMResult:
        started = time.perf_counter()
        try:
            client = self._key_manager_getter().next_client()
            response = client.models.generate_content(
                model=model,
                contents=request.prompt,
                config=self._generate_config(
                    system_instruction=request.system_instruction,
                    temperature=request.temperature,
                    max_output_tokens=request.max_output_tokens,
                ),
            )
            text, truncated = self._extract_response(response)
            finish_reason = "MAX_TOKENS" if truncated else "STOP"
            usage = getattr(response, "usage_metadata", None)
            return LLMResult(
                text=text,
                provider=self.name,
                model=model,
                finish_reason=finish_reason,
                input_tokens=getattr(usage, "prompt_token_count", None),
                output_tokens=getattr(usage, "candidates_token_count", None),
                request_id=getattr(response, "response_id", None),
                latency_seconds=time.perf_counter() - started,
                raw_response=response,
            )
        except Exception as exc:  # noqa: BLE001
            raise normalize_provider_error(exc, self.name) from exc

    @staticmethod
    def _generate_config(**kwargs):
        # Resolve this at call time so the adapter shares the application SDK
        # configuration seam and does not create a second mutable SDK surface.
        from backend.llm import llm as llm_module

        return llm_module.types.GenerateContentConfig(**kwargs)
