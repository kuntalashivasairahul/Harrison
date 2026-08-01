"""Direct Groq adapter for non-authoritative Harrison stages."""

from __future__ import annotations

import os
import re
import time

from groq import Groq

from backend.llm.contracts import LLMError, LLMErrorCategory, LLMRequest, LLMResult
from backend.llm.gemini_provider import normalize_provider_error


def _retry_after_from_message(message: str) -> float | None:
    match = re.search(r"retry[- ]after\s*[:=]?\s*(\d+(?:\.\d+)?)", message, re.I)
    return float(match.group(1)) if match else None


class GroqProvider:
    name = "groq"

    def __init__(self) -> None:
        self._api_key = os.getenv("GROQ_API_KEY", "").strip()

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    def generate(self, request: LLMRequest, model: str) -> LLMResult:
        if not self.configured:
            raise LLMError(LLMErrorCategory.AUTH, "GROQ_API_KEY is not configured.", provider=self.name)

        started = time.perf_counter()
        try:
            client = Groq(api_key=self._api_key, timeout=request.deadline_seconds)
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": request.system_instruction},
                    {"role": "user", "content": request.prompt},
                ],
                temperature=request.temperature,
                max_tokens=request.max_output_tokens,
            )
            choice = response.choices[0] if response.choices else None
            text = (getattr(getattr(choice, "message", None), "content", None) or "").strip()
            finish_reason = str(getattr(choice, "finish_reason", "UNKNOWN")).upper()
            usage = getattr(response, "usage", None)
            return LLMResult(
                text=text,
                provider=self.name,
                model=model,
                finish_reason=finish_reason,
                input_tokens=getattr(usage, "prompt_tokens", None),
                output_tokens=getattr(usage, "completion_tokens", None),
                request_id=getattr(response, "id", None),
                latency_seconds=time.perf_counter() - started,
                raw_response=response,
            )
        except LLMError:
            raise
        except Exception as exc:  # noqa: BLE001
            normalized = normalize_provider_error(exc, self.name)
            if normalized.category == LLMErrorCategory.RATE_LIMITED:
                normalized.retry_after_seconds = _retry_after_from_message(str(exc))
            raise normalized from exc
