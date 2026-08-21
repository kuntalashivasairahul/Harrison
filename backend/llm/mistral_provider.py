"""Direct Mistral adapter for non-authoritative Harrison stages.

Mistral is reached over its plain HTTPS chat-completions endpoint using
``urllib`` rather than the ``mistralai`` SDK or ``httpx``.  Both would be new
runtime dependencies: ``httpx`` is listed in ``requirements-dev.txt`` and
RULE 6.1a forbids importing it from ``backend/``, and the SDK would need a
RULE 6.2 justification to buy one JSON POST.  The request is a single blocking
call with a deadline, exactly like the Groq and Gemini adapters.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request

from backend.llm.contracts import LLMError, LLMErrorCategory, LLMRequest, LLMResult
from backend.llm.gemini_provider import normalize_provider_error

_ENDPOINT = "https://api.mistral.ai/v1/chat/completions"


class MistralProvider:
    name = "mistral"

    def __init__(self) -> None:
        self._api_key = os.getenv("MISTRAL_API_KEY", "").strip()

    @property
    def configured(self) -> bool:
        return bool(self._api_key)

    def generate(self, request: LLMRequest, model: str) -> LLMResult:
        if not self.configured:
            raise LLMError(LLMErrorCategory.AUTH, "MISTRAL_API_KEY is not configured.", provider=self.name)

        started = time.perf_counter()
        try:
            payload = self._post(request, model)
            choice = (payload.get("choices") or [{}])[0]
            usage = payload.get("usage") or {}
            return LLMResult(
                text=(choice.get("message", {}).get("content") or "").strip(),
                provider=self.name,
                model=payload.get("model") or model,
                finish_reason=str(choice.get("finish_reason") or "UNKNOWN").upper(),
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
                request_id=payload.get("id"),
                latency_seconds=time.perf_counter() - started,
                raw_response=payload,
            )
        except LLMError:
            raise
        except urllib.error.HTTPError as exc:
            # The status line alone ("HTTP Error 429: Too Many Requests") is
            # thin; the body carries the provider's own error code, which is
            # what normalize_provider_error() matches on.
            body = self._read_body(exc)
            normalized = normalize_provider_error(Exception(f"Error code: {exc.code} - {body}"), self.name)
            if normalized.category == LLMErrorCategory.RATE_LIMITED:
                normalized.retry_after_seconds = self._retry_after(exc)
            raise normalized from exc
        except Exception as exc:  # noqa: BLE001
            raise normalize_provider_error(exc, self.name) from exc

    def _post(self, request: LLMRequest, model: str) -> dict:
        body = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": request.system_instruction},
                {"role": "user", "content": request.prompt},
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens,
        }).encode("utf-8")
        http_request = urllib.request.Request(
            _ENDPOINT,
            data=body,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(http_request, timeout=request.deadline_seconds) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _read_body(exc: urllib.error.HTTPError) -> str:
        try:
            return exc.read().decode("utf-8", "replace")[:500]
        except Exception:  # noqa: BLE001
            return exc.reason or ""

    @staticmethod
    def _retry_after(exc: urllib.error.HTTPError) -> float | None:
        value = (exc.headers or {}).get("Retry-After")
        try:
            return float(value) if value else None
        except (TypeError, ValueError):
            return None
