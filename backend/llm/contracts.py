"""Provider-neutral contracts for Harrison's LLM routing layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class LLMStage(str, Enum):
    OPTIMIZER = "optimizer"
    DRAFT = "draft"
    VERIFIER = "verifier"


class LLMErrorCategory(str, Enum):
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    # A model the provider no longer serves for this key. Permanent for that
    # deployment, so retrying it is pointless — but a different deployment may
    # well work, which is why it is fallback-eligible but not retry-eligible.
    NOT_FOUND = "not_found"
    INVALID_REQUEST = "invalid_request"
    AUTH = "auth"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class LLMRequest:
    prompt: str
    system_instruction: str
    model_alias: str
    temperature: float
    max_output_tokens: int
    deadline_seconds: float
    stage: LLMStage


@dataclass
class LLMResult:
    text: str
    provider: str
    model: str
    finish_reason: str = "UNKNOWN"
    input_tokens: int | None = None
    output_tokens: int | None = None
    request_id: str | None = None
    latency_seconds: float = 0.0
    raw_response: Any = field(default=None, repr=False)


class LLMError(RuntimeError):
    """Normalized provider failure used for routing policy decisions."""

    def __init__(
        self,
        category: LLMErrorCategory,
        message: str,
        *,
        retry_after_seconds: float | None = None,
        provider: str | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.retry_after_seconds = retry_after_seconds
        self.provider = provider
