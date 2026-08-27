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


#: Every provider spelling of "output was cut at the token ceiling".  Gemini
#: says MAX_TOKENS; Mistral and Groq both speak OpenAI's dialect and say
#: "length", and Mistral adds "model_length" for its context limit.  Nothing
#: recognised either, so a truncated Mistral or Groq answer reached the user
#: with no truncation notice and no confidence cap, and was cached and
#: re-served as High confidence.  Compare through LLMResult.truncated, never
#: against a literal.
TRUNCATED_FINISH_REASONS = frozenset({"MAX_TOKENS", "LENGTH", "MODEL_LENGTH"})


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

    @property
    def truncated(self) -> bool:
        """True when generation stopped at the token ceiling, not a stop token."""
        return (self.finish_reason or "").upper() in TRUNCATED_FINISH_REASONS


class LLMError(RuntimeError):
    """Normalized provider failure used for routing policy decisions."""

    def __init__(
        self,
        category: LLMErrorCategory,
        message: str,
        *,
        retry_after_seconds: float | None = None,
        provider: str | None = None,
        model: str | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.retry_after_seconds = retry_after_seconds
        self.provider = provider
        #: The model that produced this failure. Gemini quota cooldowns are
        #: scoped per (key, model), and KeyManager otherwise has to infer the
        #: model from the last next_client() call — which is process-global, so
        #: a concurrent request on another model can win the race and get the
        #: cooldown recorded against it instead.
        self.model = model
