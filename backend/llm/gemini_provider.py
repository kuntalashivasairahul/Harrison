"""Gemini adapter used by the stage-aware router."""

from __future__ import annotations

import time
from collections.abc import Callable

from backend.llm.contracts import LLMError, LLMErrorCategory, LLMRequest, LLMResult, LLMStage


def normalize_provider_error(exc: Exception, provider: str) -> LLMError:
    text = str(exc).lower()
    # "rate_limit" (underscored) and 413 are both Groq spellings: a prompt that
    # exceeds the per-minute token allowance comes back as HTTP 413 with
    # code="rate_limit_exceeded", which matched none of the spaced markers and
    # fell through to UNKNOWN — that is not fallback-eligible, so the stage gave
    # up instead of trying the next deployment.
    if any(marker in text for marker in ("429", "413", "quota", "rate limit", "rate_limit", "resource exhausted", "resourceexhausted")):
        return LLMError(LLMErrorCategory.RATE_LIMITED, str(exc), provider=provider)
    # "deadline_exceeded" (underscored) and 504 are how google-genai reports a
    # client-side timeout: "504 DEADLINE_EXCEEDED". The spaced spelling missed
    # it and 504 is not in the 500/502/503 rule below, so it fell through to
    # UNKNOWN -- not fallback-eligible, so the first deployment to time out
    # aborted the stage with the rest of the chain untried. Same defect shape as
    # the "rate_limit" and "does not exist" spellings above.
    if any(marker in text for marker in ("timeout", "timed out", "deadline exceeded", "deadline_exceeded", "504")):
        return LLMError(LLMErrorCategory.TIMEOUT, str(exc), provider=provider)
    if any(marker in text for marker in ("401", "403", "api key", "authentication", "permission denied")):
        return LLMError(LLMErrorCategory.AUTH, str(exc), provider=provider)
    # Checked before the 400/invalid-argument rule — which is where this
    # comment always claimed it ran, while the code had it second. Mistral
    # reports a retired model as HTTP 400 with "does not exist" in the body, so
    # the 400 marker won and the error became INVALID_REQUEST: not
    # fallback-eligible, so one dead model aborted the whole stage. Three
    # separate outages in this project traced back to a model being retired.
    if any(marker in text for marker in ("404", "not_found", "not found", "does not exist", "is not supported")):
        return LLMError(LLMErrorCategory.NOT_FOUND, str(exc), provider=provider)
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
            # Pass the model so KeyManager scopes rotation and cooldown to
            # this model's quota bucket rather than the whole key.
            client = self._key_manager_getter().next_client(model)
            config_kwargs = {
                "system_instruction": request.system_instruction,
                "temperature": request.temperature,
                "max_output_tokens": request.max_output_tokens,
                # The router clamps deadline_seconds against the deployment
                # timeout and the request budget, but nothing ever handed it to
                # the SDK, so for Gemini it bounded nothing: a call that stopped
                # responding hung the request with no timeout at any layer. Seen
                # live on gemini-3.7-flash — 200s, no result, no error.
                # google-genai wants milliseconds. A TIMEOUT is fallback
                # eligible, so this escalates now instead of hanging.
                "http_options": {"timeout": int(request.deadline_seconds * 1000)},
            }
            thinking = self._thinking_config(request.stage, model)
            if thinking is not None:
                config_kwargs["thinking_config"] = thinking

            response = client.models.generate_content(
                model=model,
                contents=request.prompt,
                config=self._generate_config(**config_kwargs),
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
            error = normalize_provider_error(exc, self.name)
            error.model = model
            raise error from exc

    #: Stages that must not spend their output budget on an internal reasoning
    #: pass.  Gemini 2.5 models think by default, and those tokens come out of
    #: max_output_tokens: a trivial verifier call measured 561 thinking tokens
    #: against 17 tokens of answer.  On a real smart_summary the verifier has to
    #: reproduce a ~1,660-token draft inside a 3,000-token ceiling, so thinking
    #: pushed it over and it returned MAX_TOKENS.  Every smart summary then took
    #: the draft_fallback path and was capped at Medium confidence — the answer
    #: was never actually verified.
    #:
    #: The verifier is a deterministic (temperature=0) rewrite; it has nothing
    #: to reason about.  The draft stage keeps thinking, where it earns its cost.
    _NO_THINKING_STAGES = frozenset({LLMStage.VERIFIER, LLMStage.OPTIMIZER})

    #: How to say "do not think" — the knob is model-dependent, and getting it
    #: wrong fails loudly on one model and silently on another.  Probed live:
    #:
    #:   model                   budget=0        level=MINIMAL   level=LOW
    #:   gemini-2.5-flash        no thinking     400             400
    #:   gemini-3.5-flash        no thinking     no thinking     58 tok
    #:   gemini-3-flash-preview  no thinking     no thinking     81 tok
    #:   gemini-3.6-flash        400             no thinking     53 tok
    #:   gemini-3.7-flash        IGNORED         400             38 tok
    #:
    #: gemini-3.6 rejecting budget=0 is a 400, which is INVALID_REQUEST and so
    #: not fallback-eligible: it would abort the verifier stage outright rather
    #: than deferring to the next deployment.  gemini-3.7 is worse — it accepts
    #: the zero budget and thinks anyway (248 tokens, finish_reason=MAX_TOKENS
    #: on a 256-token ceiling), which is the silent MAX_TOKENS verifier failure
    #: this whole mechanism exists to prevent.  No setting reaches zero thinking
    #: on 3.7, so it gets the lowest level it accepts and is kept off the
    #: verifier stage in the registry.
    _THINKING_LEVEL_BY_MODEL = {"gemini-3.7-flash": "LOW"}

    #: The draft stage keeps thinking — synthesis is where it earns its cost —
    #: but only 2.5-flash gets to pick how much.  The 3.x models think far more
    #: freely at their default, and a live qa draft on gemini-3-flash-preview
    #: came back finish_reason=MAX_TOKENS inside the 3,000-token QA_MAX_TOKENS
    #: ceiling: the reasoning pass ate the answer's budget.  LOW is the floor
    #: that still reasons (81 tokens on 3-flash-preview, per the table above)
    #: rather than MINIMAL, which is indistinguishable from off.  2.x drafts are
    #: untouched — the decision to leave them alone predates the 3.x
    #: deployments and still holds for them.
    _DRAFT_THINKING_LEVEL = "LOW"

    @classmethod
    def _thinking_config(cls, stage: LLMStage, model: str = ""):
        is_gemini_3 = model.startswith("gemini-3")
        if stage in cls._NO_THINKING_STAGES:
            # level=None below means the zero budget: 2.x and anything
            # unrecognized reject thinking_level, and the zero budget is the
            # long-standing working default there.
            level = cls._THINKING_LEVEL_BY_MODEL.get(model, "MINIMAL") if is_gemini_3 else None
        elif stage is LLMStage.DRAFT and is_gemini_3:
            level = cls._DRAFT_THINKING_LEVEL
        else:
            return None

        # Imported at call time, not module scope.  A module-level binding is
        # captured whenever this module happens to be first imported, and one
        # test suite imports it while a stub SDK is installed in sys.modules —
        # the binding then pointed at the stub for the rest of the process and
        # this silently returned None.
        from google.genai import types as genai_types

        thinking_config_cls = getattr(genai_types, "ThinkingConfig", None)
        if thinking_config_cls is None:
            # Older SDK without a thinking budget knob — nothing to disable.
            return None
        if level is None:
            return thinking_config_cls(thinking_budget=0)
        return thinking_config_cls(thinking_level=level)

    @staticmethod
    def _generate_config(**kwargs):
        # Resolve this at call time so the adapter shares the application SDK
        # configuration seam and does not create a second mutable SDK surface.
        from backend.llm import llm as llm_module

        return llm_module.types.GenerateContentConfig(**kwargs)
