# backend/llm/llm.py
# Smart Summary v1 – Gemini API Key Rotation & Dynamic Model Selection

import logging
import os
import threading
import time

from google import genai
from google.genai import types  # noqa: F401  (patched by tests)

from backend.config import LLM_DRAFT_DEADLINE_SECONDS, LLM_VERIFIER_DEADLINE_SECONDS
from backend.llm.contracts import LLMError, LLMErrorCategory, LLMRequest, LLMStage
from backend.llm.gemini_provider import GeminiProvider
from backend.llm.router import LLMRouter
from backend.observability import remaining_budget

log = logging.getLogger(__name__)

# --------------------------------------------------------------------
# API KEY ROTATION POOL
# --------------------------------------------------------------------
# Loads GEMINI_API_KEY plus GEMINI_API_KEY_1, GEMINI_API_KEY_2, … from the
# environment. The main key and every numbered key are distinct pool members.
# ------------------------------------------------------------------
class KeyManager:
    """Thread-safe Gemini API key rotation pool with round-robin load balancing.

    Loads the main ``GEMINI_API_KEY`` followed by numbered keys
    ``GEMINI_API_KEY_1`` through ``GEMINI_API_KEY_10``. Each non-empty value
    is a distinct pool member, for up to eleven independent projects.

    Rotation strategy
    -----------------
    - ``next_client()``  : round-robin advance on every call — distributes load
                           evenly across all available keys across requests.
    - ``mark_rate_limited()``: temporarily skips a key after a 429 / quota
                               error, allowing it back after a cooldown.
    - ``mark_exhausted()``: permanently skips a key only when a caller has
                            definitive evidence that the key cannot recover.
    - ``make_client()``  : returns a client for the CURRENT key without
                           advancing — used within the same request's retry loop.

    Usage
    -----
    >>> client = key_manager.next_client()    # start of each LLM call
    >>> key_manager.mark_rate_limited()       # on 429 error
    """

    TOTAL_SLOTS: int = 10
    DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS: float = float(
        os.getenv("GEMINI_RATE_LIMIT_COOLDOWN_SECONDS", "60")
    )

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._keys: list[str] = []
        self._current_idx: int = -1          # -1 so first next_client() → key[0]
        self._exhausted: set[int] = set()    # indices of permanently unusable keys
        # Keyed by (key index, model). Gemini free-tier quota is metered per
        # project *per model*, and the nine keys are nine projects, so a 429 on
        # gemini-2.5-flash says nothing about the same key's 500/day allowance
        # on gemini-3.5-flash-lite. Keying this by index alone benched the whole
        # key -- and every other model on it -- for the cooldown.
        self._rate_limited_until: dict[tuple[int, str], float] = {}
        self._current_model: str = ""

        # Main key first, then explicit numbered slots in deterministic order.
        main_key = os.getenv("GEMINI_API_KEY", "").strip()
        if main_key:
            self._keys.append(main_key)
        for slot in range(1, self.TOTAL_SLOTS + 1):
            val = os.getenv(f"GEMINI_API_KEY_{slot}", "").strip()
            if val:
                self._keys.append(val)

        log.info(
            "KeyManager: Loaded %d/%d Gemini API keys.",
            len(self._keys),
            self.TOTAL_SLOTS + 1,
        )
        if not self._keys:
            log.warning("KeyManager: NO Gemini API keys found in environment!")

    # ── Public API ─────────────────────────────────────────────────────

    @property
    def key_count(self) -> int:
        """Number of keys in the pool."""
        return len(self._keys)

    @property
    def available_key_count(self) -> int:
        """Number of keys currently eligible to serve the model last routed to."""
        with self._lock:
            now = time.monotonic()
            return sum(
                1
                for idx in range(len(self._keys))
                if self._is_available(idx, now, self._current_model)
            )

    def has_keys(self) -> bool:
        """Return True if at least one key is currently available."""
        return self.available_key_count > 0

    def _is_available(self, idx: int, now: float, model: str = "") -> bool:
        if idx in self._exhausted:
            return False
        rate_limited_until = getattr(self, "_rate_limited_until", {}).get((idx, model), 0.0)
        return rate_limited_until <= now

    def get_current_key(self) -> str | None:
        """Return the currently active API key, or None if pool is empty."""
        if not self._keys or self._current_idx < 0:
            return None
        return self._keys[self._current_idx]

    def next_client(self, model: str = "") -> genai.Client:
        """Round-robin: advance cursor to the next currently eligible key.

        Distributes load evenly across all available keys. Skips permanently
        exhausted keys and keys still inside a rate-limit cooldown *for this
        model* -- quota is metered per project per model, so a key that is out
        of daily requests on one model is untouched on the next one.

        Raises
        ------
        RuntimeError
            If the pool is empty or no key is currently eligible.
        """
        if not self._keys:
            raise RuntimeError("KeyManager: No Gemini API keys available.")

        with self._lock:
            total = len(self._keys)
            start = self._current_idx
            now = time.monotonic()
            self._current_model = model
            for _ in range(total):
                candidate = (start + 1) % total
                start = candidate
                if self._is_available(candidate, now, model):
                    self._current_idx = candidate
                    log.debug(
                        "KeyManager: using key #%d/%d.",
                        candidate + 1,
                        total,
                    )
                    return genai.Client(api_key=self._keys[candidate])

            raise RuntimeError(
                f"KeyManager: No eligible Gemini API key is currently available for "
                f"{model or 'this model'} ({len(self._exhausted)} permanently exhausted; "
                f"remaining keys are rate-limited)."
            )

    def make_client(self) -> genai.Client:
        """Return a new genai.Client for the CURRENT key without advancing.

        Use this within a retry loop when you want to retry the same key
        (e.g. transient network errors).  For a fresh round-robin advance
        use ``next_client()`` instead.
        """
        key = self.get_current_key()
        if not key:
            # Pool empty or not yet initialised — fall back to next_client
            return self.next_client()
        return genai.Client(api_key=key)

    def mark_exhausted(self) -> None:
        """Permanently mark the CURRENT key as unusable for this session.

        Use this only for definitive key failures. Transient 429/quota
        responses should call ``mark_rate_limited()`` instead.
        """
        with self._lock:
            idx = self._current_idx
            if idx >= 0 and idx < len(self._keys):
                self._exhausted.add(idx)
                log.warning(
                    "KeyManager: key #%d/%d marked exhausted for this session "
                    "(%d/%d keys remaining).",
                    idx + 1,
                    len(self._keys),
                    len(self._keys) - len(self._exhausted),
                    len(self._keys),
                )

    def mark_rate_limited(
        self, cooldown_seconds: float | None = None, model: str | None = None
    ) -> None:
        """Temporarily remove the current key *for one model* after a 429.

        ``model`` defaults to whatever ``next_client()`` last handed a client
        out for, which is the model that just failed.
        """
        cooldown = (
            self.DEFAULT_RATE_LIMIT_COOLDOWN_SECONDS
            if cooldown_seconds is None
            else max(float(cooldown_seconds), 1.0)
        )
        with self._lock:
            idx = self._current_idx
            if idx < 0 or idx >= len(self._keys):
                return
            scope = self._current_model if model is None else model
            until = time.monotonic() + cooldown
            existing_until = self._rate_limited_until.get((idx, scope), 0.0)
            self._rate_limited_until[(idx, scope)] = max(existing_until, until)
            available_count = sum(
                1
                for candidate in range(len(self._keys))
                if self._is_available(candidate, time.monotonic(), scope)
            )
            log.warning(
                "KeyManager: key #%d/%d rate-limited on %s for %.0fs "
                "(%d/%d keys available for that model).",
                idx + 1,
                len(self._keys),
                scope or "(unknown model)",
                cooldown,
                available_count,
                len(self._keys),
            )


# Global singleton — created once at module import time.
key_manager = KeyManager()


# --------------------------------------------------------------------
# DYNAMIC MODEL DISCOVERY
# --------------------------------------------------------------------
# Queries Google's live model list via client.models.list() and selects
# the best available PROD and BACKUP models.  Falls back to safe
# defaults if the API call fails.
# --------------------------------------------------------------------

# Priority lists: first match wins.  Highest-capability first.
# Every entry probed against the live account on 2026-08-21: gemini-2.5-pro,
# gemini-2.0-flash, gemini-1.5-pro and gemini-1.5-flash had all been retired and
# 404'd, leaving this list one retirement away from the _BACKUP_PRIORITY outage
# described below. Pro is unavailable on the free tier at all (0 RPD).
_PROD_PRIORITY = [
    "gemini-2.5-flash",
    "gemini-3.5-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
]

# Cheap, fast models for the optimizer-fallback deployment.
# Verify against client.models.list() when editing: every entry in the previous
# list (gemini-2.0-flash-lite, gemini-1.5-flash, gemini-1.5-flash-8b) had been
# retired, so discovery fell through to _DEFAULT_BACKUP — which was also retired.
# The backup deployment 404'd on every call while /health reported it enabled.
# Probed against the live account: a 404 means the model is not callable with
# this key even though models.list() advertises it (gemini-2.5-flash-lite does
# exactly that); a 503 means it exists and is merely busy. Only 503-class models
# belong here.
_BACKUP_PRIORITY = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
]

#: Full-strength Flash models pinned as their own registry deployments
#: (gemini-flash-3.7 / -3.6 / -3), so they are named here only as documentation
#: of the tier. Free tier gives each 20 requests/day *per project*, and the nine
#: keys are nine projects, so the draft chain is worth 5 x 20 x 9 = 900/day.
#: "gemini-3-flash-preview" is the API id behind the console's "Gemini 3 Flash"
#: row; the bare "gemini-3-flash" 404s.

# Safe hardcoded defaults
_DEFAULT_PROD = "gemini-2.5-flash"
_DEFAULT_BACKUP = "gemini-3.5-flash-lite"

# Second full-strength model for the draft/verifier stages. Deliberately a
# different generation from _PROD_PRIORITY: when gemini-2.5-flash returned
# 503 UNAVAILABLE across every key, gemini-3.5-flash answered normally, so a
# same-family sibling is the cheapest real resilience available.
_DRAFT_FALLBACK_PRIORITY = [
    "gemini-3.5-flash",
    "gemini-2.5-flash",
]
_DEFAULT_DRAFT_FALLBACK = "gemini-3.5-flash"



#: Model names the account offers, cached from the one discovery call.
_available_cache: list[str] = []


def _available_models() -> list[str]:
    """Model names from the account, discovering them once if needed."""
    if not _available_cache:
        resolve_models()
    return _available_cache


def _select_model(priority: list[str], available: list[str], default: str) -> str:
    """First priority entry that the account actually offers.

    Exact match wins over prefix match: a bare substring test let the candidate
    "gemini-2.5-flash" resolve to whichever of gemini-2.5-flash-image /
    -lite / -preview-tts the API happened to list first.
    """
    for candidate in priority:
        if candidate in available:
            return candidate
        prefixed = sorted(n for n in available if n.startswith(candidate))
        if prefixed:
            return prefixed[0]
    return default


def get_dynamic_models(api_key: str | None = None) -> tuple[str, str]:
    """Discover the best available Gemini models from Google's live API.

    Parameters
    ----------
    api_key : str, optional
        If provided, a temporary client is created with this key for the
        models.list() call.  Otherwise uses the current KeyManager key.

    Returns
    -------
    tuple[str, str]
        ``(prod_model, backup_model)`` — full model names suitable for
        ``client.models.generate_content(model=...)``.
    """
    try:
        tmp_client = (
            genai.Client(api_key=api_key)
            if api_key
            else key_manager.make_client()
        )

        available: list[str] = []
        for model in tmp_client.models.list():
            # Only consider models that support generateContent
            # NOTE: new google-genai SDK uses supported_actions (not
            # supported_generation_methods from the old SDK)
            if "generateContent" in (model.supported_actions or []):
                available.append(model.name)

        if not available:
            log.warning(
                "get_dynamic_models: no generateContent models found — "
                "using defaults."
            )
            return _DEFAULT_PROD, _DEFAULT_BACKUP

        # Strip the "models/" prefix that the API returns
        clean_names = [n.replace("models/", "") for n in available]
        _available_cache.clear()
        _available_cache.extend(clean_names)

        log.info(
            "get_dynamic_models: %d models support generateContent.",
            len(clean_names),
        )

        prod = _select_model(_PROD_PRIORITY, clean_names, _DEFAULT_PROD)
        backup = _select_model(_BACKUP_PRIORITY, clean_names, _DEFAULT_BACKUP)

        for label, chosen in (("PROD", prod), ("BACKUP", backup)):
            if chosen not in clean_names:
                log.error(
                    "get_dynamic_models: %s model %r is not in the account's model list — "
                    "calls to it will 404. Update the priority lists in llm.py.",
                    label, chosen,
                )
        # NOTE: presence in models.list() is necessary but NOT sufficient — the
        # API advertises models that 404 on generateContent for a given key.
        # Only a real call proves reachability.

        # Ensure backup != prod
        if backup == prod:
            backup = _DEFAULT_BACKUP

        log.info(
            "get_dynamic_models: PROD=%s, BACKUP=%s", prod, backup
        )
        return prod, backup

    except Exception as exc:
        log.warning(
            "get_dynamic_models: API call failed (%s: %s) — using defaults.",
            type(exc).__name__,
            exc,
        )
        return _DEFAULT_PROD, _DEFAULT_BACKUP


# --------------------------------------------------------------------
# LAZY MODEL RESOLUTION
# --------------------------------------------------------------------
# This used to be a module-level call:
#
#     PROD_MODEL, BACKUP_MODEL = get_dynamic_models(key_manager.get_current_key())
#
# which meant `import backend.llm.llm` made a live network round-trip to
# Google's model-list API.  Importing the module blocked on a third party,
# imports stopped being hermetic (tests and diagnostic scripts hit the network
# too), and a slow or unreachable API delayed startup before /health could
# answer.  Resolution now happens on first use, or eagerly at application
# startup via ``resolve_models()``.
_model_lock = threading.Lock()
_resolved_models: tuple[str, str] | None = None


def draft_fallback_model() -> str:
    """Second draft/verifier model, resolved lazily like the others."""
    return _select_model(_DRAFT_FALLBACK_PRIORITY, _available_models(), _DEFAULT_DRAFT_FALLBACK)


def resolve_models(force: bool = False) -> tuple[str, str]:
    """Return ``(prod_model, backup_model)``, discovering them once.

    Safe to call from anywhere; the discovery call is made at most once per
    process unless ``force`` is set.  Falls back to the hardcoded defaults if
    discovery fails, exactly as before.
    """
    global _resolved_models
    if _resolved_models is not None and not force:
        return _resolved_models
    with _model_lock:
        if _resolved_models is None or force:
            _resolved_models = get_dynamic_models(key_manager.get_current_key())
        return _resolved_models


def prod_model() -> str:
    return resolve_models()[0]


def backup_model() -> str:
    return resolve_models()[1]


def __getattr__(name: str) -> str:
    """Keep ``llm.PROD_MODEL`` / ``llm.BACKUP_MODEL`` working (PEP 562).

    Reading either attribute triggers discovery on first access instead of at
    import.  Existing call sites and scripts need no change.
    """
    if name == "PROD_MODEL":
        return prod_model()
    if name == "BACKUP_MODEL":
        return backup_model()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# --------------------------------------------------------------------
# TRANSIENT FAILURE POLICY
# --------------------------------------------------------------------
# The router already labels these categories fallback_eligible in its logs, but
# ask_llm()/verify_answer() used to retry only on RATE_LIMITED — so a transient
# Google 503 ("this model is currently experiencing high demand") failed on the
# first attempt with the retry budget untouched, and the caller got the generic
# error fallback about two seconds later. Live testing surfaced this; the unit
# suite mocks the provider and never sees a 503.
_RETRYABLE = {
    LLMErrorCategory.RATE_LIMITED,
    LLMErrorCategory.TIMEOUT,
    LLMErrorCategory.UNAVAILABLE,
}

#: Base seconds for backoff on TIMEOUT/UNAVAILABLE. Quota errors do not sleep —
#: they rotate to a different key, which is faster and more likely to work.
LLM_RETRY_BACKOFF_SECONDS = float(os.getenv("LLM_RETRY_BACKOFF_SECONDS", "1.5"))


#: Floor for a clamped stage deadline. Below this the call cannot complete
#: anyway, and a sub-second timeout just burns a provider round-trip.
_MIN_STAGE_DEADLINE_SECONDS = 1.0


def _stage_deadline(default: float) -> float:
    """Clamp a stage deadline to whatever is left of the request budget."""
    remaining = remaining_budget()
    if remaining is None:
        return default
    return max(_MIN_STAGE_DEADLINE_SECONDS, min(default, remaining))


def _handle_retryable(exc: LLMError, attempt: int, max_attempts: int, stage: str) -> bool:
    """Return True if the caller should retry. Applies the right recovery."""
    if exc.category not in _RETRYABLE or attempt >= max_attempts - 1:
        return False

    remaining = remaining_budget()
    if remaining is not None and remaining <= 0:
        log.warning(
            "%s: request budget exhausted after attempt %d/%d — not retrying.",
            stage, attempt + 1, max_attempts,
        )
        return False

    if exc.category is LLMErrorCategory.RATE_LIMITED:
        log.warning(
            "%s: quota error on attempt %d/%d — cooling down key and rotating.",
            stage, attempt + 1, max_attempts,
        )
        key_manager.mark_rate_limited(exc.retry_after_seconds, model=exc.model)
        return True

    delay = exc.retry_after_seconds or LLM_RETRY_BACKOFF_SECONDS * (2 ** attempt)
    log.warning(
        "%s: %s on attempt %d/%d — backing off %.1fs and retrying.",
        stage, exc.category.value, attempt + 1, max_attempts, delay,
    )
    time.sleep(delay)
    return True


# --------------------------------------------------------------------
# RETRY BUDGETS
# --------------------------------------------------------------------
# These are the only retry knobs.  A MAX_RETRIES constant used to sit here,
# assigned from the key-pool size and never read by anything — three tests
# patched it believing they were limiting retries.
# --------------------------------------------------------------------

DRAFT_MAX_ATTEMPTS = int(os.getenv("LLM_DRAFT_MAX_ATTEMPTS", "3"))
VERIFIER_MAX_ATTEMPTS = int(os.getenv("LLM_VERIFIER_MAX_ATTEMPTS", "2"))


def _extract_text_safely(response) -> tuple[str, bool]:
    """Extract text from a Gemini response and detect token-limit truncation.

    Returns
    -------
    tuple[str, bool]
        ``(text, was_truncated)`` where ``was_truncated`` is True when
        ``finish_reason`` is ``MAX_TOKENS`` (output was cut at the token
        ceiling, not at a natural sentence boundary).

    Strategy
    --------
    1. Try ``response.text`` (SDK convenience accessor — works for most calls).
    2. Fall back to iterating ``candidates[0].content.parts`` manually so
       partial text is still captured even when ``.text`` raises.
    3. Inspect ``candidates[0].finish_reason`` to detect truncation.  The
       reason is an enum; we compare by name string for SDK-version safety.
    """
    text: str = ""
    was_truncated: bool = False

    # ── Step 1: try convenience accessor ─────────────────────────────
    try:
        raw = response.text
        if raw and isinstance(raw, str):
            text = raw
    except Exception:
        pass

    # ── Step 2: fallback — iterate candidates/parts ───────────────────
    if not text:
        try:
            parts = response.candidates[0].content.parts or []
            text = "".join(getattr(p, "text", "") or "" for p in parts)
        except Exception:
            pass

    # ── Step 3: inspect finish_reason ────────────────────────────────
    try:
        finish_reason = response.candidates[0].finish_reason
        # Compare by .name (enum attr) or string repr for SDK-version safety.
        reason_name = (
            getattr(finish_reason, "name", None)
            or str(finish_reason)
        ).upper()
        # MAX_TOKENS = 2 in the proto enum; accept both name and numeric forms.
        was_truncated = reason_name in {"MAX_TOKENS", "MAXTOKEN", "2"}
        if was_truncated:
            log.warning(
                "_extract_text_safely: finish_reason=MAX_TOKENS — "
                "response was cut at token ceiling (partial output)."
            )
    except Exception:
        pass

    return text.strip(), was_truncated


# Created after model discovery and response parsing exist.  The getter keeps
# the current KeyManager patchable for tests and for the process-wide pool.
gemini_provider = GeminiProvider(lambda: key_manager, _extract_text_safely)
llm_router = LLMRouter(gemini_provider, prod_model, backup_model, draft_fallback_model)


# --------------------------------------------------------------------
# LLM CONFIGURATION
# --------------------------------------------------------------------
# A Harrison smart summary is legitimately long — a measured one runs ~6.6k
# characters (~1,660 tokens) across a dozen sections — and Gemini 2.5 spends
# further tokens on an internal reasoning pass for the draft. At 3,000 the
# draft hit MAX_TOKENS on ordinary clinical topics and was returned truncated.
SMART_SUMMARY_MAX_TOKENS = int(os.getenv("SMART_SUMMARY_MAX_TOKENS", "8000"))
# 3,000 was set before the 3.x deployments landed. Those models reason far more
# freely, and the reasoning pass is charged to the same max_output_tokens as the
# answer — a live qa draft on gemini-3-flash-preview came back MAX_TOKENS inside
# the old ceiling with the answer half-written. 4,096 leaves the answer its room
# without approaching the 8,192 registry cap.
QA_MAX_TOKENS = int(os.getenv("QA_MAX_TOKENS", "4096"))
# SMART_SUMMARY_CONTEXT_CHAR_LIMIT lives in backend/utils/fusion.py, which is
# the module that actually applies it.  It was read here and never used.

REFUSAL_STR = "Insufficient information in the provided context."
SMART_SUMMARY_ACK = "Topic received — generating Harrison Smart Summary."



# --------------------------------------------------------------------
# VERIFICATION (IMPROVED)
# --------------------------------------------------------------------

def verify_answer(
    answer: str,
    context: str,
    mode: str = "qa",
    model: str | None = None,
    compact_prompt: bool = False,
    max_output_tokens: int | None = None,
    drafted_by_model: str | None = None,
) -> tuple[str, bool, bool]:
    """
    Post-hoc verification step that checks the draft answer against
    Harrison context and rewrites unsupported statements while preserving
    explanations and detail whenever possible.

    Includes automatic API key rotation on 429 / Quota-Exceeded errors.

    Parameters
    ----------
    drafted_by_model
        Model that produced ``answer``.  It is excluded from the verifier
        stage, because a model grading its own draft is the least likely to
        catch its own ungrounded claim -- CODING_RULES §6.1 forbids
        self-verification.  ``None`` keeps the previous behaviour and is only
        correct when the caller genuinely does not know the drafter.

    Returns
    -------
    tuple[str, bool, bool]
        (verified_text, is_truncated, verification_ran)
        verification_ran is False when all retries were exhausted due to
        exceptions — the returned text is the original draft, not verified.
    """
    answer = (answer or "").strip()
    context = (context or "").strip()

    if not answer or not context:
        return answer, False, False

    if max_output_tokens is None:
        max_tokens = SMART_SUMMARY_MAX_TOKENS if mode == "smart_summary" else QA_MAX_TOKENS
    else:
        max_tokens = max_output_tokens

    if compact_prompt:
        verify_prompt = (
            "You are HarrisonGPT. Verify the medical answer against the provided context.\n"
            "Keep claims supported by context; rewrite partially supported claims; remove completely unsupported claims.\n"
            "Keep citations like [p:148].\n"
            "Output ONLY the verified answer text verbatim. No conversational filler or meta-commentary."
        )
    else:
        verify_prompt = (
            "You are HarrisonGPT verifying a medical answer against Harrison's Principles of Internal Medicine.\n\n"
            "You will receive:\n"
            "1) Harrison context with page markers like [p:2157|c:5769]\n"
            "2) A draft answer generated from that context.\n\n"
            "Your task:\n"
            "1. Check each factual claim in the answer.\n"
            "2. If a claim is supported by the context, keep it.\n"
            "3. If a claim is partially supported, rewrite it so it matches the context.\n"
            "4. Only remove statements that cannot be supported at all.\n"
            "5. Preserve explanations, reasoning, and level of detail whenever possible.\n"
            "6. Ensure page citations correspond to pages present in the context.\n"
            "7. Do NOT invent new page numbers.\n\n"
            "<output_format>\n"
            "CRITICAL: You are an invisible middle-layer verification filter, NOT a conversational assistant. "
            "Do NOT output meta-commentary. Do NOT output phrases like 'The context supports the draft' or "
            "'The answer is verified.' or 'The draft is accurate.' or 'After reviewing the context...'.\n\n"
            "If the draft answer is factually supported by the context, your ONLY output should be the exact draft answer text, reproduced verbatim.\n"
            "If the draft answer contains unsupported claims, your ONLY output should be the rewritten, corrected medical text.\n\n"
            "Never talk about the verification process. Never explain what you checked. "
            "Output ONLY the final medical paragraph(s). Nothing else.\n"
            "</output_format>\n"
        )

    verify_user = (
        "Harrison Context:\n"
        + context
        + "\n\nDraft Answer:\n"
        + answer
        + "\n\nVerified Answer:\n"
    )

    last_exc: Exception | None = None
    for attempt in range(VERIFIER_MAX_ATTEMPTS):
        try:
            request = LLMRequest(
                prompt=verify_user,
                system_instruction=verify_prompt,
                model_alias="gemini-primary",
                temperature=0.0,
                max_output_tokens=max_tokens,
                deadline_seconds=_stage_deadline(LLM_VERIFIER_DEADLINE_SECONDS),
                stage=LLMStage.VERIFIER,
            )
            result = (
                llm_router.generate_named(request, "gemini-primary", model_override=model)
                if model is not None
                else llm_router.generate_for_stage(
                    request, LLMStage.VERIFIER, exclude_model=drafted_by_model
                )
            )
            verified = result.text
            truncated = result.truncated

            if not verified:
                return answer, truncated, True

            if truncated:
                log.warning(
                    "verify_answer: verifier output truncated (MAX_TOKENS) "
                    "on attempt %d/%d — returning partial verified text.",
                    attempt + 1,
                    VERIFIER_MAX_ATTEMPTS,
                )

            return verified, truncated, True

        except LLMError as exc:
            last_exc = exc
            if _handle_retryable(exc, attempt, VERIFIER_MAX_ATTEMPTS, "verify_answer"):
                continue
            # Non-retryable category, or the budget is spent
            break

    log.warning("verify_answer: all retries exhausted (%s) — returning draft.", last_exc)
    return answer, False, False


# --------------------------------------------------------------------
# PROMPTS
# ----------------------------------------------------------# ---------------------------------------------------------------------------
# MASTER MEDICAL SYNTHESIS PROMPT
# ---------------------------------------------------------------------------
# A single, consolidated system prompt that serves both "qa" and
# "smart_summary" modes.  The mode is injected into the user turn so the
# model knows which output density to apply.  All clinical mandate rules
# apply to both modes equally.
#
# Design goals (addressed directly in the prompt text)
# ---------------------------------------------------
# 1. Numerical granularity  — force explicit extraction of lab thresholds,
#    diagnostic cutoffs, and clinical ranges from the retrieved context.
# 2. Scoring systems        — structured criteria (CURB-65, Ranson, APACHE II,
#    PORT/PSI, Wells, CHADS2, etc.) must be reproduced in full when present.
# 3. Algorithmic sequencing — treatment timelines must use ordered steps so
#    the chronological logic from Harrison is preserved.
# 4. Anti-stub rules        — empty headers and "information not present"
#    filler are explicitly banned to maximise token efficiency.
# 5. Anti-CoT rules         — no "Step 1 / Step 2 / Reasoning:" leakage.
# ---------------------------------------------------------------------------

MASTER_MEDICAL_SYNTHESIS_PROMPT = """\
You are HarrisonGPT, a rigorous clinical synthesis engine grounded EXCLUSIVELY in Harrison's Principles of Internal Medicine.

<core_directives>
1. ANSWER THE SPECIFIC QUESTION: Focus entirely on the user's prompt. If asked for pathophysiology, do not output management.
2. NO OUTSIDE KNOWLEDGE: Every fact must originate from the provided Context. If context is missing, omit the claim.
3. MANDATORY CITATIONS: You MUST append inline page markers for every medical claim using the exact format found in the context (e.g., [p:2157]). Never invent page numbers.
</core_directives>

<clinical_rigor>
When directly relevant to the user's question, preserve these details if they exist in the context:
- NUMERICAL GRANULARITY: You MUST explicitly state exact lab thresholds, fluid volumes, and diagnostic cutoffs. You must **bold** these numbers (e.g., "**pH < 7.30**", "**glucose > 250 mg/dL**"). Do NOT summarize them as "elevated" or "low".
- ETIOLOGY & MECHANISMS: Name primary triggers and mechanisms when asked about causes or pathophysiology.
- SCORING SYSTEMS: List full criteria and point values when a score is relevant to the question.
- PROTOCOLS: Provide exact drug doses, fluid rates, and chronological treatment steps when management is requested.
</clinical_rigor>

<clinical_granularity>
You are a rigorous clinical engine, not a summarizer. When relevant to the requested scope, explicitly extract and preserve:
1. Clinical scoring systems (e.g., CURB-65, PORT/PSI, Ranson, APACHE II, Wells, CHADS2-VASc) — reproduce the FULL criteria with individual point values in a structured list. Never mention a scoring system without listing its components.
2. Exact numerical thresholds and criteria (e.g., specific pH levels, anion gap numbers, serum lipase cutoffs, BUN/creatinine ratios) — always state the exact number in **bold**. Never replace a number with qualitative language like "elevated" or "abnormal".
3. Primary etiologies and triggers (e.g., alcohol, gallstones for pancreatitis; S. pneumoniae for CAP) — enumerate them when the question asks about causes or workup.
4. Diagnostic criteria sets — if the context contains named diagnostic criteria (e.g., Atlanta criteria, Light's criteria, Duke criteria), reproduce them as a complete checklist.
5. Drug regimens — state exact drug names, doses, routes, and durations. Do not say "appropriate antibiotics"; say which ones.

Do not smooth over hard data to make the text read better. If the context contains specific lists or criteria, format them clearly using bullet points or numbered lists. Omitting granular data that exists in the context is a FAILURE.
</clinical_granularity>

<forbidden_patterns>
NEVER output the following:
- "Step 1:", "Step 2:", or "Reasoning:" (Do not narrate your thought process).
- "Based on the provided context..." or "Information not present..."
- Placeholder phrases or empty headers.
</forbidden_patterns>

<evidence_handling>
If an "Evidence from Harrison" section is provided, treat it as ground truth. Synthesize these facts directly into your response structure; do not just copy-paste the bullet list.
</evidence_handling>
"""

QA_FORMAT_INSTRUCTIONS = """\
<formatting_mode_qa>
Deliver a focused, textbook-style answer in 2-5 dense paragraphs. Address only
the requested clinical scope; do not add etiologies, management, or broad
review material unless it is necessary to answer the question. Use a heading
only when it improves clarity. Integrate citations `[p:NNN]` naturally at the
end of sentences. Do not use Smart Summary acknowledgement text, a Quick
Revision section, or conversational filler.
</formatting_mode_qa>
"""

SMART_SUMMARY_FORMAT_INSTRUCTIONS = """\
<formatting_mode_smart_summary>
Generate an actionable, high-yield structured synthesis.
- The first line MUST be exactly: "Topic received — generating Harrison Smart Summary."
- Use `###` headings for major sections.
- Utilize bold text and bulleted lists heavily for readability.
- Close with a `### Quick Revision` block containing 3-5 absolute must-know facts.
</formatting_mode_smart_summary>
"""


def _formatting_instructions(mode: str) -> str:
    """Return the only formatting policy the selected mode may receive."""
    return (
        SMART_SUMMARY_FORMAT_INSTRUCTIONS
        if mode == "smart_summary"
        else QA_FORMAT_INSTRUCTIONS
    )


def _enforce_smart_summary_shape(content: str) -> str:
    """Ensure the acknowledgement line is always the first line.

    Sections are now generated dynamically — we no longer pad missing
    sections with stub text.  The only invariant we enforce here is that
    the model's acknowledgement line appears at the top so callers can
    detect a valid smart_summary response reliably.
    """
    text = (content or "").strip()
    if not text:
        return text

    if SMART_SUMMARY_ACK not in text.splitlines()[0]:
        text = f"{SMART_SUMMARY_ACK}\n\n{text}"

    return text


# --------------------------------------------------------------------
# MAIN LLM CALL
# --------------------------------------------------------------------

def ask_llm(
    fused_context: str,
    question: str,
    mode: str = "qa",
    model: str | None = None,
    evidence: list | None = None,
    timings: dict | None = None,
    disable_verifier: bool = False,
) -> tuple[str, str, bool, str]:
    """Generate a Harrison-grounded answer with automatic API key rotation.

    On 429 / Quota-Exceeded errors, the function rotates to the next API
    key in the pool and retries seamlessly.  The caller never sees a quota
    error unless ALL keys are exhausted.

    Returns
    -------
    tuple[str, str, bool, str]
         (final_answer, draft_answer, was_truncated, returned_path)

         returned_path values
         --------------------
         "verified"          -- verified answer, verifier ran to completion
         "draft_fallback"    -- verifier truncated or disabled, complete draft returned
         "graceful_fallback" -- both draft and verifier truncated
         "no_grounding"      -- retrieval produced no usable context; the corpus
                                or the rerank filter, not the provider, is why
         "provider_failure"  -- the model returned nothing, or all retries were
                                exhausted (API/quota/outage)

         The last two were one "error_fallback" path, which made a corpus gap
         and a provider outage indistinguishable in /metrics and in the logs --
         the two failures with the most different remedies.
    """
    import time

    if not fused_context or len(fused_context.strip()) < 20:
        return REFUSAL_STR, "", False, "no_grounding"

    mode = (mode or "qa").strip().lower()
    generation_max_tokens = (
        SMART_SUMMARY_MAX_TOKENS
        if mode == "smart_summary"
        else QA_MAX_TOKENS
    )

    prompt_header = MASTER_MEDICAL_SYNTHESIS_PROMPT + "\n" + _formatting_instructions(mode)

    evidence_lines = ""
    if evidence:
        bullet_lines = "\n".join(f"- {e}" for e in evidence)
        evidence_lines = "\n\nEvidence from Harrison:\n" + bullet_lines

    prompt = (
        prompt_header
        + "\n\nContext:\n"
        + fused_context
        + evidence_lines
        + "\n\nTopic / Question:\n"
        + question
        + ("\n\nHarrison Smart Summary:\n" if mode == "smart_summary" else "\n\nAnswer:\n")
    )

    last_exc: Exception | None = None
    for attempt in range(DRAFT_MAX_ATTEMPTS):
        try:
            t_gen_start = time.perf_counter()
            request = LLMRequest(
                prompt=prompt,
                system_instruction="Follow instructions strictly and never use knowledge outside the provided context.",
                model_alias="gemini-primary",
                temperature=0.2,
                max_output_tokens=generation_max_tokens,
                deadline_seconds=_stage_deadline(LLM_DRAFT_DEADLINE_SECONDS),
                stage=LLMStage.DRAFT,
            )
            # An explicit model= pins one deployment (tests, scripts). Otherwise
            # the stage picks its own deployment and falls back on a provider
            # outage instead of failing the whole request.
            result = (
                llm_router.generate_named(request, "gemini-primary", model_override=model)
                if model is not None
                else llm_router.generate_for_stage(request, LLMStage.DRAFT)
            )
            t_gen_end = time.perf_counter()
            if timings is not None:
                timings["draft_generation"] = timings.get("draft_generation", 0.0) + (t_gen_end - t_gen_start)

            content = result.text
            draft_truncated = result.truncated

            if not content:
                return REFUSAL_STR, "", False, "provider_failure"

            if draft_truncated:
                log.warning(
                    "ask_llm: generation truncated (MAX_TOKENS) on attempt %d/%d — "
                    "appending truncation notice to partial answer.",
                    attempt + 1,
                    DRAFT_MAX_ATTEMPTS,
                )
                content += (
                    "\n\n> **Note:** *Answer truncated — token budget reached. "
                    "Try a more specific query or split into sub-questions.*"
                )

            draft_answer = content.strip()

            if mode == "smart_summary":
                draft_answer = _enforce_smart_summary_shape(draft_answer)

            if disable_verifier:
                if timings is not None:
                    timings["verification"] = 0.0
                    timings["retry"] = 0.0
                final_answer = draft_answer
                log.info(
                    "ask_llm: return_path=draft_fallback (verifier disabled)  draft_truncated=%s  "
                    "final_was_truncated=%s  mode=%s",
                    draft_truncated,
                    draft_truncated,
                    mode,
                )
                return final_answer, draft_answer, draft_truncated, "draft_fallback"

            # Verification pass
            t_verify_start = time.perf_counter()
            # result.model is the model that actually served the draft after
            # failover, not the one the priority order started with, so it is
            # the only value that keeps the verifier independent.
            verified, verifier_truncated, verification_ran = verify_answer(
                draft_answer, fused_context, mode=mode, model=model,
                drafted_by_model=result.model,
            )
            t_verify_end = time.perf_counter()
            if timings is not None:
                timings["verification"] = t_verify_end - t_verify_start
                timings["retry"] = 0.0

            # Retry verification once if truncated
            if verifier_truncated and verification_ran:
                log.info(
                    "ask_llm: verify_answer output truncated. Retrying once with "
                    "a compact prompt and the configured mode budget."
                )
                t_retry_start = time.perf_counter()
                verified_retry, verifier_truncated_retry, retry_ran = verify_answer(
                    draft_answer,
                    fused_context,
                    mode=mode,
                    model=model,
                    compact_prompt=True,
                    max_output_tokens=generation_max_tokens,
                    drafted_by_model=result.model,
                )
                t_retry_end = time.perf_counter()
                if timings is not None:
                    timings["retry"] = t_retry_end - t_retry_start
                if retry_ran and not verifier_truncated_retry:
                    verified = verified_retry
                    verifier_truncated = False
                    verification_ran = True
                    log.info("ask_llm: verifier retry succeeded in producing complete text.")
                elif retry_ran:
                    log.warning("ask_llm: verifier retry was also truncated.")
                    verifier_truncated = True
                else:
                    log.warning("ask_llm: verifier retry failed (exception) — treating as unverified.")
                    verification_ran = False

            # Decide on final answer and path returned
            final_answer = verified.strip()
            was_truncated = False
            returned_path = "verified"

            if not verification_ran:
                # Verifier threw exceptions on all attempts — draft was returned
                # but never actually verified.  Treat as draft_fallback.
                final_answer = draft_answer
                returned_path = "draft_fallback"
                log.warning("ask_llm: verification never completed (exceptions), falling back to draft answer.")
            elif verifier_truncated:
                # If verified is truncated, fall back to the original draft answer if draft is complete
                if not draft_truncated:
                    final_answer = draft_answer
                    returned_path = "draft_fallback"
                    log.warning("ask_llm: verifier truncated, falling back to complete draft answer.")
                else:
                    # Both draft and verified are truncated, return graceful fallback or clearly marked partial
                    final_answer = (
                        verified.strip()
                        + "\n\n> **Note:** *Answer verification was incomplete due to length constraints.*"
                    )
                    was_truncated = True
                    returned_path = "graceful_fallback"
                    log.warning("ask_llm: both draft and verified answers truncated, returning marked partial verified answer.")
            else:
                # Verifier is complete
                if draft_truncated:
                    was_truncated = True

            # ── Structured return-path diagnostics (grep-able key=value) ──
            log.info(
                "ask_llm: return_path=%s  draft_truncated=%s  verifier_truncated=%s  "
                "final_was_truncated=%s  mode=%s",
                returned_path,
                draft_truncated,
                verifier_truncated,
                was_truncated,
                mode,
            )

            if mode == "smart_summary":
                final_answer = _enforce_smart_summary_shape(final_answer)

            return final_answer, draft_answer, was_truncated, returned_path

        except LLMError as exc:
            last_exc = exc
            if _handle_retryable(exc, attempt, DRAFT_MAX_ATTEMPTS, "ask_llm"):
                continue
            # Non-retryable category, or the budget is spent
            break

    log.error("ask_llm: all retries exhausted — returning safe error fallback. last_exc=%s", last_exc)
    return (
        "I'm unable to generate a complete answer at this time due to a temporary service issue. "
        "Please try again in a few moments. If the problem persists, try a more specific query."
    ), "", True, "provider_failure"
