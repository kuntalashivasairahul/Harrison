# backend/llm/llm.py
# Smart Summary v1 – Gemini API Key Rotation & Dynamic Model Selection

import os
import logging
import threading
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pathlib import Path

# --------------------------------------------------------------------
# ENV LOADING
# --------------------------------------------------------------------
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

log = logging.getLogger(__name__)

# --------------------------------------------------------------------
# API KEY ROTATION POOL
# --------------------------------------------------------------------
# Loads all GEMINI_API_KEY_1, GEMINI_API_KEY_2, … from the environment.
# Falls back to the single GEMINI_API_KEY if no numbered keys are found.
# ------------------------------------------------------------------
class KeyManager:
    """Thread-safe Gemini API key rotation pool with round-robin load balancing.

    Loads keys from ``GEMINI_API_KEY_1`` through ``GEMINI_API_KEY_10``.
    ``GEMINI_API_KEY`` (no suffix) is treated as an alias for slot 1.

    Rotation strategy
    -----------------
    - ``next_client()``  : round-robin advance on every call — distributes load
                           evenly across all available keys across requests.
    - ``mark_exhausted()``: permanently skips a key for the session when it
                           returns a 429 / quota error.
    - ``make_client()``  : returns a client for the CURRENT key without
                           advancing — used within the same request's retry loop.
    - ``rotate()``       : legacy helper; advances to next non-exhausted key.

    Usage
    -----
    >>> client = key_manager.next_client()    # start of each LLM call
    >>> key_manager.mark_exhausted()          # on 429 error
    """

    TOTAL_SLOTS: int = 10

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._keys: list[str] = []
        self._current_idx: int = -1          # -1 so first next_client() → key[0]
        self._exhausted: set[int] = set()    # indices of quota-exceeded keys

        # Explicit 10-slot loading — deterministic order, no env-scan surprises.
        for slot in range(1, self.TOTAL_SLOTS + 1):
            val = os.getenv(f"GEMINI_API_KEY_{slot}", "").strip()
            if not val and slot == 1:
                # Bare GEMINI_API_KEY is an alias for slot 1
                val = os.getenv("GEMINI_API_KEY", "").strip()
            if val:
                self._keys.append(val)

        log.info(
            "KeyManager: Loaded %d/%d Gemini API keys.",
            len(self._keys),
            self.TOTAL_SLOTS,
        )
        if not self._keys:
            log.warning("KeyManager: NO Gemini API keys found in environment!")

    # ── Public API ─────────────────────────────────────────────────────

    @property
    def key_count(self) -> int:
        """Number of keys in the pool."""
        return len(self._keys)

    def has_keys(self) -> bool:
        """Return True if at least one key is available."""
        return len(self._keys) > 0

    def get_current_key(self) -> str | None:
        """Return the currently active API key, or None if pool is empty."""
        if not self._keys or self._current_idx < 0:
            return None
        return self._keys[self._current_idx]

    def next_client(self) -> genai.Client:
        """Round-robin: advance cursor to the next non-exhausted key.

        Distributes load evenly across all available keys.  Skips any key
        that has been marked exhausted in this session.

        Raises
        ------
        RuntimeError
            If the pool is empty or all keys are exhausted.
        """
        if not self._keys:
            raise RuntimeError("KeyManager: No Gemini API keys available.")

        with self._lock:
            total = len(self._keys)
            start = self._current_idx
            for _ in range(total):
                candidate = (start + 1) % total
                start = candidate
                if candidate not in self._exhausted:
                    self._current_idx = candidate
                    log.debug(
                        "KeyManager: using key #%d/%d.",
                        candidate + 1,
                        total,
                    )
                    return genai.Client(api_key=self._keys[candidate])

            raise RuntimeError(
                f"KeyManager: All {total} API key(s) exhausted for this session."
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
        """Mark the CURRENT key as quota-exceeded for this session.

        The key will be skipped by all future ``next_client()`` and
        ``rotate()`` calls.  Thread-safe.
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

    def rotate(self) -> str | None:
        """Advance to the next non-exhausted key and return the new key.

        Thread-safe.  Wraps around after exhausting the pool.  Returns
        ``None`` if the pool is empty or all keys are exhausted.
        """
        if not self._keys:
            return None
        with self._lock:
            total = len(self._keys)
            start = self._current_idx
            for _ in range(total):
                candidate = (start + 1) % total
                start = candidate
                if candidate not in self._exhausted:
                    self._current_idx = candidate
                    key = self._keys[candidate]
                    log.info(
                        "KeyManager: rotated to key #%d/%d.",
                        candidate + 1,
                        total,
                    )
                    return key
            log.error("KeyManager: all keys exhausted — cannot rotate further.")
            return None


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
_PROD_PRIORITY = [
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-1.5-pro",
    "gemini-1.5-flash",
]

_BACKUP_PRIORITY = [
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
]

# Safe hardcoded defaults
_DEFAULT_PROD = "gemini-2.5-flash"
_DEFAULT_BACKUP = "gemini-1.5-flash"


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

        log.info(
            "get_dynamic_models: %d models support generateContent.",
            len(clean_names),
        )

        # Select PROD model — first match in priority list
        prod = _DEFAULT_PROD
        for candidate in _PROD_PRIORITY:
            if any(candidate in name for name in clean_names):
                # Find the exact matching name
                for name in clean_names:
                    if candidate in name:
                        prod = name
                        break
                break

        # Select BACKUP model — first match in backup priority list
        backup = _DEFAULT_BACKUP
        for candidate in _BACKUP_PRIORITY:
            if any(candidate in name for name in clean_names):
                for name in clean_names:
                    if candidate in name:
                        backup = name
                        break
                break

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


# Resolve models once at import time
PROD_MODEL, BACKUP_MODEL = get_dynamic_models(key_manager.get_current_key())

# --------------------------------------------------------------------
# RETRY / ROTATION HELPER
# --------------------------------------------------------------------
# Max retries equals the number of keys in the pool (try each key once),
# with a floor of 3 attempts.
# --------------------------------------------------------------------

MAX_RETRIES = max(key_manager.key_count, 3)


def _is_quota_error(exc: Exception) -> bool:
    """Return True if the exception looks like a 429 / quota-exceeded error."""
    text = str(exc).lower()
    return any(
        marker in text
        for marker in ("429", "quota", "rate limit", "resource exhausted",
                       "resourceexhausted", "too many requests")
    )


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


# --------------------------------------------------------------------
# LLM CONFIGURATION
# --------------------------------------------------------------------
SMART_SUMMARY_MAX_TOKENS = int(os.getenv("SMART_SUMMARY_MAX_TOKENS", "3000"))
QA_MAX_TOKENS = int(os.getenv("QA_MAX_TOKENS", "3000"))
SMART_SUMMARY_CONTEXT_CHAR_LIMIT = int(os.getenv("SMART_SUMMARY_CONTEXT_CHAR_LIMIT", "12000"))

REFUSAL_STR = "Insufficient information in the provided context."
MISSING_INFO_STR = "Information not present in the retrieved Harrison excerpt."
SMART_SUMMARY_ACK = "Topic received — generating Harrison Smart Summary."

SMART_SUMMARY_SECTIONS = [
  "# 1. Harrison Definition (verbatim essence)",
  "# 2. High-Yield Overview (2–4 lines)",
  "# 3. Etiology / Causes (Harrison classification)",
  "# 4. Pathophysiology (with a compressed Harrison-style flowchart)",
  "# 5. Clinical Manifestations",
  "# 6. Diagnostic Criteria (with key Harrison lines emphasized)",
  "# 7. Investigations (important Harrison-points only)",
  "# 8. Severity Scoring Systems (Ranson, BISAP, APACHE II)",
  "# 9. Complications (local + systemic)",
  "# 10. Management (Acute phase → nutritional → complications)",
  "# 11. Important Harrison Tables (compress + reconstruct)",
  "# 12. Harrison Flowcharts (shortened but accurate)",
  "# 13. One-page exam-revision sheet at the end",
]


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
) -> tuple[str, bool, bool]:
    """
    Post-hoc verification step that checks the draft answer against
    Harrison context and rewrites unsupported statements while preserving
    explanations and detail whenever possible.

    Includes automatic API key rotation on 429 / Quota-Exceeded errors.

    Returns
    -------
    tuple[str, bool, bool]
        (verified_text, is_truncated, verification_ran)
        verification_ran is False when all retries were exhausted due to
        exceptions — the returned text is the original draft, not verified.
    """
    if model is None:
        model = PROD_MODEL

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
    for attempt in range(MAX_RETRIES):
        try:
            client = key_manager.next_client()   # round-robin advance per attempt
            resp = client.models.generate_content(
                model=model,
                contents=verify_user,
                config=types.GenerateContentConfig(
                    system_instruction=verify_prompt,
                    temperature=0.0,
                    max_output_tokens=max_tokens,
                )
            )

            verified, truncated = _extract_text_safely(resp)

            if not verified:
                return answer, truncated, True

            if truncated:
                log.warning(
                    "verify_answer: verifier output truncated (MAX_TOKENS) "
                    "on attempt %d/%d — returning partial verified text.",
                    attempt + 1,
                    MAX_RETRIES,
                )

            return verified, truncated, True

        except Exception as exc:
            last_exc = exc
            if _is_quota_error(exc) and attempt < MAX_RETRIES - 1:
                log.warning(
                    "verify_answer: quota error on attempt %d/%d — "
                    "marking key exhausted and rotating.",
                    attempt + 1,
                    MAX_RETRIES,
                )
                key_manager.mark_exhausted()
                continue
            # Non-quota error or final attempt — fall through
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
To combat summarization bias, you MUST obey these extraction rules if the data exists in the context:
- NUMERICAL GRANULARITY: You MUST explicitly state exact lab thresholds, fluid volumes, and diagnostic cutoffs. You must **bold** these numbers (e.g., "**pH < 7.30**", "**glucose > 250 mg/dL**"). Do NOT summarize them as "elevated" or "low".
- ETIOLOGY & MECHANISMS: Always name the primary triggers of a disease (e.g., gallstones, alcohol) and the exact cellular enzymes/pathways involved.
- SCORING SYSTEMS: List full criteria and point values for clinical scores (e.g., CURB-65, Ranson).
- PROTOCOLS: Provide exact drug dosages, fluid rates, and chronological treatment steps.
</clinical_rigor>

<clinical_granularity>
You are a rigorous clinical engine, not a summarizer. You MUST explicitly extract and preserve:
1. Clinical scoring systems (e.g., CURB-65, PORT/PSI, Ranson, APACHE II, Wells, CHADS2-VASc) — reproduce the FULL criteria with individual point values in a structured list. Never mention a scoring system without listing its components.
2. Exact numerical thresholds and criteria (e.g., specific pH levels, anion gap numbers, serum lipase cutoffs, BUN/creatinine ratios) — always state the exact number in **bold**. Never replace a number with qualitative language like "elevated" or "abnormal".
3. Primary etiologies and triggers (e.g., alcohol, gallstones for pancreatitis; S. pneumoniae for CAP) — enumerate ALL etiologies mentioned in the context, not just the top two.
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

<formatting_mode_{mode}>
IF MODE = qa:
Deliver a dense, textbook-style clinical explanation in 2-5 paragraphs. Use markdown headers for clinical topics ONLY (e.g., `### Pathophysiology`, `### Management`). Integrate all citations `[p:NNN]` naturally at the end of sentences. Do not use conversational filler.

IF MODE = smart_summary:
Generate an actionable, high-yield structured synthesis.
- The first line MUST be exactly: "Topic received — generating Harrison Smart Summary."
- Use `###` headings for major sections.
- Utilize bold text and bulleted lists heavily for readability.
- Close with a `### Quick Revision` block containing 3-5 absolute must-know facts.
</formatting_mode_{mode}>

<evidence_handling>
If an "Evidence from Harrison" section is provided, treat it as ground truth. Synthesize these facts directly into your response structure; do not just copy-paste the bullet list.
</evidence_handling>
"""


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
         "error_fallback"    -- all retries exhausted (API/quota failure)
    """
    import time
    if model is None:
        model = PROD_MODEL

    if not fused_context or len(fused_context.strip()) < 20:
        return REFUSAL_STR, "", False, "error_fallback"

    mode = (mode or "qa").strip().lower()
    generation_max_tokens = (
        SMART_SUMMARY_MAX_TOKENS
        if mode == "smart_summary"
        else QA_MAX_TOKENS
    )

    prompt_header = MASTER_MEDICAL_SYNTHESIS_PROMPT

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
    for attempt in range(MAX_RETRIES):
        try:
            client = key_manager.next_client()   # round-robin advance per attempt
            t_gen_start = time.perf_counter()
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction="Follow instructions strictly and never use knowledge outside the provided context.",
                    temperature=0.2,
                    max_output_tokens=generation_max_tokens,
                )
            )
            t_gen_end = time.perf_counter()
            if timings is not None:
                timings["draft_generation"] = timings.get("draft_generation", 0.0) + (t_gen_end - t_gen_start)

            content, draft_truncated = _extract_text_safely(response)

            if not content:
                return REFUSAL_STR, "", False, "error_fallback"

            if draft_truncated:
                log.warning(
                    "ask_llm: generation truncated (MAX_TOKENS) on attempt %d/%d — "
                    "appending truncation notice to partial answer.",
                    attempt + 1,
                    MAX_RETRIES,
                )
                content += (
                    "\n\n> ⚠️ *Answer truncated — token budget reached. "
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
            verified, verifier_truncated, verification_ran = verify_answer(draft_answer, fused_context, mode=mode, model=model)
            t_verify_end = time.perf_counter()
            if timings is not None:
                timings["verification"] = t_verify_end - t_verify_start
                timings["retry"] = 0.0

            # Retry verification once if truncated
            if verifier_truncated and verification_ran:
                log.info("ask_llm: verify_answer output truncated. Retrying verification once with compact prompt and larger budget.")
                t_retry_start = time.perf_counter()
                verified_retry, verifier_truncated_retry, retry_ran = verify_answer(
                    draft_answer,
                    fused_context,
                    mode=mode,
                    model=model,
                    compact_prompt=True,
                    max_output_tokens=8192,
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
                        + "\n\n> ⚠️ *Answer verification was incomplete due to length constraints.*"
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

        except Exception as exc:
            last_exc = exc
            if _is_quota_error(exc) and attempt < MAX_RETRIES - 1:
                log.warning(
                    "ask_llm: quota error on attempt %d/%d — "
                    "marking key exhausted and rotating.",
                    attempt + 1,
                    MAX_RETRIES,
                )
                key_manager.mark_exhausted()
                continue
            # Non-quota error or final attempt — fall through
            break

    log.error("ask_llm: all retries exhausted — returning safe error fallback. last_exc=%s", last_exc)
    return (
        "I'm unable to generate a complete answer at this time due to a temporary service issue. "
        "Please try again in a few moments. If the problem persists, try a more specific query."
    ), "", True, "error_fallback"
