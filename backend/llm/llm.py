# backend/llm/llm.py
# Smart Summary v1 – Gemini API Key Rotation & Dynamic Model Selection

import os
import logging
import threading
import google.generativeai as genai
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
# --------------------------------------------------------------------


class KeyManager:
    """Thread-safe Gemini API key rotation pool.

    Loads every environment variable matching ``GEMINI_API_KEY_*`` into an
    ordered list.  When a 429 / Quota-Exceeded error is caught, call
    ``rotate()`` to switch to the next key and reconfigure ``genai``.

    Usage
    -----
    >>> key_manager = KeyManager()
    >>> key_manager.configure_current()        # call once at startup
    >>> key_manager.rotate()                   # call on 429 errors
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._keys: list[str] = []
        self._current_idx: int = 0

        # Collect numbered keys: GEMINI_API_KEY_1, GEMINI_API_KEY_2, …
        numbered: dict[int, str] = {}
        for var, val in os.environ.items():
            if var.startswith("GEMINI_API_KEY_") and val.strip():
                suffix = var.replace("GEMINI_API_KEY_", "")
                try:
                    numbered[int(suffix)] = val.strip()
                except ValueError:
                    # Non-numeric suffix — skip
                    continue

        if numbered:
            # Sort by suffix number so key ordering is deterministic
            self._keys = [numbered[k] for k in sorted(numbered)]
            log.info(
                "KeyManager: loaded %d API key(s) from GEMINI_API_KEY_* vars.",
                len(self._keys),
            )
        else:
            # Fallback: single GEMINI_API_KEY
            single = os.getenv("GEMINI_API_KEY", "").strip()
            if single:
                self._keys = [single]
                log.info(
                    "KeyManager: no numbered keys found — using single GEMINI_API_KEY."
                )
            else:
                log.warning(
                    "KeyManager: NO Gemini API keys found in environment!"
                )

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
        if not self._keys:
            return None
        return self._keys[self._current_idx]

    def configure_current(self) -> None:
        """Configure ``genai`` with the current active key."""
        key = self.get_current_key()
        if key:
            genai.configure(api_key=key)

    def rotate(self) -> str | None:
        """Advance to the next key, reconfigure ``genai``, return new key.

        Thread-safe.  Wraps around to the first key after exhausting the
        pool.  Returns ``None`` if the pool is empty.
        """
        if not self._keys:
            return None
        with self._lock:
            self._current_idx = (self._current_idx + 1) % len(self._keys)
            key = self._keys[self._current_idx]
            genai.configure(api_key=key)
            log.info(
                "KeyManager: rotated to key #%d/%d.",
                self._current_idx + 1,
                len(self._keys),
            )
            return key


# Global singleton — created once at module import time.
key_manager = KeyManager()
key_manager.configure_current()


# --------------------------------------------------------------------
# DYNAMIC MODEL DISCOVERY
# --------------------------------------------------------------------
# Queries Google's live model list via genai.list_models() and selects
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
        If provided, temporarily configures genai with this key for the
        list_models() call.  Otherwise uses the currently configured key.

    Returns
    -------
    tuple[str, str]
        ``(prod_model, backup_model)`` — full model names suitable for
        ``genai.GenerativeModel(model_name=...)``.
    """
    try:
        if api_key:
            genai.configure(api_key=api_key)

        available: list[str] = []
        for model in genai.list_models():
            # Only consider models that support generateContent
            if "generateContent" in (model.supported_generation_methods or []):
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

# Re-configure genai to the current key (list_models may have changed it)
key_manager.configure_current()

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


# --------------------------------------------------------------------
# LLM CONFIGURATION
# --------------------------------------------------------------------
SMART_SUMMARY_MAX_TOKENS = int(os.getenv("SMART_SUMMARY_MAX_TOKENS", "1500"))
QA_MAX_TOKENS = int(os.getenv("QA_MAX_TOKENS", "1500"))
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

def verify_answer(answer: str, context: str, mode: str = "qa", model: str | None = None) -> str:
    """
    Post-hoc verification step that checks the draft answer against
    Harrison context and rewrites unsupported statements while preserving
    explanations and detail whenever possible.

    Includes automatic API key rotation on 429 / Quota-Exceeded errors.
    """
    if model is None:
        model = PROD_MODEL

    answer = (answer or "").strip()
    context = (context or "").strip()

    if not answer or not context:
        return answer

    max_tokens = SMART_SUMMARY_MAX_TOKENS if mode == "smart_summary" else QA_MAX_TOKENS

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
            key_manager.configure_current()
            gemini_model = genai.GenerativeModel(
                model_name=model,
                system_instruction=verify_prompt,  # full task instructions (was a brief one-liner)
            )
            resp = gemini_model.generate_content(
                verify_user,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0,
                    max_output_tokens=max_tokens,  # mode-aware limit (was hardcoded 1024)
                )
            )

            verified = resp.text

            if not verified or not isinstance(verified, str):
                return answer

            return verified.strip()

        except Exception as exc:
            last_exc = exc
            if _is_quota_error(exc) and attempt < MAX_RETRIES - 1:
                log.warning(
                    "verify_answer: 429/quota error on attempt %d/%d — rotating key.",
                    attempt + 1,
                    MAX_RETRIES,
                )
                key_manager.rotate()
                continue
            # Non-quota error or final attempt — fall through
            break

    log.warning("verify_answer: all retries exhausted (%s) — returning draft.", last_exc)
    return answer


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
1. Clinical scoring systems (e.g., CURB-65, PORT/PSI, Ranson, APACHE II, Wells, CHADS₂-VASc) — reproduce the FULL criteria with individual point values in a structured list. Never mention a scoring system without listing its components.
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
):
    """Generate a Harrison-grounded answer with automatic API key rotation.

    On 429 / Quota-Exceeded errors, the function rotates to the next API
    key in the pool and retries seamlessly.  The caller never sees a quota
    error unless ALL keys are exhausted.
    """
    if model is None:
        model = PROD_MODEL

    if not fused_context or len(fused_context.strip()) < 20:
        return REFUSAL_STR

    mode = (mode or "qa").strip().lower()

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
            key_manager.configure_current()
            gemini_model = genai.GenerativeModel(
                model_name=model,
                system_instruction="Follow instructions strictly and never use knowledge outside the provided context."
            )
            response = gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1 if mode == "smart_summary" else 0.2,
                    max_output_tokens=1024,
                )
            )

            content = response.text

            if not content or not isinstance(content, str):
                return REFUSAL_STR

            draft_answer = content.strip()

            if mode == "smart_summary":
                draft_answer = _enforce_smart_summary_shape(draft_answer)

            verified = verify_answer(draft_answer, fused_context, mode=mode, model=model)

            if mode == "smart_summary":
                return _enforce_smart_summary_shape(verified)

            return verified.strip()

        except Exception as exc:
            last_exc = exc
            if _is_quota_error(exc) and attempt < MAX_RETRIES - 1:
                log.warning(
                    "ask_llm: 429/quota error on attempt %d/%d — rotating key.",
                    attempt + 1,
                    MAX_RETRIES,
                )
                key_manager.rotate()
                continue
            # Non-quota error or final attempt — fall through
            break

    return f"LLM call failed: {str(last_exc)}"