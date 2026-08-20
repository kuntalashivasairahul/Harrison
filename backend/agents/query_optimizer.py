# backend/agents/query_optimizer.py
"""
QueryOptimizer Agent — Pre-Retrieval Stage
==========================================
Intercepts a raw user query and rewrites it into a structured search plan
optimised for FAISS semantic search over Harrison's Principles of Internal
Medicine.

Responsibilities
----------------
- Expand medical acronyms to full clinical terms.
- Add diagnostic/therapeutic framing to under-specified queries.
- Detect whether the query is medical in nature.
- Return a deterministic, machine-readable dict on every call.

Guarantees (CODING_RULES.md §1, §2, §3)
-----------------------------------------
- Pure fallback: if the LLM is unavailable or returns unparseable output,
  ``optimize_query`` always returns a safe, well-structured dict built from
  the original raw query.  The pipeline never crashes.
- No medical claims are invented — the agent only rewrites the *query*, not
  the *answer*.
- No imports from ``retrieval/``, ``llm/``, ``api/``, or ``rendering/``.
This module delegates optional LLM work to the approved stage router and
always retains a deterministic local fallback.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Environment & client setup
# ---------------------------------------------------------------------------

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from backend.config import LLM_OPTIMIZER_DEADLINE_SECONDS
from backend.llm.contracts import LLMRequest, LLMStage
from backend.llm.llm import llm_router

# The optimizer only needs a small JSON object, but the approved Groq model is
# a reasoning model: it spends tokens on an internal chain before emitting the
# answer.  At 256 the budget was consumed by reasoning and the response came
# back with finish_reason="length" and empty content on every clinical query.
_MAX_TOKENS = 512

_TEMPERATURE = 0.0  # Deterministic — query rewriting must be stable.

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------


class OptimizedQuery(TypedDict):
    """
    Structured output of the QueryOptimizer agent.

    Fields
    ------
    is_medical_query : bool
        True when the query is clearly within the medical/clinical domain.
        False for off-topic or administrative queries.
    expanded_query : str
        A rewritten, FAISS-ready query string.  Acronyms are expanded,
        clinical context (e.g. "diagnosis and treatment") is added when
        absent, and ambiguous terms are disambiguated.
    focus : str
        A concise label describing the primary clinical intent of the query.
        Examples: "diagnosis", "pathophysiology", "management",
        "pharmacology", "epidemiology", "prognosis".
    complexity : str
        Query complexity classification — one of:
        ``"simple"``  — single-fact lookup, isolated definition, or specific
                        drug dose.  Example: "What is the half-life of warfarin?"
        ``"complex"`` — multi-part question, combined pathophysiology +
                        management, or questions requiring diagnostic criteria
                        and scoring systems.
                        Example: "Pathophysiology and management of DKA"
        Used by the pipeline to set adaptive retrieval depth (final_k).
    original_query : str
        The unmodified raw query, preserved for logging and fallback use.
    optimizer_used : bool
        True when the LLM expansion succeeded; False when the fallback
        (rule-based) path was taken.
    """

    is_medical_query: bool
    expanded_query:   str
    focus:            str
    complexity:       str   # "simple" | "complex"
    original_query:   str
    optimizer_used:   bool


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a clinical query expansion agent for HarrisonGPT, a medical RAG system \
built on Harrison's Principles of Internal Medicine.

Your job is to rewrite a raw user query into a precise, clinically-grounded \
search query optimised for semantic search over a medical textbook.

Rules:
1. Expand ALL medical acronyms to their full clinical terms.
   Examples: "MI" → "myocardial infarction", "HF" → "heart failure",
   "T2DM" → "type 2 diabetes mellitus", "ARDS" → "acute respiratory \
distress syndrome".
2. If the query lacks clinical framing, add it.
   Examples: "chest pain" → "chest pain differential diagnosis and evaluation",
   "hypertension" → "hypertension pathophysiology diagnosis and management".
3. Identify the primary clinical focus from this list ONLY:
   diagnosis | pathophysiology | management | pharmacology | epidemiology | \
prognosis | definition | investigation | complication | other
4. Determine if the query is medical. Non-medical queries (e.g. "what is the \
capital of France") should have is_medical_query: false and an unchanged \
expanded_query.
5. Keep the expanded_query under 120 characters.
6. Classify the query complexity:
   "simple"  — single-concept lookup: one isolated fact, a drug dose,
               a specific lab value, or a brief definition.
               Examples: "What is the normal INR range?",
                         "Definition of bradycardia",
                         "Dose of amoxicillin for strep throat".
   "complex" — multi-part or multi-domain question: combines pathophysiology
               WITH management, requires diagnostic criteria AND scoring
               systems, or asks about complications AND treatment.
               Examples: "Pathophysiology and management of DKA",
                         "Diagnostic criteria and treatment of CAP",
                         "Mechanism and complications of acute pancreatitis".
   When in doubt, classify as "complex" to maximise retrieval depth.

You MUST respond with ONLY a valid JSON object. No prose, no markdown, no \
code fences. The JSON must have exactly these five keys:
{
  "is_medical_query": <bool>,
  "expanded_query": "<string>",
  "focus": "<string>",
  "complexity": "simple" | "complex"
}
"""

_USER_TEMPLATE = "Raw query: {raw_query}"

# ---------------------------------------------------------------------------
# JSON extraction helper
# ---------------------------------------------------------------------------

_JSON_BLOCK_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _extract_json(text: str) -> dict | None:
    """
    Attempt to parse the first JSON object found in ``text``.

    The LLM occasionally wraps output in markdown fences or adds preamble
    text despite explicit instructions.  This helper strips that noise.
    """
    text = text.strip()

    # Fast path: the whole string is valid JSON.
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Slow path: find the first {...} block.
    match = _JSON_BLOCK_RE.search(text)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return None


# ---------------------------------------------------------------------------
# Schema validator
# ---------------------------------------------------------------------------

def _validate_payload(payload: dict, raw_query: str) -> OptimizedQuery | None:
    """
    Validate that ``payload`` contains the expected keys with correct types.

    Returns None if validation fails so the caller can invoke the fallback.
    """
    is_medical  = payload.get("is_medical_query")
    expanded    = payload.get("expanded_query")
    focus       = payload.get("focus")
    complexity  = payload.get("complexity", "complex")  # default to complex if absent

    if not isinstance(is_medical, bool):
        log.debug("QueryOptimizer: is_medical_query missing or not bool.")
        return None
    if not isinstance(expanded, str) or not expanded.strip():
        log.debug("QueryOptimizer: expanded_query missing or empty.")
        return None
    if not isinstance(focus, str) or not focus.strip():
        log.debug("QueryOptimizer: focus missing or empty.")
        return None
    # Normalise complexity — any unrecognised value defaults to 'complex' for safety.
    if not isinstance(complexity, str) or complexity.strip().lower() not in ("simple", "complex"):
        log.debug(
            "QueryOptimizer: complexity %r not recognised — defaulting to 'complex'.",
            complexity,
        )
        complexity = "complex"
    else:
        complexity = complexity.strip().lower()

    return OptimizedQuery(
        is_medical_query=is_medical,
        expanded_query=expanded.strip(),
        focus=focus.strip().lower(),
        complexity=complexity,
        original_query=raw_query,
        optimizer_used=True,
    )


# ---------------------------------------------------------------------------
# Fallback builder
# ---------------------------------------------------------------------------

def _build_fallback(raw_query: str) -> OptimizedQuery:
    """
    Construct a safe, pass-through OptimizedQuery from the original query.

    Called whenever the LLM path fails for any reason.  The pipeline receives
    a well-typed dict and continues without interruption.

    ``complexity`` defaults to ``"complex"`` so that the retrieval layer always
    fetches the maximum number of chunks when the optimizer is unavailable —
    maximising recall at the cost of slightly higher latency.  This is the
    conservative, safe choice.
    """
    return OptimizedQuery(
        is_medical_query=True,       # Conservative: assume medical in medical app.
        expanded_query=raw_query,    # Pass-through: no expansion attempted.
        focus="other",               # Unknown focus — retrieval handles it.
        complexity="complex",        # Max recall when LLM is unavailable.
        original_query=raw_query,
        optimizer_used=False,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def optimize_query(raw_query: str) -> OptimizedQuery:
    """
    Expand and disambiguate a raw user query for FAISS semantic search.

    This is the single public entry-point for the QueryOptimizer agent.
    It is safe to call from any context — it never raises.

    Parameters
    ----------
    raw_query : str
        The unprocessed string exactly as typed by the user.

    Returns
    -------
    OptimizedQuery
        A TypedDict with the fields:
        - ``is_medical_query`` (bool)
        - ``expanded_query`` (str)  — FAISS-ready search string
        - ``focus`` (str)           — clinical intent label
        - ``original_query`` (str)  — unmodified input
        - ``optimizer_used`` (bool) — True if LLM expansion succeeded

    Behaviour on failure
    --------------------
    If the Groq call raises, times out, or returns unparseable output,
    the function logs the error at WARNING level and returns the original
    query verbatim via ``_build_fallback()``.  The pipeline is never
    interrupted.
    """
    raw_query = (raw_query or "").strip()
    if not raw_query:
        log.warning("QueryOptimizer: received empty query — returning fallback.")
        return _build_fallback("")

    try:
        result = llm_router.optimize(
            LLMRequest(
                prompt=_USER_TEMPLATE.format(raw_query=raw_query),
                system_instruction=_SYSTEM_PROMPT,
                model_alias="optimizer",
                temperature=_TEMPERATURE,
                max_output_tokens=_MAX_TOKENS,
                deadline_seconds=LLM_OPTIMIZER_DEADLINE_SECONDS,
                stage=LLMStage.OPTIMIZER,
            )
        )
        raw_content = result.text
        payload = _extract_json(raw_content)
        if payload is None:
            log.warning("QueryOptimizer: optimizer_failed=True fallback_to_original_query=True reason=unparseable_response provider=%s", result.provider)
            return _build_fallback(raw_query)
        validated = _validate_payload(payload, raw_query)
        if validated is None:
            log.warning("QueryOptimizer: optimizer_failed=True fallback_to_original_query=True reason=schema_validation_failed provider=%s", result.provider)
            return _build_fallback(raw_query)
        log.info("QueryOptimizer: optimizer_used=True provider=%s model=%s '%s' -> '%s' focus=%s complexity=%s medical=%s", result.provider, result.model, raw_query, validated["expanded_query"], validated["focus"], validated["complexity"], validated["is_medical_query"])
        return validated
    except Exception as exc:  # noqa: BLE001
        log.warning("QueryOptimizer: optimizer_failed=True fallback_to_original_query=True reason=exception exc_type=%s exc=%s", type(exc).__name__, exc)
        return _build_fallback(raw_query)
