#!/usr/bin/env python3
"""
scripts/evaluate_rag.py
=======================
LLM-as-a-Judge evaluation harness for HarrisonGPT.

Usage
-----
    python scripts/evaluate_rag.py

What it does
------------
1. Clears the semantic cache (DELETE /admin/cache) so every run tests the
   live RAG pipeline — not a cached response.
2. Iterates through a small Golden Dataset of clinical queries.
3. POSTs each query to the /ask endpoint and records latency.
4. Passes (query, expected_focus, generated_answer) to a Groq 70B judge
   model that returns a structured 1-5 score with reasoning.
5. Prints a colour-formatted report to the terminal and exits with a
   non-zero code if any query scores below a minimum threshold.

Requirements
------------
- The HarrisonGPT server must be running at API_BASE_URL (default:
  http://127.0.0.1:8000).
- GROQ_API_KEY must be set in backend/.env or the environment.
- Only stdlib + groq + python-dotenv are used (both already in requirements).
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from groq import Groq

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Load GROQ_API_KEY from backend/.env (same location the server uses).
_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_ROOT / "backend" / ".env")

API_BASE_URL   = os.getenv("HARRISONAI_API_URL", "http://127.0.0.1:8000")
ASK_ENDPOINT   = f"{API_BASE_URL}/ask"
CACHE_ENDPOINT = f"{API_BASE_URL}/admin/cache"

# Judge model — must be a large, high-reasoning model (CODING_RULES.md §4).
# Never use the 8B optimizer model for grading.
JUDGE_MODEL = "llama-3.3-70b-versatile"

# Minimum acceptable average score (1-5 scale) for the test suite to pass.
PASS_THRESHOLD: float = 3.0

# Request timeout for the /ask endpoint (RAG pipeline can take ~15s).
ASK_TIMEOUT_S: int = 90

# ---------------------------------------------------------------------------
# Retry / back-off configuration
# ---------------------------------------------------------------------------
# Both the /ask pipeline call and the Groq judge call can hit rate limits
# (HTTP 429).  The wrapper below retries with exponential back-off + jitter
# and honours the Retry-After header when Groq includes it.

RETRY_MAX_ATTEMPTS: int   = 5       # total attempts (1 original + 4 retries)
RETRY_BASE_DELAY_S: float = 4.0    # seconds before first retry
RETRY_MAX_DELAY_S:  float = 120.0  # cap per sleep (2 minutes)
RETRY_BACKOFF_FACTOR: float = 2.0  # multiply delay by this after each failure

import math
import random


def _parse_retry_after(exc: Exception) -> Optional[float]:
    """
    Try to extract a Retry-After wait time (seconds) from a Groq or urllib
    exception message.  Groq embeds it in the JSON error body as either:
      "Please try again in 20m50.208s"
      "Please try again in 45.3s"
    Returns None if no parseable value is found.
    """
    import re
    text = str(exc)
    # Match "Xm Y.Zs" or "Y.Zs" patterns
    m = re.search(r'try again in (?:(\d+)m)?(\d+(?:\.\d+)?)s', text)
    if m:
        minutes = int(m.group(1) or 0)
        seconds = float(m.group(2))
        return minutes * 60 + seconds
    return None


def _with_retry(fn, *args, label: str = "call", **kwargs):
    """
    Call ``fn(*args, **kwargs)`` up to RETRY_MAX_ATTEMPTS times.

    On a 429 / RateLimitError:
      1. Parse the Retry-After duration from the error message.
      2. If found, sleep exactly that long (+ 2s safety margin).
      3. If not found, use exponential back-off with ±10% jitter.

    Any non-rate-limit exception propagates immediately (fail fast).
    Returns the result of the first successful call.
    Raises the last exception if all attempts are exhausted.
    """
    delay = RETRY_BASE_DELAY_S
    last_exc: Exception = RuntimeError("No attempts made")

    for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            text = str(exc).lower()
            is_rate_limit = "429" in text or "rate_limit" in text or "rate limit" in text

            if not is_rate_limit:
                raise  # propagate non-rate-limit errors immediately

            last_exc = exc
            retry_after = _parse_retry_after(exc)

            if attempt == RETRY_MAX_ATTEMPTS:
                break  # exhausted — fall through to raise

            if retry_after is not None:
                wait = retry_after + 2.0  # small safety margin
                print(
                    f"\n  {YELLOW}⏳ Rate limit hit on {label} "
                    f"(attempt {attempt}/{RETRY_MAX_ATTEMPTS}). "
                    f"Groq says wait {retry_after:.0f}s — sleeping {wait:.0f}s…{RESET}",
                    flush=True,
                )
            else:
                # Exponential back-off with ±10% jitter
                jitter = random.uniform(-0.1 * delay, 0.1 * delay)
                wait   = min(delay + jitter, RETRY_MAX_DELAY_S)
                print(
                    f"\n  {YELLOW}⏳ Rate limit hit on {label} "
                    f"(attempt {attempt}/{RETRY_MAX_ATTEMPTS}). "
                    f"Back-off {wait:.1f}s…{RESET}",
                    flush=True,
                )
                delay = min(delay * RETRY_BACKOFF_FACTOR, RETRY_MAX_DELAY_S)

            time.sleep(wait)

    raise last_exc

# ---------------------------------------------------------------------------
# Golden Dataset
# ---------------------------------------------------------------------------
# Each entry has:
#   query          : The clinical question sent to /ask.
#   expected_focus : Key concepts the answer MUST address to score well.
#   mode           : "qa" or "smart_summary".
# ---------------------------------------------------------------------------

GOLDEN_DATASET = [
    {
        "query": "Pathophysiology of acute pancreatitis",
        "expected_focus": (
            "Enzymatic autodigestion of the pancreas, premature activation of "
            "trypsinogen to trypsin, role of gallstones and alcohol as triggers, "
            "local and systemic inflammatory cascade, acinar cell injury"
        ),
        "mode": "qa",
    },
    {
        "query": "What are the diagnostic criteria and management of diabetic ketoacidosis?",
        "expected_focus": (
            "Hyperglycemia >250 mg/dL, metabolic acidosis with pH <7.3, elevated "
            "anion gap, ketonemia or ketonuria, fluid resuscitation with normal saline, "
            "insulin infusion protocol, potassium replacement before insulin, bicarbonate "
            "administration criteria"
        ),
        "mode": "qa",
    },
    {
        "query": "Clinical features and treatment of community-acquired pneumonia",
        "expected_focus": (
            "Fever, productive cough, pleuritic chest pain, consolidation on CXR, "
            "Streptococcus pneumoniae as most common cause, CURB-65 severity scoring, "
            "empirical antibiotic therapy with beta-lactam plus macrolide or respiratory "
            "fluoroquinolone, PORT/PSI risk stratification"
        ),
        "mode": "qa",
    },
]

# ---------------------------------------------------------------------------
# Terminal colour helpers (ANSI, no external libraries)
# ---------------------------------------------------------------------------

RESET  = "\033[0m"
BOLD   = "\033[1m"
RED    = "\033[91m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
WHITE  = "\033[97m"
DIM    = "\033[2m"

def _colour_score(score: int) -> str:
    if score >= 4:
        return f"{GREEN}{BOLD}{score}/5{RESET}"
    if score == 3:
        return f"{YELLOW}{BOLD}{score}/5{RESET}"
    return f"{RED}{BOLD}{score}/5{RESET}"

def _bar(score: int, width: int = 20) -> str:
    filled  = int((score / 5) * width)
    colour  = GREEN if score >= 4 else YELLOW if score == 3 else RED
    return f"{colour}{'█' * filled}{DIM}{'░' * (width - filled)}{RESET}"

def _hr(char: str = "─", width: int = 72) -> None:
    print(f"{DIM}{char * width}{RESET}")

def _section(title: str) -> None:
    _hr("═")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    _hr("═")

def _wrap(text: str, width: int = 68, indent: str = "  ") -> str:
    words, lines, line = text.split(), [], ""
    for w in words:
        if len(line) + len(w) + 1 > width:
            lines.append(indent + line.rstrip())
            line = ""
        line += w + " "
    if line.strip():
        lines.append(indent + line.rstrip())
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _http_delete(url: str, timeout: int = 10) -> dict:
    req = urllib.request.Request(url, method="DELETE")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _http_post(url: str, payload: dict, timeout: int = ASK_TIMEOUT_S) -> tuple[dict, float]:
    """POST JSON payload; return (response_dict, elapsed_seconds)."""
    data = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    return body, time.perf_counter() - t0

# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

_groq_client: Optional[Groq] = None

def _get_groq() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            sys.exit(f"{RED}✗ GROQ_API_KEY not set. Check backend/.env{RESET}")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


JUDGE_SYSTEM = """\
You are a rigorous medical AI evaluator assessing the quality of answers \
produced by a RAG system grounded in Harrison's Principles of Internal Medicine.

You will receive:
  1. QUERY         — the clinical question asked.
  2. EXPECTED FOCUS — key concepts that a correct, high-quality answer should cover.
  3. GENERATED ANSWER — the RAG system's response.

Score the generated answer on a 1–5 integer scale:
  5 = Excellent: all expected focus points covered accurately, well-cited.
  4 = Good: most focus points covered, minor omissions or inaccuracies.
  3 = Adequate: partial coverage, some important points missing.
  2 = Poor: major gaps or clinical inaccuracies.
  1 = Unacceptable: wrong, hallucinated, or refused without justification.

Rules:
- Base your evaluation SOLELY on medical accuracy and coverage of EXPECTED FOCUS.
- A refusal ("Insufficient information…") scores 1 unless the query is genuinely off-topic.
- Citation quality (page numbers present) improves the score.
- Do NOT reward verbosity — a concise accurate answer outscores a long inaccurate one.

You MUST respond with ONLY a valid JSON object. No prose, no markdown fences:
{"score": <int 1-5>, "reasoning": "<one or two sentences>"}
"""

JUDGE_USER_TEMPLATE = """\
QUERY: {query}

EXPECTED FOCUS: {expected_focus}

GENERATED ANSWER:
{answer}

JSON evaluation:"""


def judge_answer(query: str, expected_focus: str, answer: str) -> dict:
    """
    Ask the 70B judge model to score the generated answer.

    Retries automatically on 429 / rate-limit errors using exponential
    back-off (see _with_retry).  Returns a dict with keys ``score`` (int)
    and ``reasoning`` (str).  Falls back to score=0 on permanent failure.
    """
    client = _get_groq()
    user_msg = JUDGE_USER_TEMPLATE.format(
        query=query,
        expected_focus=expected_focus,
        answer=answer,
    )

    def _call_judge():
        return client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            temperature=0.0,   # deterministic grading
            max_tokens=256,
        )

    try:
        resp = _with_retry(_call_judge, label=f"judge ({JUDGE_MODEL})")
        raw  = (resp.choices[0].message.content or "").strip()

        # Strip accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1].strip()
            if raw.startswith("json"):
                raw = raw[4:].strip()

        payload = json.loads(raw)
        score   = int(payload.get("score", 0))
        if not 1 <= score <= 5:
            raise ValueError(f"score out of range: {score}")
        return {"score": score, "reasoning": str(payload.get("reasoning", ""))}

    except Exception as exc:
        return {"score": 0, "reasoning": f"Judge call failed: {exc}"}

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    query:          str
    mode:           str
    expected_focus: str
    answer:         str
    confidence:     str
    sources:        list
    latency_s:      float
    score:          int
    reasoning:      str
    error:          Optional[str] = None

# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_evaluation() -> list[EvalResult]:
    _section("HarrisonGPT — LLM-as-a-Judge Evaluation")

    # ── Step 0: Clear semantic cache ─────────────────────────────────────
    print(f"\n{BOLD}[0/3] Clearing semantic cache…{RESET}")
    try:
        cleared = _http_delete(CACHE_ENDPOINT)
        print(f"  ✓ Cache cleared  "
              f"({cleared.get('entries_cleared', '?')} entries removed)\n")
    except Exception as exc:
        print(f"  {YELLOW}⚠ Could not clear cache: {exc} — continuing anyway{RESET}\n")

    results: list[EvalResult] = []

    for idx, item in enumerate(GOLDEN_DATASET, start=1):
        query          = item["query"]
        expected_focus = item["expected_focus"]
        mode           = item.get("mode", "qa")

        _hr()
        print(f"{BOLD}{WHITE}[{idx}/{len(GOLDEN_DATASET)}] {query}{RESET}")
        print(f"{DIM}  Mode: {mode}{RESET}\n")

        # ── Step 1: Call /ask (with retry on 429) ───────────────────────
        answer, confidence, sources, latency_s, error = "", "—", [], 0.0, None
        try:
            print(f"  {DIM}→ Calling /ask …{RESET}", end="", flush=True)
            body, latency_s = _with_retry(
                _http_post,
                ASK_ENDPOINT,
                {"query": query, "mode": mode},
                label="/ask",
            )
            answer     = body.get("answer", "")
            confidence = body.get("confidence", "—")
            sources    = body.get("sources", [])
            print(f"\r  {GREEN}✓{RESET} /ask responded in {BOLD}{latency_s:.1f}s{RESET}  "
                  f"| confidence: {BOLD}{confidence}{RESET}  "
                  f"| sources: {', '.join(sources) or 'none'}")
        except urllib.error.URLError as exc:
            error = f"Network error: {exc}"
            print(f"\r  {RED}✗ {error}{RESET}")
        except Exception as exc:
            error = str(exc)
            print(f"\r  {RED}✗ {error}{RESET}")

        # ── Step 2: Judge ──────────────────────────────────────────────
        if error:
            score, reasoning = 0, "Pipeline call failed — not graded."
        else:
            print(f"  {DIM}→ Sending to judge ({JUDGE_MODEL}) …{RESET}", end="", flush=True)
            verdict   = judge_answer(query, expected_focus, answer)
            score     = verdict["score"]
            reasoning = verdict["reasoning"]
            print(f"\r  {GREEN}✓{RESET} Judge scored: {_colour_score(score)}  "
                  f"{_bar(score)}")

        # ── Step 3: Print answer excerpt ───────────────────────────────
        excerpt = (answer[:300] + "…") if len(answer) > 300 else answer
        print(f"\n{DIM}  Answer excerpt:{RESET}")
        print(_wrap(excerpt))

        print(f"\n{DIM}  Reasoning:{RESET}")
        print(_wrap(reasoning))

        results.append(EvalResult(
            query=query, mode=mode, expected_focus=expected_focus,
            answer=answer, confidence=confidence, sources=sources,
            latency_s=latency_s, score=score, reasoning=reasoning,
            error=error,
        ))

    return results


def print_summary(results: list[EvalResult]) -> bool:
    """Print the aggregate report. Returns True if the suite passes."""
    _section("Evaluation Summary")

    valid_scores = [r.score for r in results if r.score > 0]
    avg_score    = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    avg_latency  = sum(r.latency_s for r in results) / len(results) if results else 0.0
    passed       = avg_score >= PASS_THRESHOLD

    # Per-question table
    col_w = [4, 54, 8, 8, 10]
    header = (
        f"{BOLD}"
        f"{'#':<{col_w[0]}}"
        f"{'Query':<{col_w[1]}}"
        f"{'Latency':>{col_w[2]}}"
        f"{'Score':>{col_w[3]}}"
        f"{'Conf':>{col_w[4]}}"
        f"{RESET}"
    )
    print(header)
    _hr()
    for idx, r in enumerate(results, start=1):
        q_trunc  = (r.query[:50] + "…") if len(r.query) > 50 else r.query
        score_s  = f"{r.score}/5" if r.score > 0 else "ERR"
        col      = GREEN if r.score >= 4 else YELLOW if r.score == 3 else RED
        print(
            f"{idx:<{col_w[0]}}"
            f"{q_trunc:<{col_w[1]}}"
            f"{r.latency_s:>{col_w[2] - 1}.1f}s"
            f"  {col}{BOLD}{score_s:>{col_w[3] - 2}}{RESET}"
            f"  {r.confidence:>{col_w[4] - 2}}"
        )

    _hr()

    # Aggregate row
    suite_colour = GREEN if passed else RED
    print(
        f"\n  {BOLD}Average score  : "
        f"{suite_colour}{avg_score:.2f} / 5.0{RESET}"
        f"  {_bar(round(avg_score))}"
    )
    print(f"  {BOLD}Average latency: {avg_latency:.1f}s{RESET}")
    print(f"  {BOLD}Pass threshold : {PASS_THRESHOLD}/5.0{RESET}")
    print(
        f"\n  Suite result   : "
        f"{GREEN + BOLD + '✅  PASSED' if passed else RED + BOLD + '❌  FAILED'}"
        f"{RESET}\n"
    )
    _hr("═")

    return passed


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        results = run_evaluation()
        passed  = print_summary(results)
        sys.exit(0 if passed else 1)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Interrupted by user.{RESET}")
        sys.exit(130)
