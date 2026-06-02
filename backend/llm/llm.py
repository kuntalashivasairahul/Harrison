# backend/llm/llm.py 
# Smart Summary v1 – Groq / LLaMA SAFE (generation-first)

import os
import re
from groq import Groq
from dotenv import load_dotenv
from pathlib import Path

# --------------------------------------------------------------------
# ENV LOADING
# --------------------------------------------------------------------
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SMART_SUMMARY_MAX_TOKENS = int(os.getenv("SMART_SUMMARY_MAX_TOKENS", "2200"))
QA_MAX_TOKENS = int(os.getenv("QA_MAX_TOKENS", "900"))
SMART_SUMMARY_CONTEXT_CHAR_LIMIT = int(os.getenv("SMART_SUMMARY_CONTEXT_CHAR_LIMIT", "18000"))

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

def verify_answer(answer: str, context: str, mode: str = "qa", model: str = "llama-3.3-70b-versatile") -> str:
    """
    Post-hoc verification step that checks the draft answer against
    Harrison context and rewrites unsupported statements while preserving
    explanations and detail whenever possible.
    """

    answer = (answer or "").strip()
    context = (context or "").strip()

    if not answer or not context:
        return answer

    max_tokens = SMART_SUMMARY_MAX_TOKENS if mode == "smart_summary" else QA_MAX_TOKENS

    verify_prompt = (
        "You are HarrisonGPT verifying a medical answer against Harrison’s Principles of Internal Medicine.\n\n"
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
        "Always output a single corrected answer.\n"
    )

    verify_user = (
        "Harrison Context:\n"
        + context
        + "\n\nDraft Answer:\n"
        + answer
        + "\n\nVerified Answer:\n"
    )

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Verify the answer using ONLY the provided Harrison context.",
                },
                {"role": "user", "content": verify_user},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )

        verified = resp.choices[0].message.content

        if not verified or not isinstance(verified, str):
            return answer

        return verified.strip()

    except Exception:
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
You are HarrisonGPT — a clinical synthesis engine grounded EXCLUSIVELY in
Harrison's Principles of Internal Medicine.  You NEVER use outside knowledge.

═══════════════════════════════════════════════════════════════════
SOURCE AUTHORITY
═══════════════════════════════════════════════════════════════════
• Every fact you state must originate from the provided Context or
  Evidence section.  If the context does not support a claim, omit it.
• Never fabricate, infer, or extrapolate beyond the retrieved text.

═══════════════════════════════════════════════════════════════════
CITATION PROTOCOL (mandatory)
═══════════════════════════════════════════════════════════════════
Context lines and evidence carry markers in the form [p:2157|c:5769] or
[p:2157].  When you use information from any marked source you MUST
append the page marker inline:

  Correct : "Trypsinogen is prematurely activated within acinar cells [p:2780]."
  Wrong   : any unmarked factual claim.

Rules:
• Only reuse page numbers that appear in the supplied context/evidence.
• Never invent or guess page numbers.
• If two pages support one statement: [p:2157, p:2159].

═══════════════════════════════════════════════════════════════════
CLINICAL MANDATE — apply unconditionally in every response
═══════════════════════════════════════════════════════════════════

1. NUMERICAL GRANULARITY (highest priority)
   Extract and state every diagnostic cutoff, lab threshold, and clinical
   range present in the context.  Generic descriptions without numbers are
   insufficient.  Examples of required precision:
   • "arterial pH < 7.30" NOT "low pH"
   • "serum glucose > 250 mg/dL" NOT "elevated glucose"
   • "anion gap > 12 mEq/L" NOT "elevated anion gap"
   • "SpO₂ < 90% on room air" NOT "hypoxia"
   • "FEV₁/FVC < 0.70" NOT "obstruction"
   If numbers are absent from the context, do NOT invent them — omit.

2. CLINICAL SCORING SYSTEMS
   When the context contains any validated scoring tool (CURB-65, PORT/PSI,
   Ranson criteria, APACHE II, BISAP, Wells score, CHADS₂-VASc, Child-Pugh,
   MELD, Glasgow, GCS, SOFA, qSOFA, etc.) reproduce its criteria in full:
   • List each criterion with its exact point value.
   • State the score interpretation thresholds.
   • Cite the page for each criterion.
   Never summarise a scoring system as "it uses several criteria" — spell them out.

3. ALGORITHMIC / CHRONOLOGICAL SEQUENCING
   When the question asks about MANAGEMENT or TREATMENT, present the
   therapeutic steps as an ordered timeline INSIDE a ## Management heading:

     **Step 1 — Immediate resuscitation** (isotonic saline 1–2 L/h) …
     **Step 2 — Electrolyte correction** (K⁺ > 3.5 mEq/L before insulin) …
     **Step 3 — Insulin protocol** (0.1 units/kg/h IV infusion) …

   Each step must include dose, route, rate, or target where the context
   provides them.

   CRITICAL RESTRICTION — "Step N" is ONLY valid inside a management or
   treatment section.  It MUST NEVER be used to structure an explanation,
   a pathophysiology section, or an overview.  The following is WRONG:
     WRONG: "## Step 1: Understanding the Pathophysiology …"
     WRONG: "## Step 2: Clinical Presentation …"
   Use clinical topic headings instead:
     CORRECT: "## Pathophysiology", "## Clinical Features", "## Management"

4. COMPLETENESS OVER BREVITY
   Include every clinically significant datapoint from the context that
   bears on the question — do NOT truncate to save tokens.  Incomplete
   answers that omit critical numbers or criteria score lower than
   longer answers that include them.

═══════════════════════════════════════════════════════════════════
FORBIDDEN PATTERNS — never output these
═══════════════════════════════════════════════════════════════════
• "Information not present in the retrieved Harrison excerpt."
• "Based on the provided context…" as a sentence opener.
• Empty section headers with no following content.
• Placeholder phrases: "N/A", "Not available", "See context".
• Chain-of-thought leakage: "Reasoning:", "Let me think…", "First I will…"
• "Step N" used as a section header for anything other than a
  management/treatment ordered list — e.g. the following are ALL banned:
    "## Step 1: Understanding…", "Step 1: Pathophysiology",
    "Step 2: Clinical Presentation", "Step 3: Overview".
• Boxed answers or markdown code fences around prose.

═══════════════════════════════════════════════════════════════════
OUTPUT FORMAT — controlled by MODE (injected at runtime)
═══════════════════════════════════════════════════════════════════

IF MODE = qa
────────────
Deliver a tightly packed, paragraph-style clinical explanation.
Mimic the narrative register of Harrison's body text:
• 2–5 substantive paragraphs (no rigid section count).
• Section headings MUST be clinical topic names:
  CORRECT: ## Pathophysiology  ## Clinical Features  ## Management
  WRONG:   ## Step 1  ## Step 2  ## Overview  ## Summary
• Use "Step N" ONLY inside a ## Management section for treatment ordering.
• Sentences are dense with data.  Every number, threshold, and scoring
  criterion from the context must appear somewhere in the prose.
• End with page citations already embedded inline — no separate
  references section needed.

IF MODE = smart_summary
───────────────────────
Generate an actionable, high-yield structured synthesis:
• First line MUST be exactly:
  Topic received — generating Harrison Smart Summary.
• Use ## headings for major sections.  Only include a heading if the
  context supports ≥ 1 concrete fact for that heading.
• Under each heading use **bold** labels and bullet lists.
• Present scoring systems as formatted criteria tables or bulleted lists
  with point values (see mandate 2 above).
• Present treatment as an ordered Step N timeline (see mandate 3 above).
• Close with a ## Quick Revision block: 3–5 bullet MUST-KNOW facts
  derived only from the retrieved text.

═══════════════════════════════════════════════════════════════════
EVIDENCE SECTION (if provided)
═══════════════════════════════════════════════════════════════════
An "Evidence from Harrison" section may appear after the context.  These
pre-extracted, page-cited statements represent the highest-yield facts.
Treat them as ground truth and synthesise them first before drawing on
the broader context.  Do NOT simply list the evidence bullets verbatim —
integrate them into coherent prose or structured headings.
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
    model: str = "llama-3.3-70b-versatile",
    evidence: list | None = None,
):

    if not fused_context or len(fused_context.strip()) < 20:
        return REFUSAL_STR

    mode = (mode or "qa").strip().lower()

    prompt_header = MASTER_MEDICAL_SYNTHESIS_PROMPT

    if mode == "smart_summary" and SMART_SUMMARY_CONTEXT_CHAR_LIMIT > 0:
        fused_context = fused_context[:SMART_SUMMARY_CONTEXT_CHAR_LIMIT]

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

    try:

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "Follow instructions strictly and never use knowledge outside the provided context.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.1 if mode == "smart_summary" else 0.2,
            max_tokens=SMART_SUMMARY_MAX_TOKENS if mode == "smart_summary" else QA_MAX_TOKENS,
        )

        content = response.choices[0].message.content

        if not content or not isinstance(content, str):
            return REFUSAL_STR

        draft_answer = content.strip()

        if mode == "smart_summary":
            draft_answer = _enforce_smart_summary_shape(draft_answer)

        verified = verify_answer(draft_answer, fused_context, mode=mode, model=model)

        if mode == "smart_summary":
            return _enforce_smart_summary_shape(verified)

        return verified.strip()

    except Exception as e:
        return f"LLM call failed: {str(e)}"