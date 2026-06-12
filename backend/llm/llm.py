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

def verify_answer(answer: str, context: str, mode: str = "qa", model: str = "llama-3.1-8b-instant") -> str:
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
    model: str = "llama-3.1-8b-instant",
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