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
# --------------------------------------------------------------------

BASE_QA_PROMPT = """You are HarrisonGPT, a medical reasoning assistant.

Use ONLY the provided Harrison’s Principles of Internal Medicine context.
Do NOT add information not present in the context.

Evidence Usage Rules:

You will receive an "Evidence from Harrison" section.

Use these evidence statements to construct the answer.

Do NOT perform step-by-step reasoning.
Do NOT include phrases like:
"Step 1", "Step 2", "reasoning", or "final answer".

Instead, synthesize the evidence into a concise medical explanation,
similar to a paragraph from a medical textbook.

Write the final explanation directly.

You may also be given an "Evidence from Harrison" section that lists concise,
page-cited evidence statements derived from the context. Treat these as the
highest-yield, most reliable facts when constructing your answer.

When multiple evidence statements are relevant, synthesize them into a
coherent medical explanation rather than repeating them individually.

Use concise medical section headings such as "Mechanism of Action", "Pathophysiology", or "Clinical Relevance".

The context lines and evidence may include page and chunk markers such as:
- Some fact about disease X [p:2157|c:5769]
- EVIDENCE: Severe anemia reduces oxygen supply [p:2157]

When you use information from the context or evidence, you MUST:
- Cite the page numbers in square brackets using the same markers.
- Example: "Severe anemia can cause oxygen supply-demand imbalance [p:2157]."
- If multiple distinct pages support a statement, you may write: "[p:2157, p:2159]".
- Only reuse page markers that actually appear in the context or evidence.
- Never invent or guess new page numbers.

Guidelines:
-Write the answer as if it were a paragraph from Harrison's textbook.
- Do NOT show step-by-step reasoning.
- Do NOT include "Step 1", "Step 2", numbered reasoning, or chain-of-thought.
- Do NOT show derivations or thinking steps.
- Provide the final medical explanation directly.
- Use short paragraphs or section headings if useful.
- Focus on high-yield clinical facts.
"""

SMART_SUMMARY_PROMPT = """You are HarrisonGPT, a medical reasoning assistant designed to generate
high-fidelity, exam-relevant summaries extracted ONLY from the provided
Harrison’s Principles of Internal Medicine context.

You may also be given an "Evidence from Harrison" section that lists concise,
page-cited evidence statements derived from the context. Treat these as the
highest-yield, most reliable facts when constructing your summary.

When multiple evidence statements are relevant, synthesize them into a
clear Harrison-style explanation.

The context lines and evidence may include page and chunk markers such as:
- Some fact about disease X [p:2157|c:5769]
- EVIDENCE: Severe anemia reduces oxygen supply [p:2157]

When you use information from the context or evidence, you MUST:
- Cite the page numbers in square brackets using the same markers.
- Example: "Severe anemia can cause oxygen supply-demand imbalance [p:2157]."
- If multiple distinct pages support a statement, you may write: "[p:2157, p:2159]".
- Only reuse page markers that actually appear in the context or evidence.
- Never invent or guess new page numbers.

STRICT RULES:
1. Use ONLY the retrieved Harrison text in Context.
2. Do NOT use outside knowledge.
3. Preserve Harrison-level accuracy and terminology.
4. Remove low-yield narrative while preserving exam-relevant information.
5. If a section lacks evidence, write:
   "Information not present in the retrieved Harrison excerpt."
6. First line must be exactly:
   "Topic received — generating Harrison Smart Summary."
"""


def _enforce_smart_summary_shape(content: str) -> str:
    text = (content or "").strip()
    if not text:
        return text

    if SMART_SUMMARY_ACK not in text.splitlines()[0]:
        text = f"{SMART_SUMMARY_ACK}\n\n{text}"

    missing_sections = []
    for header in SMART_SUMMARY_SECTIONS:
        if re.search(rf"^{re.escape(header)}$", text, flags=re.MULTILINE) is None:
            missing_sections.append(header)

    if missing_sections:
        additions = "\n\n".join(f"{header}\n{MISSING_INFO_STR}" for header in missing_sections)
        text = f"{text}\n\n{additions}".strip()

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

    prompt_header = SMART_SUMMARY_PROMPT if mode == "smart_summary" else BASE_QA_PROMPT

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