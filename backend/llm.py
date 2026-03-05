# backend/llm.py
# Smart Summary v1 – Groq / LLaMA SAFE (generation-first)

import os
import re
from groq import Groq
from dotenv import load_dotenv
from pathlib import Path

# --------------------------------------------------------------------
# ENV LOADING (FIXED FOR YOUR STRUCTURE)
# --------------------------------------------------------------------
load_dotenv(Path(__file__).resolve().parent / ".env")

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
# PROMPTS
# --------------------------------------------------------------------

BASE_QA_PROMPT = """You are HarrisonGPT, a medical reasoning assistant.

Use ONLY the provided Harrison’s Principles of Internal Medicine context.
Do NOT add information not present in the context.

Guidelines:
- Answer in a clinical, textbook-oriented style.
- Structure the response with clear headings where appropriate.
- Focus on high-yield, exam-relevant content.
- If the answer cannot be derived from the provided context, say:
  "Insufficient information in the provided context."
"""

SMART_SUMMARY_PROMPT = """You are HarrisonGPT, a medical reasoning assistant designed to generate
high-fidelity, exam-relevant summaries extracted ONLY from the provided
Harrison’s Principles of Internal Medicine context.

Your task: Convert the provided Harrison material into a high-density,
exam-focused smart summary that preserves the exact essence, terminology,
clinical emphasis, and Harrison-like structure while removing non-essential narrative.

STRICT RULES (Extremely Important):
1. Use ONLY the retrieved Harrison text in Context.
2. Do NOT use outside knowledge or fabricate missing details.
3. Preserve Harrison-level accuracy, definitions, classifications, mechanisms,
   clinical features, diagnostic criteria, investigations, complications, and management.
4. Remove low-yield/redundant narrative (history/repetition/long research detail)
   while preserving high-yield exam content.
5. If a section lacks enough evidence in context, write exactly:
   "Information not present in the retrieved Harrison excerpt."
6. First line must be exactly:
   "Topic received — generating Harrison Smart Summary."

FINAL OUTPUT FORMAT (Use This Exactly):
# 1. Harrison Definition (verbatim essence)
# 2. High-Yield Overview (2–4 lines)
# 3. Etiology / Causes (Harrison classification)
# 4. Pathophysiology (with a compressed Harrison-style flowchart)
# 5. Clinical Manifestations
# 6. Diagnostic Criteria (with key Harrison lines emphasized)
# 7. Investigations (important Harrison-points only)
# 8. Severity Scoring Systems (Ranson, BISAP, APACHE II)
# 9. Complications (local + systemic)
# 10. Management (Acute phase → nutritional → complications)
# 11. Important Harrison Tables (compress + reconstruct)
# 12. Harrison Flowcharts (shortened but accurate)
# 13. One-page exam-revision sheet at the end

No extra sections. Keep medical language precise and Harrison-consistent.
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
    mode: str = "qa",  # "qa" or "smart_summary"
    model: str = "llama-3.3-70b-versatile",
):
    """
    fused_context: text fused from retrieved Harrison chunks
    question: user query / topic
    mode: "qa" (default) or "smart_summary"
    """

    if not fused_context or len(fused_context.strip()) < 20:
        return REFUSAL_STR

    mode = (mode or "qa").strip().lower()

    prompt_header = SMART_SUMMARY_PROMPT if mode == "smart_summary" else BASE_QA_PROMPT

    if mode == "smart_summary" and SMART_SUMMARY_CONTEXT_CHAR_LIMIT > 0:
        fused_context = fused_context[:SMART_SUMMARY_CONTEXT_CHAR_LIMIT]

    prompt = (
        prompt_header
        + "\n\nContext:\n"
        + fused_context
        + "\n\nTopic / Question:\n"
        + question
        + ("\n\nHarrison Smart Summary:\n" if mode == "smart_summary" else "\n\nAnswer:\n")
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Follow instructions strictly and never use knowledge outside the provided context."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1 if mode == "smart_summary" else 0.2,
            max_tokens=SMART_SUMMARY_MAX_TOKENS if mode == "smart_summary" else QA_MAX_TOKENS,
        )

        content = response.choices[0].message.content

        if not content or not isinstance(content, str):
            return REFUSAL_STR

        if mode == "smart_summary":
                        return _enforce_smart_summary_shape(content)

        return content.strip()

    except Exception as e:
        return f"LLM call failed: {str(e)}"
