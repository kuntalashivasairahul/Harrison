---
name: log-analyzer
description: Use when logs, stack traces, command output, or evaluation traces need to be turned into likely root causes and next debugging actions.
---

Use this skill when debugging from logs rather than from direct code inspection.

Process:
1. Identify the first meaningful failure signal.
2. Distinguish root cause from downstream noise.
3. Group findings into:
   - confirmed evidence
   - likely causes
   - unknowns
4. Recommend the smallest next debugging step.

Rules:
- Do not over-claim certainty.
- Quote the exact error or symptom that matters most.
- Prefer one strong next step over many weak guesses.
