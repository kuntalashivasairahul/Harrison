---
name: prompt-tester
description: Use when prompt wording, system instructions, verification prompts, or answer formatting rules may affect response quality, faithfulness, completeness, or verbosity.
---

Use this skill when prompts are part of the failure surface.

Check for:
- ambiguity
- conflicting instructions
- over-constraint
- truncation risk
- formatting brittleness
- hidden assumptions

For RAG prompts, inspect:
- grounding instructions
- citation/source usage
- abstention behavior
- answer completeness
- verbosity control
- verification-step behavior

Output:
1. Prompt risks
2. Likely failure modes
3. Smallest prompt changes worth testing
4. Suggested A/B test plan
