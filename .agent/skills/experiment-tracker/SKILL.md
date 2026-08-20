---
name: experiment-tracker
description: Use when running ML or RAG experiments that change prompts, retrieval settings, chunking, reranking, embeddings, or evaluation configuration.
---

Use this skill to keep experiment work structured and comparable.

When relevant:
- Record the hypothesis.
- Record exactly what changed.
- Record dataset or benchmark scope.
- Record metrics, failures, and unexpected side effects.
- Summarize whether the change should be kept, reverted, or retested.

Preferred output structure:
1. Experiment goal
2. Changes made
3. Evaluation scope
4. Results
5. Interpretation
6. Next step

Rules:
- Be concrete, not vague.
- Separate observed results from opinions.
- Prefer small, reversible experiments.
