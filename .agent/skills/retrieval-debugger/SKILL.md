---
name: retrieval-debugger
description: Use when RAG answers are irrelevant, incomplete, hallucinated, or likely caused by poor retrieval, chunking, filtering, or ranking.
---

Use this skill when debugging retrieval quality in RAG pipelines.

Inspect:
- chunk size and overlap
- embedding model choice
- top-k selection
- metadata filters
- duplicate chunks
- missing source attribution

Output:
- likely retrieval failure mode
- top suspected causes
- smallest high-impact fixes
