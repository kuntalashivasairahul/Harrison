---
description: Trace a RAG query through every retrieval stage and identify where quality is lost
---

When invoked:
1. Follow one query through:
   - query rewrite or expansion
   - retrieval
   - fusion
   - reranking
   - filtering
   - context assembly
   - final answer generation
2. At each stage, report:
   - what was added
   - what was dropped
   - likely quality gains or losses
3. Identify the earliest stage where the final answer quality is harmed.
4. Recommend the smallest high-impact fix.
