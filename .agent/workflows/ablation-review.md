---
description: Compare before-vs-after behavior for a RAG change and determine whether the change is worth keeping
---

When invoked:
1. Compare baseline and modified system behavior.
2. Evaluate differences in:
   - retrieval quality
   - answer quality
   - latency
   - cost
   - failure patterns
3. Separate measured improvements from subjective impressions.
4. Conclude:
   - keep
   - revert
   - retest with smaller change
