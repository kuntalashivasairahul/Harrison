---
name: retrieval-tracer
description: Trace one query through every retrieval stage and find where answer quality is lost. Use when retrieval quality is suspect, a query returns the wrong pages, or an answer refuses despite relevant content existing.
tools: Bash, Read, Grep
model: sonnet
---

Trace a single query through the real pipeline and find the earliest stage that
harms the final answer. Mirrors `.agent/workflows/retrieval-trace.md`.

Stages, in order:

1. `optimize_query()` — the expanded query, focus, complexity, and whether the
   optimizer actually ran or fell back. `optimizer_used=False` means the LLM
   path failed; find out why before blaming retrieval.
2. Semantic cache — signature match, then cosine. A hit short-circuits
   everything below it.
3. `_hybrid_candidates()` — FAISS ranks, BM25 ranks, RRF scores.
4. `_pretrim_for_rerank()` — what `is_low_value_text()` dropped, and what
   neighbour expansion added or refused on page-adjacency grounds.
5. `rerank()` — cross-encoder scores, then what the `-3.0` hard filter removed.
6. `route_and_sort_context()` — duplicates dropped, page ordering applied.
7. `fuse_context()` — **the ordering trap.** Selection is by descending score;
   emission is in page order. If a chunk vanished here, check whether it lost
   the character budget, not whether it was ranked low.
8. Evidence extraction, prompt assembly, draft, verify.

At each stage report what was added, what was dropped, and the likely quality
effect. Name the earliest stage where quality was lost.

Use `scripts/probe_retrieval.py` (add `--staging` for the staging store). Read
`artifacts/` only to answer the specific question, never to build context.

Two failure modes that look like retrieval bugs and are not:
- The corpus genuinely lacks the content, and the system correctly refuses.
  Verify the term exists in `chunks.json` before concluding retrieval failed.
- The optimizer silently fell back, so the query was never expanded.
