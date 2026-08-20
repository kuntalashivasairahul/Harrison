# Quality Scorecard

Assessment date: 2026-07-24

Scores reflect the checked-in implementation, focused runtime probes, and the
automated test suite. They are not a claim that live Gemini answer quality has
been clinically validated.

| Aspect | Initial | Current | Evidence and remaining limitation |
| --- | ---: | ---: | --- |
| Architecture and boundaries | 82 | 89 | Core modules have clear ownership and cache metadata is explicit. Import-time model discovery remains a startup concern. |
| Retrieval and index compatibility | 55 | 90 | BGE-M3/1024 matches the production FAISS index; deterministic expansion and positive-score BM25 filtering are covered. Retrieval quality still needs a production-faithful benchmark. |
| LLM generation and verification | 76 | 87 | Gemini SDK migration, return paths, truncation handling, and token limits are unit-tested. No live model acceptance run was performed. |
| API, cache, and response safety | 70 | 88 | Cache keys include mode, verifier state, parameters, and vectorstore fingerprint; unsafe response paths are not saved. |
| Medical grounding and confidence | 79 | 88 | Confidence is conservative for missing rerank scores and fallback paths are capped. Thresholds remain empirical rather than clinically calibrated. |
| Observability and operations | 64 | 84 | Health checks validate embedding/index dimension; request logs distinguish optimizer and verifier state. Promotion is still not atomic across index and chunks. |
| Scripts and evaluation | 61 | 84 | Diagnostics use the live embedding model and the evaluator uses google-genai. Live evaluator execution requires a running API and Gemini credentials. |
| Documentation and governance | 58 | 88 | Runtime-facing docs now describe Gemini, BGE-M3/1024, -3.0 threshold, .venv312, and smart-summary caps. Historical audit reports remain intentionally historical. |
| Test coverage and regression protection | 68 | 91 | 97 unit tests plus 3 truncation tests pass; focused coverage was added for cache safety, retrieval determinism, quota rotation, fusion budgets, and token limits. |

Average score: 87.7 / 100

## Iteration History

1. Baseline: 80 / 100. Major defects included a live embedding/index dimension
   mismatch, stale Gemini evaluator SDK usage, cache under-keying, and an
   unenforced fusion budget.
2. Improvement plan: repair hard runtime failures first, then make retrieval
   behavior deterministic, make configuration effective, improve operational
   signals, and protect each behavior with no-network tests.
3. Current result: 87.7 / 100. The target average of 85 / 100 is met.

## Next Improvement Queue

1. Make index/chunk promotion atomic with validation and rollback.
2. Add a production-faithful retrieval benchmark that records each pipeline
   stage instead of only dense-index comparisons.
3. Move Gemini model discovery out of import time and expose the selected
   models through health/startup diagnostics.
4. Add a page-image smoke test after promotion to detect resolver offset drift.
5. Run a controlled live evaluation set with clinician-reviewed expectations.
