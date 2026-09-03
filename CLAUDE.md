# HarrisonGPT — Claude Code Instructions

Medical RAG over *Harrison's Principles of Internal Medicine*. FastAPI + FAISS +
BM25 + cross-encoder rerank + Gemini draft/verify.

## Read these first — they are authoritative, this file is not

| File | What it governs | Status |
|---|---|---|
| `CODING_RULES.md` | **MANDATORY.** Layer boundaries, immutable pipeline steps, dependency governance, testing gates. | Read before any change. |
| `ARCHITECTURE.md` | Request flow, frozen `QueryResponse` contract, import-time purity. | Read before any refactor. |
| `README.md` | Setup, configuration, how to run. | |
| `workflow.md` | Step-by-step control-flow trace. | |
| `docs/archive/` | Historical analysis. **Stale — do not trust.** | Ignore. |

`CODING_RULES.md` is not advisory. It opens with "Violations in a medical AI
system are not style infractions — they are patient-safety risks." When this
file and `CODING_RULES.md` disagree, `CODING_RULES.md` wins.

## Non-negotiables (summary — the full list is RULE 3.2)

- Never invent a medical claim or a `[p:NNN]` citation. Insufficient context
  returns `REFUSAL_STR`, verbatim, unsoftened.
- Never bypass `verify_answer()`, `resolve_page_urls()`, or the
  `RERANK_SCORE_THRESHOLD` filter.
- Never hardcode a confidence value. It comes from `calculate_confidence()`.
- Never change `RRF_K` (60) or `RERANK_SCORE_THRESHOLD` (-3.0) without written
  justification plus evaluation results, updated in `ARCHITECTURE.md` **and**
  `CODING_RULES.md`.
- Keep the `QueryResponse` field set intact.

## Runtime

`.venv312` (Python 3.12) is the **only** supported runtime. There is no second
virtualenv; if you find one, it is a mistake.

```bash
./scripts/setup_env.sh                        # creates .venv312, installs runtime + dev deps
.venv312/bin/python -m pytest                 # 381 tests, ~5s, fully hermetic
.venv312/bin/python -m ruff check backend/ tests/
.venv312/bin/python -m uvicorn backend.api.main:app --reload --host 127.0.0.1 --port 8000
```

`git-lfs` is a prerequisite. Without it `artifacts/vectorstore/*` checks out as
pointer files and the API starts degraded.

## Testing rules that are easy to get wrong

- **The suite is hermetic by design** — no network, no model weights, no FAISS
  index. That is why it runs in seconds. Do not add a test that loads the real
  index or calls a live model to `tests/`; it belongs in a separate integration
  tier that does not yet exist.
- `tests/_api_harness.py` stubs the heavy modules so `backend.api.main` imports
  in ~0.2s instead of ~13s. Use it for anything touching the HTTP layer.
- A test that mutates `sys.modules` **must** restore it. One that did not
  silently poisoned the Google SDK for every test collected after it.
- Programs in `scripts/` are named `probe_*.py`, not `test_*.py`. They are
  interactive diagnostics that load real models; pytest must never collect them.

## Import-time purity — load-bearing, easy to regress

No module under `backend/` may do expensive or networked work at import.
The index, BM25 corpus, encoder, and Gemini model discovery all resolve on
first use and are forced deliberately in the FastAPI `lifespan` handler.

`configure_logging()` must be called **before** any other `backend.*` import in
an entry point, with one deliberate exception: `backend.config` is imported
first, because it loads `backend/.env` and `configure_logging()` reads
`HARRISON_LOG_LEVEL` from the environment. `backend.config` imports nothing
under `backend/` and logs nothing, so it cannot hide a diagnostic. Modules get
loggers with `logging.getLogger(__name__)` — never reach for `"uvicorn.error"`,
which re-hides the bug `backend/logging_config.py` exists to fix.

## Before you call a change done (RULE 5)

Changes to `rag.py`, `llm.py`, `main.py`, `confidence_scorer.py`, or
`rerank.py` require:

1. `pytest` green and `ruff` clean.
2. A live `/ask` call in **both** `smart_summary` and `qa` mode.
3. `/health` returning `status: "ok"` (it returns 503 when degraded).
4. For retrieval-parameter changes: the `evaluation/` harness, before and after.

## Do not read or index

`artifacts/`, `storage/`, `data/` — binary indexes, page renders, and the raw
textbook. Large, not source, and licensed content (RULE 3.1).

Exception: reading them to *measure* a specific defect is legitimate, but say so
and do not let them into general context.

## Gotchas that have already cost time

- **Gemini thinks by default** and those tokens come out of
  `max_output_tokens`. Thinking is off for the verifier and optimizer stages.
  The draft keeps it — but on 3.x only at `thinking_level=LOW`, because their
  default reasoning pass overran the qa ceiling. 2.5-flash drafts are still
  left alone. The knob is model-dependent (`thinking_budget` on 2.x,
  `thinking_level` on 3.x) and one model ignores it; the probe table is in
  `gemini_provider.py` — do not re-derive it.
- **Groq models get decommissioned.** The optimizer silently fell back for an
  unknown period after `llama-3.1-8b-instant` was retired. Check
  `client.models.list()` before assuming a model exists.
- `GROQ_OPTIMIZER_MODEL` in `backend/.env` **overrides** the registry, so
  changing `model_registry.json` alone may do nothing.
- The router clamps `max_output_tokens` with `min()` against the registry.
  Raising an env var past the registry cap is silently ineffective (it now logs).
- The corpus has gaps. "Ranson" appears in 1 chunk of 16,983 and it is not the
  criteria table, so the system correctly refuses a question the prompt template
  promises to answer.
  When the gap is partial the model does something worse than refuse: it
  declines in its own words and pads the refusal with true but tangential cited
  facts ("thyroid storm is not detailed in the provided chapters ... during
  subacute thyroiditis uptake is low [p:3074]"). That lands on the `verified`
  path, so no return-path cap fires. `answer_declines()` in
  `confidence_scorer.py` is the floor that catches it — citation presence alone
  does not, because the padded refusals carry citations.
- **Cached answers store the confidence they were labelled with.** Change the
  confidence rules and the entries already in `artifacts/semantic_cache.json`
  keep serving the old label; bump `CACHE_SCHEMA_VERSION` in `main.py` to retire
  them. Bumping now *deletes* them: `SemanticCache(schema_version=...)` drops
  entries whose `metadata.schema` differs at load and rewrites the file once.
  Before that they were loaded, scanned, and re-flushed forever — six of the
  twenty-one entries on disk were pre-v5.
- **Deadlines and cooldowns live in `backend/config.py`** and are imported by
  call sites. Do not reintroduce `os.getenv("LLM_..._DEADLINE_SECONDS")` at a
  call site — config and the live values silently drifted apart that way before.
  `backend/config.py` is also the **only** place `load_dotenv()` runs, above its
  own `os.getenv` block. It used to run in `llm.py` *after* that module had
  already imported config, so every `.env` deadline was read before the file
  was loaded and silently took its default. `tests/test_lazy_loading.py` fails
  if a second `load_dotenv()` reappears anywhere under `backend/`.
- The evidence block is built only from chunks the fused context could **not**
  carry. Building both from the same list sends every context chunk twice —
  that was 30% of the input budget.
- Draft/verifier failover exists (`router.generate_for_stage`). Passing an
  explicit `model=` to `ask_llm()` pins one deployment and bypasses it.
- **No model verifies its own draft.** `ask_llm()` passes the drafting
  `result.model` to `verify_answer(drafted_by_model=...)`, which reaches
  `generate_for_stage(exclude_model=...)`. The exclusion is per *model*, not per
  provider — barring the provider empties the verifier order whenever Mistral is
  dark, which loses verification entirely. It compares `_resolve_model()`, not
  `deployment.model`: the Gemini entries carry `dynamic-*` sentinels and a raw
  comparison silently never matches. CODING_RULES §6.1, amended 2026-08-27.
- **An outage cools the deployment; a Gemini 429 does not.** `_cooldown()`
  benches UNAVAILABLE/TIMEOUT for 15s so the verifier stage of the same request
  does not re-probe a model that just 503'd. Gemini quota stays uncooled because
  KeyManager rotates keys per project.
- Anything in `scripts/` is documented in `scripts/README.md`. Check there
  before assuming a script is dead.

## Deploy Configuration (configured by /setup-deploy)

- **Platform:** Hugging Face Space, Docker SDK, free CPU basic (2 vCPU, 16 GB
  RAM, 50 GB disk, no credit card). Chosen because the measured production
  footprint is **1.47 GB peak RSS** (FAISS 16,983 vectors + BM25 corpus +
  BGE-M3 + cross-encoder, 23 s warm-up). Render's free tier is 512 MB, Fly's
  free allowances are gone, and Vercel/Netlify have no long-lived process, so
  this is the only free tier that actually holds the app.
- **Production URL:** `https://<user>-harrisongpt.hf.space` (share this, not
  the `huggingface.co/spaces/...` page, which embeds the app in an iframe).
- **Deploy trigger:** `git push` to the Space remote. HF rebuilds the image.
- **Deploy status:** the Space build log, plus the health check below.
- **Post-deploy health check:** `GET /health`. Returns 503 when degraded, so it
  is a real check. `curl -sf https://<user>-harrisongpt.hf.space/health`
- **Merge method:** squash.
- **Project type:** FastAPI web app, server-rendered UI (Jinja2 + `/static`),
  no separate frontend build.

### Two repos, on purpose

| Repo | Visibility | Contents |
|---|---|---|
| `spaces/<user>/harrisongpt` | public | code only |
| `datasets/<user>/harrison-corpus` | **private** | `artifacts/vectorstore/{index.faiss,chunks.json}`, `storage/pages/small/*` |

`chunks.json` is ~33 MB of verbatim Harrison's prose and `storage/pages/` is
4.2 GB of scans of it. Both are licensed content under RULE 3.1, so neither may
sit in a public repo or a public image layer. The dataset mirrors the app's own
paths, so one `snapshot_download` lands every file where the code already looks
for it.

### Never add the Space as a git remote of this repo

The licensed files are **tracked, and they are in the commit history**:

```
$ git ls-files | grep -E '^(artifacts|data)/'
artifacts/vectorstore/chunks.json                   33 MB of verbatim textbook
artifacts/vectorstore/index.faiss
artifacts/vectorstore_backup/.../chunks.json
artifacts/vectorstore_staging/table_chunks.json
data/harrison.md                                    30 MB, the whole book
```

`git push space main` would publish all of it to a public Space, history
included. `.gitignore` does not untrack what is already tracked, and
`.dockerignore` governs the image, not the push — it gives no protection here
at all. A push you cannot recall is the failure mode this whole two-repo split
exists to prevent.

The Space gets its **own clean history** with only deploy files in it. Use
`scripts/sync_space.sh <space-checkout>`, which copies from an allowlist and
then verifies what actually landed: it aborts if `data/`, `storage/`,
`artifacts/` or `backend/.env` appear, or if any file exceeds 5 MB (a code-only
deploy has none; the largest legitimate file is vendored three.js at well under
1 MB). A correct sync is 56 files and about 1.2 MB.

### Boot order is load-bearing

`entrypoint.sh` runs `scripts/fetch_corpus.py` **before** uvicorn, because
`backend/api/main.py` mounts `StaticFiles("storage/pages")` at import time and
StaticFiles raises when the directory is missing. The FastAPI `lifespan`
handler runs too late to fix this, and import-time purity forbids the download
living under `backend/`. Do not move the fetch into the app.

### uvicorn flags that are not optional

`--proxy-headers --forwarded-allow-ips="*"`. `main.py` builds every page-image
URL from `str(request.base_url)`. Behind HF's HTTPS proxy, uvicorn honours
`X-Forwarded-Proto` only from IPs in `--forwarded-allow-ips` (default
`127.0.0.1`), which the platform proxy is not. Without the wildcard, `base_url`
resolves to `http://` on an `https://` page, every thumbnail is blocked as
mixed content, and the cited-pages rail silently renders broken images. The
wildcard is safe only because nothing but the platform proxy can reach the
container; narrow it for any deploy with direct public ingress.

### Space secrets (Settings > Variables and secrets)

`HF_TOKEN` (read-scoped, for the private dataset), `HARRISON_CORPUS_REPO`,
`GEMINI_API_KEY` plus `GEMINI_API_KEY_1..10`, optionally `GROQ_API_KEY` +
`GROQ_ENABLED=true`, and `HARRISON_ADMIN_TOKEN` (without it `main.py` warns and
disables `/admin/*`). Set `HARRISON_PAGE_FULL_RES=false`: the 3.8 GB full-res
renders are not deployed, and the flag makes `resolve_page_urls()` point
`full_url` at the thumbnail instead of a 404. All three `visual_context` keys
stay present, so the frozen `QueryResponse` shape is unchanged.

### Build on the M4 with `--platform linux/amd64`

`docker build --platform linux/amd64 -t harrisongpt .` Apple Silicon builds
arm64 by default; Spaces run x86_64, and `torch` and `faiss-cpu` ship different
wheels per architecture. An unqualified build produces an image that works
locally and can still fail on the Space. torch comes from
`download.pytorch.org/whl/cpu` deliberately: the default PyPI amd64 wheel
bundles a CUDA runtime this image will never use.

### Known free-tier behaviour, not bugs

- Spaces pause after 48 h idle. A cold wake pays the corpus pull plus the 23 s
  warm-up, so the first visitor waits roughly two minutes.
- Disk is ephemeral, so `artifacts/semantic_cache.json` starts empty on every
  wake. `CACHE_SCHEMA_VERSION` still matters locally; on the Space nothing
  survives to be retired.
- The rate limiter in `main.py` is in-process and assumes a single worker.
  Free tier runs one container, so that holds. It stops holding if you ever
  scale out.
