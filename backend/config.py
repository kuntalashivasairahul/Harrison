import os
import time
from pathlib import Path

from dotenv import load_dotenv

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The one place backend/.env is loaded.  It used to happen in backend/llm/llm.py
# and backend/agents/query_optimizer.py, both of which sit *below* this module in
# the import graph -- llm.py imported backend.config on line 14 and only called
# load_dotenv on line 23, so any entry point whose first backend import was
# backend.llm.llm evaluated the os.getenv block below against an environment that
# had never seen the .env file.  Every LLM_*_SECONDS setting silently fell back to
# its default.  It went unnoticed because query_optimizer happened to load the file
# before importing this one, and the API entry point happened to import
# query_optimizer first.  Loading here makes the ordering a property of the import
# graph rather than a coincidence.  override=False, so a real environment variable
# still wins over the file.
load_dotenv(PROJECT_ROOT / "backend" / ".env")

# Artifact paths
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
VECTORSTORE_DIR = ARTIFACTS_DIR / "vectorstore"
LOG_DIR = ARTIFACTS_DIR / "retrieval_logs"

# Page renders.  storage/pages/full is 3.8 GB of full-resolution PNGs, which a
# free-tier deploy cannot carry -- a paused Space re-pulls its corpus on every
# wake, so shipping the full renders would mean a 4.3 GB cold start instead of
# a 0.5 GB one.  When they are absent the lightbox falls back to the WebP
# thumbnail rather than opening a 404.
#
# This lives here rather than in backend/rendering/ for two reasons: rendering/
# may import only `re` and `typing` (RULE 2.1), and config is the one module
# that runs load_dotenv().  api/main.py reads it and passes it down, which is
# also why there is no os.getenv() for it at a call site.
PAGE_FULL_RES_AVAILABLE = os.getenv("HARRISON_PAGE_FULL_RES", "true").strip().lower() not in (
    "0",
    "false",
    "no",
    "off",
)

# Models
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Retrieval parameters
DEFAULT_K = 30
DEFAULT_RERANK_POOL = 24
RRF_K = 60
RERANK_SCORE_THRESHOLD = -3.0

# LLM routing deadlines. Provider-specific capability limits live in
# backend/llm/model_registry.json; these bound an end-user request path.
#
# These are read from the environment HERE and imported by the call sites.
# They were previously plain literals that nothing imported, while each call
# site did its own os.getenv() with a hardcoded default — so editing this file
# changed nothing, and the two sources disagreed (60.0 here, "30" there).
LLM_OPTIMIZER_DEADLINE_SECONDS = float(os.getenv("LLM_OPTIMIZER_DEADLINE_SECONDS", "8"))
LLM_DRAFT_DEADLINE_SECONDS = float(os.getenv("LLM_DRAFT_DEADLINE_SECONDS", "60"))
LLM_VERIFIER_DEADLINE_SECONDS = float(os.getenv("LLM_VERIFIER_DEADLINE_SECONDS", "60"))
LLM_PROVIDER_COOLDOWN_SECONDS = float(os.getenv("LLM_PROVIDER_COOLDOWN_SECONDS", "60"))

# Wall-clock ceiling for one /ask request.  The per-stage deadlines above are
# independent and each stage retries, so without this a single request could
# legitimately run for minutes.  Stage deadlines clamp against what is left of
# this budget; retries stop when it is spent.  Set to 0 to disable.
LLM_TOTAL_REQUEST_BUDGET_SECONDS = float(
    os.getenv("LLM_TOTAL_REQUEST_BUDGET_SECONDS", "90")
)

# --------------------------------------------------------------------
# Static asset cache-busting.  Appended to every /static URL a template
# emits, so a deploy invalidates the browser cache without a filename hash
# or a build step.
#
# The per-boot fallback only changes when the process restarts, so an edited
# stylesheet keeps its cached URL and a browser reload will NOT pick it up.
# Restart uvicorn after a CSS edit; a stale site.css cost real debugging time
# during the UI build precisely because a reload looked like it should work.
#
# Pure computation at import: no I/O, no network (ARCHITECTURE §"import-time
# purity").  It reads an env var, which is why it lives here with the other
# os.getenv calls rather than at a call site.
# --------------------------------------------------------------------
ASSET_VERSION = os.getenv("HARRISON_ASSET_VERSION") or str(int(time.time()))
