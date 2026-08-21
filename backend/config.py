import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Artifact paths
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
VECTORSTORE_DIR = ARTIFACTS_DIR / "vectorstore"
LOG_DIR = ARTIFACTS_DIR / "retrieval_logs"

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
