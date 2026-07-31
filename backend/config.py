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
DEFAULT_FINAL_K = 6
DEFAULT_RERANK_POOL = 24
RRF_K = 60
RERANK_SCORE_THRESHOLD = -3.0
