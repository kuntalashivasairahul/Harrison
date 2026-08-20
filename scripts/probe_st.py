import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from backend.config import EMBEDDING_DIM, EMBEDDING_MODEL

print("Loading model...")
model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
print("Model loaded.")
emb = model.encode("test query")
print("Embedded shape:", emb.shape)
if emb.shape[0] != EMBEDDING_DIM:
    raise RuntimeError(f"Embedding dim {emb.shape[0]} does not match config dim {EMBEDDING_DIM}")
