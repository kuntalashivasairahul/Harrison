import json
import faiss
import numpy as np
import sys
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.config import EMBEDDING_DIM, EMBEDDING_MODEL

CHUNKS_PATH = _ROOT / "artifacts" / "vectorstore" / "chunks.json"
INDEX_PATH = _ROOT / "artifacts" / "vectorstore" / "index.faiss"

print("Loading chunks...")
with open(CHUNKS_PATH, encoding="utf-8") as f:
    chunks = json.load(f)
print("Loaded", len(chunks))

print("Loading index...")
index = faiss.read_index(str(INDEX_PATH))
print("Index loaded. dim=", index.d)

print("Building BM25...")
tokenized = [c.get("text", "").lower().split() for c in chunks[:1000]]
bm25 = BM25Okapi(tokenized)
print("BM25 built.")

print(f"Loading embedding model {EMBEDDING_MODEL!r}...")
model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
if hasattr(model, "get_embedding_dimension"):
    model_dim = int(model.get_embedding_dimension())
else:
    model_dim = int(model.get_sentence_embedding_dimension())
print("Embedding model loaded. dim=", model_dim)
assert model_dim == EMBEDDING_DIM, f"Expected model dim {EMBEDDING_DIM}, got {model_dim}"
assert index.d == model_dim, f"Expected FAISS dim {index.d} to match embedding dim {model_dim}"

q = "diabetic ketoacidosis"
print("Encoding query...")
q_emb = model.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
assert q_emb.shape == (1, EMBEDDING_DIM), f"Expected query shape (1, {EMBEDDING_DIM}), got {q_emb.shape}"
print("Query encoded. Searching index...")
distances, ids = index.search(q_emb, 20)
print("Search done. retrieved IDs:", ids[0])
