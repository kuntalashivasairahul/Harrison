import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from backend.config import EMBEDDING_MODEL

device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Loading model on {device.upper()}...")
model = SentenceTransformer(EMBEDDING_MODEL, device=device)
print("Model loaded. Encoding...")
emb = model.encode("test query")
print("Encoded.")

print("Importing faiss...")
import faiss
print("Loading FAISS index...")
index = faiss.read_index("artifacts/vectorstore/index.faiss")
print("Index loaded successfully! Running search...")
if emb.shape[0] != index.d:
    raise RuntimeError(f"Embedding dim {emb.shape[0]} does not match FAISS dim {index.d}")
distances, ids = index.search(emb.reshape(1, -1), 5)
print("Search successful, IDs:", ids)
