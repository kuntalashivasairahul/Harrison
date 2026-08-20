import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import numpy as np
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from backend.config import EMBEDDING_MODEL

print("Imports OK.")
print("Loading model...")
model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
print("Model loaded.")
