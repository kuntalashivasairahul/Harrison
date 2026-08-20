#!/usr/bin/env python3
"""
scripts/probe_import_order.py
=============================
Smoke-test that torch/sentence-transformers and FAISS can be imported and used
in the same process, in that order, and that the encoder's output dimension
matches the index. Import order between these two has historically been a
source of native-library crashes on Apple Silicon.

    python scripts/probe_import_order.py          # MPS if available
    python scripts/probe_import_order.py --cpu    # force CPU

The --cpu flag replaces what used to be a second, near-identical file
(probe_import_order_cpu.py) differing only in the device string.
"""
import argparse
import sys
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from backend.config import EMBEDDING_MODEL  # noqa: E402  (after sys.path setup)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--cpu", action="store_true", help="Force CPU instead of MPS")
args = parser.parse_args()

device = "cpu" if args.cpu else ("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Loading model on {device.upper()}...")
model = SentenceTransformer(EMBEDDING_MODEL, device=device)
print("Model loaded. Encoding...")
emb = model.encode("test query")
print("Encoded.")

print("Importing faiss...")
import faiss  # noqa: E402  (import order is the point of this probe)

print("Loading FAISS index...")
index = faiss.read_index(str(_ROOT / "artifacts" / "vectorstore" / "index.faiss"))
print("Index loaded successfully! Running search...")
if emb.shape[0] != index.d:
    raise RuntimeError(f"Embedding dim {emb.shape[0]} does not match FAISS dim {index.d}")
distances, ids = index.search(emb.reshape(1, -1), 5)
print("Search successful, IDs:", ids)
