#!/usr/bin/env python3
"""
scripts/test_environment.py
===========================
Verification script to verify each dependency in isolation.
Enforces OMP_NUM_THREADS=1 to prevent native macOS threading crashes.
"""

import os
# Prevent OpenMP conflict crashes
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys
from pathlib import Path
import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.config import EMBEDDING_DIM, EMBEDDING_MODEL

print("=== 1. Testing SentenceTransformer Import & Load ===")
try:
    from sentence_transformers import SentenceTransformer
    print("SentenceTransformer imported successfully.")
except Exception as e:
    print(f"Error importing SentenceTransformer: {e}")
    sys.exit(1)

device = "cpu"
# Try using MPS if possible to see if it is stable
import torch
if torch.backends.mps.is_available():
    print("Apple Silicon MPS is available.")
    # We will use CPU by default for stability, but we can verify both.

print(f"Loading model {EMBEDDING_MODEL!r} on {device}...")
try:
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)


print("\n=== 2. Testing One Embedding Call ===")
try:
    q = "What is the initial fluid replacement rate and type for a patient in diabetic ketoacidosis?"
    emb = model.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    print(f"Embedding successful. Shape: {emb.shape}")
    assert emb.shape == (1, EMBEDDING_DIM), f"Expected shape (1, {EMBEDDING_DIM}), got {emb.shape}"
except Exception as e:
    print(f"Error during embedding call: {e}")
    sys.exit(1)


print("\n=== 3. Testing FAISS Import & Index Build ===")
try:
    import faiss
    print(f"FAISS imported successfully. Version: {faiss.__version__}")

    # Build a small dummy index
    dim = emb.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(emb)
    print(f"FAISS index built successfully. Total vectors: {index.ntotal}")
    assert index.ntotal == 1, f"Expected 1 vector, got {index.ntotal}"
except Exception as e:
    print(f"Error during FAISS setup/build: {e}")
    sys.exit(1)


print("\n=== 4. Testing One Retrieval Query ===")
try:
    distances, ids = index.search(emb, 1)
    print(f"Retrieval query successful.")
    print(f"Distances: {distances}")
    print(f"Retrieved IDs: {ids}")
    assert ids[0][0] == 0, f"Expected retrieved ID to be 0, got {ids[0][0]}"
except Exception as e:
    print(f"Error during retrieval search: {e}")
    sys.exit(1)

print("\n=== All isolated dependency tests passed successfully! ===")
