"""Query embedding for FAISS retrieval.

The SentenceTransformer used to be constructed at module scope, so importing
this module — directly or transitively, which every module in ``backend`` does
— loaded BGE-M3 weights.  That cost was paid by test runs and diagnostic
scripts that never embed anything.  The encoder is now built on first use.
"""

from __future__ import annotations

import logging
import threading

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.config import EMBEDDING_MODEL

log = logging.getLogger(__name__)

_model: SentenceTransformer | None = None
_model_lock = threading.Lock()


def get_model() -> SentenceTransformer:
    """Return the process-wide encoder, loading it on first call."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                log.info("Loading embedding model %s ...", EMBEDDING_MODEL)
                _model = SentenceTransformer(EMBEDDING_MODEL)
                log.info("Embedding model ready.")
    return _model


def warmup() -> None:
    """Load the encoder ahead of the first request (called at startup)."""
    get_model()


def __getattr__(name: str):
    """Keep ``embeddings.model`` working for existing callers (PEP 562)."""
    if name == "model":
        return get_model()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def embedding_dimension() -> int:
    model = get_model()
    if hasattr(model, "get_embedding_dimension"):
        return int(model.get_embedding_dimension())
    return int(model.get_sentence_embedding_dimension())


def embed_text(text: str):
    embedding = get_model().encode(
        [text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.array(embedding).astype("float32")
