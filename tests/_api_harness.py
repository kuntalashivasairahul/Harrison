"""Import ``backend.api.main`` hermetically.

Importing the real module pulls in the FAISS index, a 33 MB chunk registry, a
full-corpus BM25 build, the BGE-M3 encoder, and a live Gemini model-discovery
call.  Tests stub those modules in ``sys.modules`` before import so the HTTP
layer can be exercised in milliseconds with no network and no model weights.

Not named ``test_*`` on purpose — this is a helper, not a test module.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

REFUSAL_STR = "Insufficient information in the provided context."


def _module(name: str, **attrs) -> types.ModuleType:
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


class _FakeIndex:
    d = 1024
    ntotal = 16983


def build_stubs() -> dict[str, types.ModuleType]:
    """Return the ``sys.modules`` entries needed to import the API cheaply."""
    fake_rag = _module(
        "backend.retrieval.rag",
        index=_FakeIndex(),
        chunks=[{"page": 1, "text": "chunk"}] * 16983,
        retrieve=MagicMock(return_value=[]),
        warmup=MagicMock(),
    )

    fake_embeddings = _module(
        "backend.retrieval.embeddings",
        embed_text=MagicMock(return_value=_FakeArray()),
        embedding_dimension=MagicMock(return_value=1024),
        warmup=MagicMock(),
        get_model=MagicMock(),
    )

    fake_rerank = _module(
        "backend.retrieval.rerank",
        warmup_reranker=MagicMock(),
        rerank=MagicMock(return_value=[]),
    )

    key_manager = MagicMock()
    key_manager.has_keys.return_value = True
    key_manager.key_count = 3
    key_manager.available_key_count = 3

    llm_router = MagicMock()
    llm_router.status.return_value = []

    fake_llm = _module(
        "backend.llm.llm",
        ask_llm=MagicMock(return_value=("answer", "draft", False, "verified")),
        verify_answer=MagicMock(),
        REFUSAL_STR=REFUSAL_STR,
        key_manager=key_manager,
        llm_router=llm_router,
        PROD_MODEL="gemini-2.5-flash",
        BACKUP_MODEL="gemini-1.5-flash",
        prod_model=MagicMock(return_value="gemini-2.5-flash"),
        backup_model=MagicMock(return_value="gemini-1.5-flash"),
        resolve_models=MagicMock(return_value=("gemini-2.5-flash", "gemini-1.5-flash")),
    )

    return {
        "backend.retrieval.rag": fake_rag,
        "backend.retrieval.embeddings": fake_embeddings,
        "backend.retrieval.rerank": fake_rerank,
        "backend.llm.llm": fake_llm,
    }


class _FakeArray:
    """Minimal stand-in for the numpy array returned by ``embed_text``."""

    def flatten(self):
        return self

    def tolist(self):
        return [0.1] * 1024


def import_main():
    """Import (or re-import) ``backend.api.main`` with stubs installed."""
    stubs = build_stubs()
    saved = {name: sys.modules.get(name) for name in stubs}
    saved["backend.api.main"] = sys.modules.get("backend.api.main")

    sys.modules.update(stubs)
    sys.modules.pop("backend.api.main", None)
    try:
        import backend.api.main as main  # noqa: PLC0415
        return main
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
