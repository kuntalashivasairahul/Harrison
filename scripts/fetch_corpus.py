#!/usr/bin/env python3
"""
scripts/fetch_corpus.py
=======================
Pull the licensed corpus from a private Hugging Face Dataset into the layout
the app expects, before the API process starts.

Why this exists
---------------
The FAISS index and ``chunks.json`` are derived from Harrison's Principles of
Internal Medicine.  ``chunks.json`` is ~33 MB of verbatim textbook prose, so it
cannot ship inside a public Space repo or a public image layer (RULE 3.1).  It
lives in a *private* Dataset instead and arrives here at boot, authenticated by
the ``HF_TOKEN`` Space secret.

Why it is a script and not part of the app
------------------------------------------
Two rules force it out of ``backend/``:

* ``backend/api/main.py`` mounts ``StaticFiles("storage/pages")`` at **import**
  time and StaticFiles raises when the directory is missing, so this must
  complete before ``backend.api.main`` is imported at all -- the FastAPI
  lifespan handler runs far too late.
* CLAUDE.md's import-time purity rule forbids networked work at import under
  ``backend/``, and this downloads roughly half a gigabyte.

Dataset layout
--------------
The private Dataset mirrors the application's own paths, so a single
``snapshot_download`` lands every file where the code already looks for it and
nothing has to be moved afterwards::

    artifacts/vectorstore/index.faiss     <- backend/config.py VECTORSTORE_DIR
    artifacts/vectorstore/chunks.json     <- backend/retrieval/rag.py CHUNKS_PATH
    storage/pages/small/page_*.webp       <- served at /pages/small/* by main.py

``storage/pages/full/`` is deliberately absent.  It is 3.8 GB of PNGs, which
would mean a 4.3 GB pull on every cold wake of a free Space.  Set
``HARRISON_PAGE_FULL_RES=false`` so the lightbox falls back to the thumbnail
instead of opening a 404.

Local runs
----------
With the corpus already on disk (a dev machine, or any checkout with git-lfs
working) this is a no-op: it verifies and exits 0 without touching the network.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]

# Minimum plausible size per file.  A git-lfs pointer is ~130 bytes, so this
# catches the failure that actually happens in practice: a checkout without
# git-lfs, or a partial download, leaving a file that exists and is useless.
_MIN_BYTES = 1_000_000

REQUIRED = {
    APP_ROOT / "artifacts" / "vectorstore" / "index.faiss": _MIN_BYTES,
    APP_ROOT / "artifacts" / "vectorstore" / "chunks.json": _MIN_BYTES,
}

# Only these reach the container.  Anything else in the Dataset stays there.
ALLOW_PATTERNS = [
    "artifacts/vectorstore/index.faiss",
    "artifacts/vectorstore/chunks.json",
    "storage/pages/small/*",
]


def _missing() -> list[Path]:
    """Required files that are absent, empty, or too small to be real."""
    return [p for p, floor in REQUIRED.items() if not p.is_file() or p.stat().st_size < floor]


def _die(message: str) -> None:
    """Exit loudly.

    The container must not limp on to uvicorn after a failed fetch.  It would
    crash a second later inside a StaticFiles mount, and that traceback names
    a directory rather than the actual cause, which is a slow thing to debug
    through a Space build log.
    """
    print(f"fetch_corpus: FATAL: {message}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    if not _missing():
        print("fetch_corpus: corpus already present, skipping download.")
        _ensure_page_dirs()
        return

    repo = os.getenv("HARRISON_CORPUS_REPO")
    token = os.getenv("HF_TOKEN")

    if not repo:
        _die(
            "corpus files are missing and HARRISON_CORPUS_REPO is unset.\n"
            "  On a Space: set HARRISON_CORPUS_REPO to your private dataset id\n"
            "  (e.g. 'your-name/harrison-corpus') in Settings > Variables and secrets.\n"
            "  Locally: run `git lfs pull` to materialise artifacts/vectorstore/."
        )
    if not token:
        _die(
            f"HARRISON_CORPUS_REPO is set to '{repo}' but HF_TOKEN is unset.\n"
            "  A private dataset needs a read-scoped token added as a Space secret."
        )

    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError

    print(f"fetch_corpus: downloading corpus from dataset '{repo}' ...")
    try:
        snapshot_download(
            repo_id=repo,
            repo_type="dataset",
            token=token,
            local_dir=str(APP_ROOT),
            allow_patterns=ALLOW_PATTERNS,
        )
    except HfHubHTTPError as exc:
        # 401/403 here is the single most likely production failure: a token
        # that expired, was revoked, or lacks read access to the dataset.
        _die(f"download from '{repo}' failed: {exc}")

    still_missing = _missing()
    if still_missing:
        _die(
            "download completed but required files are still missing or truncated:\n  "
            + "\n  ".join(str(p.relative_to(APP_ROOT)) for p in still_missing)
            + f"\n  Check that the dataset mirrors these exact paths: {ALLOW_PATTERNS}"
        )

    _ensure_page_dirs()
    for path in REQUIRED:
        print(f"fetch_corpus: ok  {path.relative_to(APP_ROOT)}  ({path.stat().st_size / 1e6:.0f} MB)")


def _ensure_page_dirs() -> None:
    """StaticFiles(directory=...) raises at import if the directory is absent.

    ``storage/pages`` must exist even when no page images were shipped, or
    ``backend.api.main`` cannot be imported at all.  ``full`` is created empty
    on purpose: with HARRISON_PAGE_FULL_RES=false nothing routes there, but an
    existing empty directory fails as a clean 404 rather than a mount error.
    """
    for sub in ("small", "full"):
        (APP_ROOT / "storage" / "pages" / sub).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
