#!/usr/bin/env python3
"""
scripts/convert_pdf.py
======================
PDF → Markdown conversion with explicit [p:NNN] page markers.

Problem solved
--------------
pymupdf4llm's default single-string output drops page boundary information,
causing every chunk produced by ingest_tables.py to inherit page=0.
context_router.py then cannot sort chunks chronologically.

This script fixes that by using page_chunks=True, iterating over each
page dict, and injecting a standalone [p:{page_num}] marker at the top
of each page's Markdown text before concatenating.

The marker format [p:NNN] is understood natively by:
  - ingest_tables.py → _extract_page()
  - rag.py           → context markers in fused context
  - context_router.py → _get_page() for chronological sort

Usage
-----
    python scripts/convert_pdf.py [--pdf data/harrison.pdf] [--out data/harrison.md]

    Optional flags:
      --pdf PATH   Path to the source PDF  (default: data/harrison.pdf)
      --out PATH   Path to write Markdown  (default: data/harrison.md)
      --quiet      Suppress per-page progress output

Output
------
    data/harrison.md  — Markdown with [p:NNN] injected at each page boundary.

Safety
------
    Read-only access to the PDF; the live backend/ and artifacts/vectorstore/
    directories are NEVER touched.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pymupdf4llm  # pip install pymupdf4llm

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("convert_pdf")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_ROOT         = Path(__file__).resolve().parents[1]
DEFAULT_PDF   = _ROOT / "data" / "harrison.pdf"
DEFAULT_OUT   = _ROOT / "data" / "harrison.md"

# Log a progress line every N pages (keeps noise low for 4000+ page books).
PROGRESS_EVERY = 100


# ---------------------------------------------------------------------------
# Core conversion
# ---------------------------------------------------------------------------

def convert(pdf_path: Path, out_path: Path, quiet: bool = False) -> None:
    """
    Extract per-page Markdown from ``pdf_path`` and write it to ``out_path``
    with [p:NNN] markers injected at every page boundary.

    Parameters
    ----------
    pdf_path : Path
        Source PDF file.  Must exist.
    out_path : Path
        Destination Markdown file.  Parent directory will be created if absent.
    quiet : bool
        If True, suppress per-page progress logging.
    """
    if not pdf_path.exists():
        log.error("PDF not found: %s", pdf_path)
        sys.exit(1)

    log.info("Source  : %s  (%.1f MB)", pdf_path, pdf_path.stat().st_size / 1e6)
    log.info("Output  : %s", out_path)
    log.info("Starting PDF → Markdown conversion with page markers…")

    t0 = time.perf_counter()

    # page_chunks=True returns a list of dicts, one per page.
    # Each dict has keys: "metadata" (dict with "page" key, 0-based) and "text" (str).
    page_chunks = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)

    total_pages = len(page_chunks)
    log.info("pymupdf4llm extracted %d pages.", total_pages)

    parts: list[str] = []

    for chunk in page_chunks:
        # pymupdf4llm uses 0-based page indexing → convert to 1-based to match
        # Harrison's printed page numbers after applying the FAISS offset elsewhere.
        page_num: int = chunk.get("metadata", {}).get("page", 0) + 1
        text: str     = chunk.get("text", "")

        # Inject the page marker as a standalone block so _extract_page()
        # in ingest_tables.py reliably picks it up via the [p:NNN] regex.
        parts.append(f"\n\n[p:{page_num}]\n\n{text}")

        if not quiet and page_num % PROGRESS_EVERY == 0:
            pct = page_num / total_pages * 100
            log.info("  … processed page %d / %d  (%.0f%%)", page_num, total_pages, pct)

    full_md: str = "".join(parts)

    # Ensure output directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(full_md, encoding="utf-8")

    elapsed = time.perf_counter() - t0
    out_mb  = out_path.stat().st_size / 1e6

    log.info("=" * 60)
    log.info("✅  Conversion complete.")
    log.info("   Pages processed : %d", total_pages)
    log.info("   Output size     : %.1f MB", out_mb)
    log.info("   Elapsed         : %.1f s  (%.0f pages/s)", elapsed, total_pages / elapsed)
    log.info("   Page markers    : [p:1] … [p:%d] injected", total_pages)
    log.info("=" * 60)
    log.info("Next step: python scripts/ingest_tables.py --source %s", out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--pdf",
        type=Path,
        default=DEFAULT_PDF,
        help=f"Path to the source PDF (default: {DEFAULT_PDF})",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Path to write the Markdown output (default: {DEFAULT_OUT})",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-page progress logging",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(pdf_path=args.pdf.resolve(), out_path=args.out.resolve(), quiet=args.quiet)