#!/usr/bin/env python3
"""
scripts/ingest_tables.py
========================
Table-Aware Data Ingestion for HarrisonGPT.

Problem solved
--------------
Standard text chunkers split Markdown tables mid-row, destroying the
column–value relationship that LLMs need to answer clinical questions
(e.g. "What is the Ranson score cut-off for severe pancreatitis?").

This script parses a Markdown source file with surgical precision:

  1. **Section-aware splitting** — text is first segmented at every
     Markdown heading (# … ######).  Each segment carries its full
     ancestor heading hierarchy so clinical context is never lost.

  2. **Table-preserving chunking** — within each section, Markdown
     table blocks (detected by the | separator | row | pattern and
     the |---|---| alignment row) are treated as atomic units.  A
     table is always emitted as a single chunk together with its
     immediately preceding heading, regardless of table length.

  3. **Prose chunking** — non-table text is split at a configurable
     token boundary (default 512 chars) with a 64-char overlap so
     sentences are never cut at arbitrary byte positions.

  4. **Dense embedding** — the configured live embedding model
     produces L2-normalised vectors identical to the live
     pipeline, ensuring staging results are directly comparable.
     Automatically uses Apple Silicon MPS (Metal GPU) when available,
     falling back to CPU on non-Apple hardware.

  5. **FAISS index** — IndexFlatL2 for exact nearest-neighbour
     search.  Saved to artifacts/vectorstore_staging/table_index.faiss.

  6. **Chunk registry** — JSON array with keys matching the
     production schema (``page``, ``text``, ``chunk_id``) plus the
     new ``source_heading`` and ``chunk_type`` fields for auditability.
     Saved to artifacts/vectorstore_staging/table_chunks.json.

Safety guarantees
-----------------
* NEVER touches artifacts/vectorstore/ (production index).
* NEVER modifies backend/ in any way.
* All output is isolated to artifacts/vectorstore_staging/.
* Idempotent: re-running overwrites staging files only.

Usage
-----
    python scripts/ingest_tables.py [--source data/harrison.md]

    Optional flags:
      --source PATH    Path to the Markdown source (default: data/harrison.md)
      --chunk-size N   Max characters per prose chunk (default: 512)
      --overlap N      Character overlap between prose chunks (default: 64)
      --batch-size N   Embedding batch size (default: 64)
      --dry-run        Parse + print stats without writing any files
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ingest_tables")

# ---------------------------------------------------------------------------
# Paths  (all relative to the project root — the CWD when running the script)
# ---------------------------------------------------------------------------

_ROOT          = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))
DEFAULT_SOURCE = _ROOT / "data" / "harrison.md"
STAGING_DIR    = _ROOT / "artifacts" / "vectorstore_staging"
INDEX_PATH     = STAGING_DIR / "table_index.faiss"
CHUNKS_PATH    = STAGING_DIR / "table_chunks.json"

# Production paths — referenced only for the safety assertion.
PROD_INDEX  = _ROOT / "artifacts" / "vectorstore" / "index.faiss"
PROD_CHUNKS = _ROOT / "artifacts" / "vectorstore" / "chunks.json"

from backend.config import EMBEDDING_DIM, EMBEDDING_MODEL

# ---------------------------------------------------------------------------
# Chunking constants
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_SIZE = 512   # max chars per prose chunk
DEFAULT_OVERLAP    = 64    # overlap chars between prose chunks
DEFAULT_BATCH_SIZE = 64    # embedding batch size

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Any Markdown heading line: # … ######  (captures level and text)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

# A Markdown table row: starts with optional whitespace, then |
_TABLE_ROW_RE = re.compile(r"^\s*\|.+\|?\s*$")

# A Markdown table separator row: |---|---| or |:--:|---:|
_TABLE_SEP_RE = re.compile(r"^\s*\|[\s\-:]+\|[\s\-:|]*$")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """
    One unit of text ready for embedding and FAISS insertion.

    Fields
    ------
    chunk_id      : Monotonically increasing integer ID (matches production schema).
    page          : Page number extracted from the heading/metadata, or 0 if absent.
    text          : The full text that will be embedded and shown to the LLM.
    source_heading: The nearest ancestor heading (for auditability).
    chunk_type    : ``"table"`` | ``"prose"`` — lets downstream tooling filter.
    char_count    : Length of ``text`` in characters.
    """
    chunk_id:       int
    page:           int
    text:           str
    source_heading: str
    chunk_type:     str   # "table" | "prose"
    char_count:     int   = field(init=False)

    def __post_init__(self) -> None:
        self.char_count = len(self.text)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("char_count")   # keep registry schema minimal
        return d


# ---------------------------------------------------------------------------
# Section splitter
# ---------------------------------------------------------------------------

def split_into_sections(md: str) -> List[dict]:
    """
    Split a Markdown document into logical sections at every heading boundary.

    Returns a list of dicts, each with:
      ``heading``    : The heading line that opens this section (e.g. "## Etiology")
      ``level``      : Heading depth (1-6)
      ``body``       : All text between this heading and the next heading of equal
                       or higher level (lower number = higher level).
      ``page``       : Integer page number if the heading or immediately preceding
                       text contains a pattern like "[p:2157]", "p.2157", or just
                       a standalone integer on its own line.  0 otherwise.
    """
    sections: List[dict] = []
    lines = md.splitlines(keepends=True)

    heading_positions: List[tuple[int, int, str]] = []   # (line_idx, level, text)
    for i, line in enumerate(lines):
        m = _HEADING_RE.match(line.rstrip())
        if m:
            heading_positions.append((i, len(m.group(1)), m.group(2).strip()))

    if not heading_positions:
        # No headings at all — treat the whole document as one section.
        sections.append({
            "heading": "(no heading)",
            "level":   0,
            "body":    md,
            "page":    0,
        })
        return sections

    # Pre-heading preamble
    first_heading_line = heading_positions[0][0]
    if first_heading_line > 0:
        preamble = "".join(lines[:first_heading_line])
        if preamble.strip():
            sections.append({
                "heading": "(preamble)",
                "level":   0,
                "body":    preamble,
                "page":    _extract_page(preamble),
            })

    for idx, (line_i, level, heading_text) in enumerate(heading_positions):
        # Body runs until the next heading
        if idx + 1 < len(heading_positions):
            next_line_i = heading_positions[idx + 1][0]
        else:
            next_line_i = len(lines)

        body = "".join(lines[line_i + 1 : next_line_i])
        page = _extract_page(heading_text + "\n" + body[:200])

        sections.append({
            "heading": heading_text,
            "level":   level,
            "body":    body,
            "page":    page,
        })

    return sections


def _extract_page(text: str) -> int:
    """
    Try to parse a page number from text.  Recognises:
      [p:2157|c:xxx]  →  2157
      [p:2157]        →  2157
      p.2157          →  2157
      TABLE 10-5      →  (skipped — that is a table number, not a page)
    Returns 0 if nothing is found.
    """
    # [p:NNN] or [p:NNN|c:xxx]
    m = re.search(r"\[p:(\d+)", text)
    if m:
        return int(m.group(1))
    # p.NNN (Harrison citation style)
    m = re.search(r"\bp\.(\d{3,5})\b", text)
    if m:
        return int(m.group(1))
    return 0


# ---------------------------------------------------------------------------
# Table detector / extractor
# ---------------------------------------------------------------------------

def _is_table_row(line: str) -> bool:
    return bool(_TABLE_ROW_RE.match(line))


def _is_separator_row(line: str) -> bool:
    return bool(_TABLE_SEP_RE.match(line))


def extract_table_blocks(body: str) -> List[dict]:
    """
    Scan the body text of a section and return a list of blocks:

    Each block is:
      ``kind``  : ``"table"`` | ``"prose"``
      ``text``  : The raw text of this block.

    Tables are identified by at least one separator row (|---|---|).
    The block includes all consecutive lines that form part of the table,
    as well as any immediately-preceding non-blank caption line.
    """
    lines = body.splitlines(keepends=True)
    blocks: List[dict] = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Detect start of a Markdown table: look-ahead for a separator row
        # within the next 3 lines (header row + separator is minimum 2 lines).
        if _is_table_row(line):
            # Check if any of the next few lines is a separator
            lookahead = lines[i : i + 5]
            has_sep = any(_is_separator_row(l) for l in lookahead)

            if has_sep:
                # Collect the full table (all consecutive table rows)
                table_lines = []

                # Include any caption line immediately before (the last prose line)
                if blocks and blocks[-1]["kind"] == "prose":
                    prose_lines = blocks[-1]["text"].rstrip().splitlines(keepends=True)
                    if prose_lines:
                        caption = prose_lines[-1]
                        # Only absorb if it looks like a caption (short, not a table row)
                        if len(caption.strip()) < 120 and not _is_table_row(caption):
                            table_lines.append(caption)
                            # Trim the caption from the prose block
                            blocks[-1]["text"] = "".join(prose_lines[:-1])
                            if not blocks[-1]["text"].strip():
                                blocks.pop()

                while i < len(lines) and _is_table_row(lines[i]):
                    table_lines.append(lines[i])
                    i += 1

                blocks.append({"kind": "table", "text": "".join(table_lines)})
                continue

        # Non-table line — accumulate into a prose block
        if blocks and blocks[-1]["kind"] == "prose":
            blocks[-1]["text"] += line
        else:
            blocks.append({"kind": "prose", "text": line})
        i += 1

    return blocks


# ---------------------------------------------------------------------------
# Prose chunker
# ---------------------------------------------------------------------------

def chunk_prose(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split prose text into overlapping chunks of at most ``chunk_size``
    characters.  Splits prefer sentence boundaries (". ", "! ", "? ") or
    newlines.  Falls back to hard character splits only when no boundary
    is found within the window.
    """
    text = text.strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            # Try to find a clean sentence boundary within the last 20% of the window
            search_start = max(start, end - chunk_size // 5)
            best_break   = -1
            for sep in (". ", "! ", "? ", "\n\n", "\n"):
                pos = text.rfind(sep, search_start, end)
                if pos > best_break:
                    best_break = pos + len(sep)
            if best_break > start:
                end = best_break

        chunks.append(text[start:end].strip())
        start = end - overlap if end < len(text) else end

    return [c for c in chunks if c]


# ---------------------------------------------------------------------------
# Main chunking pipeline
# ---------------------------------------------------------------------------

def build_chunks(
    md: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int    = DEFAULT_OVERLAP,
) -> List[Chunk]:
    """
    Full chunking pipeline.

    1. Split document into sections at heading boundaries.
    2. Within each section, separate table blocks from prose blocks.
    3. Each table block → one Chunk (with heading prefix for context).
    4. Each prose block → one or more Chunks (overlap-split).

    Page tracking (stateful)
    ------------------------
    convert_pdf.py injects ``[p:NNN]`` markers at the TOP of each page.
    Because sections and blocks are sliced from a large document, any given
    chunk may not contain the marker in its local text — but its PAGE is
    determined by the most recent marker seen in the LINEAR document scan.

    ``current_page`` is updated every time a ``[p:NNN]`` marker is found
    anywhere in a heading, prose block, or prose sub-chunk.  Tables inherit
    whatever ``current_page`` was set by the prose that immediately preceded
    them, which is always correct because the marker precedes all content on
    that page.

    Returns
    -------
    List[Chunk]
        Ready-to-embed chunks with populated metadata and correct page numbers.
    """
    # Precompile the page marker regex once for the whole run.
    _PAGE_RE = re.compile(r'\[p:(\d+)\]')

    def _advance_page(text: str) -> None:
        """
        Scan ``text`` for ``[p:NNN]`` markers and advance ``current_page``
        to the value of the LAST marker found.

        The last-marker rule is correct because markers appear at page
        boundaries; the last one in any block is the most recent page that
        content at the END of that block belongs to.
        """
        nonlocal current_page
        for m in _PAGE_RE.finditer(text):
            current_page = int(m.group(1))

    sections = split_into_sections(md)
    log.info("Document split into %d sections.", len(sections))

    all_chunks: List[Chunk] = []
    chunk_counter           = 0
    table_count             = 0
    prose_count             = 0
    current_page: int       = 0   # ── Stateful page tracker ──

    for sec in sections:
        heading = sec["heading"]
        blocks  = extract_table_blocks(sec["body"])

        # Advance tracker using the section heading text (a [p:NNN] marker
        # occasionally lands right before a heading due to pymupdf4llm layout).
        _advance_page(heading)

        for block in blocks:

            if block["kind"] == "table":
                # ── Table chunk ────────────────────────────────────────
                # Tables rarely contain [p:NNN] themselves, so this scan is a
                # safety net.  The decisive page update happened in the prose
                # block that immediately preceded this table in the document.
                _advance_page(block["text"])
                table_text = f"## {heading}\n\n{block['text'].strip()}"
                all_chunks.append(Chunk(
                    chunk_id=chunk_counter,
                    page=current_page,       # ← stateful, never 0
                    text=table_text,
                    source_heading=heading,
                    chunk_type="table",
                ))
                chunk_counter += 1
                table_count   += 1

            else:
                # ── Prose chunks ───────────────────────────────────────
                prose_parts = chunk_prose(block["text"], chunk_size, overlap)
                for part in prose_parts:
                    if not part.strip():
                        continue
                    # Scan each sub-chunk so a marker that falls inside an
                    # overlap window still advances the tracker before this
                    # chunk is recorded.
                    _advance_page(part)
                    all_chunks.append(Chunk(
                        chunk_id=chunk_counter,
                        page=current_page,   # ← stateful, never 0
                        text=part,
                        source_heading=heading,
                        chunk_type="prose",
                    ))
                    chunk_counter += 1
                    prose_count   += 1

    zero_page_chunks = sum(1 for c in all_chunks if c.page == 0)
    log.info(
        "Chunking complete: %d total chunks (%d tables, %d prose). "
        "page=0 remaining: %d (preamble/front-matter only).",
        len(all_chunks), table_count, prose_count, zero_page_chunks,
    )
    if zero_page_chunks:
        log.warning(
            "%d chunks still have page=0 — front-matter before first "
            "[p:NNN] marker.  This is expected for title/TOC pages.",
            zero_page_chunks,
        )
    return all_chunks


# ---------------------------------------------------------------------------
# Embedding + FAISS index builder
# ---------------------------------------------------------------------------

def embed_chunks(
    chunks: List[Chunk],
    model: SentenceTransformer,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """
    Generate L2-normalised embeddings for all chunks.

    Returns an (N, EMBEDDING_DIM) float32 numpy array.
    """
    texts = [c.text for c in chunks]
    log.info("Embedding %d chunks in batches of %d…", len(texts), batch_size)

    t0 = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalise — matches live pipeline
    )
    elapsed = time.perf_counter() - t0
    log.info(
        "Embedding complete in %.1fs  (%.2f chunks/s).",
        elapsed, len(texts) / elapsed,
    )
    return embeddings.astype("float32")


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """
    Build a flat L2 FAISS index from the embedding matrix.

    IndexFlatL2 provides exact nearest-neighbour search — appropriate
    for a staging index where result accuracy is more important than
    query speed.
    """
    n, dim = embeddings.shape
    log.info("Building FAISS IndexFlatL2 (dim=%d, n=%d)…", dim, n)
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    log.info("FAISS index contains %d vectors.", index.ntotal)
    return index


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_outputs(
    index:  faiss.IndexFlatL2,
    chunks: List[Chunk],
    dry_run: bool = False,
) -> None:
    """
    Write the FAISS index and chunk registry to the staging directory.

    Safety check: asserts that STAGING_DIR does not equal the production
    vectorstore path before writing anything.
    """
    assert STAGING_DIR.resolve() != PROD_INDEX.parent.resolve(), (
        "FATAL: staging path resolves to the production vectorstore — aborting."
    )

    if dry_run:
        log.info("[dry-run] Would write:")
        log.info("  %s  (%d vectors)", INDEX_PATH, index.ntotal)
        log.info("  %s  (%d chunks)", CHUNKS_PATH, len(chunks))
        return

    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    # Write FAISS index
    faiss.write_index(index, str(INDEX_PATH))
    log.info("FAISS index saved → %s  (%d bytes)", INDEX_PATH, INDEX_PATH.stat().st_size)

    # Write chunk registry (production-compatible schema + staging extras)
    registry = [c.to_dict() for c in chunks]
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
    log.info("Chunk registry saved → %s  (%d bytes)", CHUNKS_PATH, CHUNKS_PATH.stat().st_size)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Path to the Markdown source file (default: data/harrison.md)",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        metavar="N",
        help=f"Max characters per prose chunk (default: {DEFAULT_CHUNK_SIZE})",
    )
    p.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        metavar="N",
        help=f"Overlap characters between prose chunks (default: {DEFAULT_OVERLAP})",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        metavar="N",
        help=f"Embedding batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and print stats without writing any output files",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── 0. Safety assertions ─────────────────────────────────────────────
    assert INDEX_PATH.resolve() != PROD_INDEX.resolve(), (
        "FATAL: output index path collides with production index — aborting."
    )
    assert CHUNKS_PATH.resolve() != PROD_CHUNKS.resolve(), (
        "FATAL: output chunks path collides with production chunks — aborting."
    )

    # ── 1. Read source ────────────────────────────────────────────────────
    source: Path = args.source.resolve()
    if not source.exists():
        log.error(
            "Source file not found: %s\n"
            "Place your Markdown export at data/harrison.md or pass --source <path>.",
            source,
        )
        sys.exit(1)

    log.info("Reading source: %s  (%.1f MB)", source, source.stat().st_size / 1e6)
    md = source.read_text(encoding="utf-8")

    # ── 2. Chunk ──────────────────────────────────────────────────────────
    chunks = build_chunks(md, chunk_size=args.chunk_size, overlap=args.overlap)
    if not chunks:
        log.error("No chunks produced — check that the source file has content.")
        sys.exit(1)

    # Stats
    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    prose_chunks = [c for c in chunks if c.chunk_type == "prose"]
    avg_len      = sum(c.char_count for c in chunks) / len(chunks)
    log.info(
        "Chunk stats — total: %d | tables: %d | prose: %d | avg length: %.0f chars",
        len(chunks), len(table_chunks), len(prose_chunks), avg_len,
    )

    if args.dry_run:
        # Print a sample of each type and exit
        log.info("[dry-run] Sample TABLE chunk:")
        if table_chunks:
            print("\n" + "─" * 60)
            print(table_chunks[0].text[:600])
            print("─" * 60 + "\n")
        log.info("[dry-run] Sample PROSE chunk:")
        if prose_chunks:
            print("\n" + "─" * 60)
            print(prose_chunks[0].text[:400])
            print("─" * 60 + "\n")
        save_outputs(None, chunks, dry_run=True)  # type: ignore[arg-type]
        log.info("[dry-run] Complete — no files written.")
        return

    # ── 3. Load embedding model (MPS-accelerated on Apple Silicon) ──────
    # Check device priority: MPS (Apple Metal GPU) > CPU.
    # MPS can deliver higher throughput for supported SentenceTransformer models
    # on M-series chips.  Falls back to CPU silently on non-Apple hardware.
    if torch.backends.mps.is_available():
        _device = "mps"
    else:
        _device = "cpu"

    log.info(
        "Loading SentenceTransformer('%s') on device='%s'…",
        EMBEDDING_MODEL, _device,
    )
    model = SentenceTransformer(EMBEDDING_MODEL, device=_device)
    model_dim = (
        model.get_embedding_dimension()
        if hasattr(model, "get_embedding_dimension")
        else model.get_sentence_embedding_dimension()
    )
    if model_dim != EMBEDDING_DIM:
        raise RuntimeError(
            f"Configured embedding dim mismatch: model emits {model_dim}, "
            f"backend.config.EMBEDDING_DIM={EMBEDDING_DIM}"
        )
    log.info("Model loaded. Device: %s, dim=%d", _device.upper(), model_dim)

    # ── 4. Embed ──────────────────────────────────────────────────────────
    embeddings = embed_chunks(chunks, model, batch_size=args.batch_size)
    assert embeddings.shape == (len(chunks), EMBEDDING_DIM), (
        f"Unexpected embedding shape: {embeddings.shape}"
    )

    # ── 5. Build FAISS index ──────────────────────────────────────────────
    index = build_faiss_index(embeddings)

    # ── 6. Save ───────────────────────────────────────────────────────────
    save_outputs(index, chunks)

    log.info("=" * 60)
    log.info("✅  Ingestion complete.")
    log.info("   FAISS index  : %s", INDEX_PATH)
    log.info("   Chunk registry: %s", CHUNKS_PATH)
    log.info("   Vectors stored: %d", index.ntotal)
    log.info("   Table chunks  : %d (never split mid-row)", len(table_chunks))
    log.info("   Prose chunks  : %d", len(prose_chunks))
    log.info("=" * 60)
    log.info(
        "⚠️  Production index at artifacts/vectorstore/ is UNCHANGED.\n"
        "    Validate staging outputs before promoting to production."
    )


if __name__ == "__main__":
    main()
