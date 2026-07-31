#!/usr/bin/env python3
"""
scripts/ingest_tables_aware.py
==============================
Table-Aware Markdown Ingestion for HarrisonGPT  (production-grade successor).

Problem solved
--------------
Standard text chunkers blindly slice through Markdown tables mid-row, destroying
the column-value relationships that LLMs require to answer clinical questions
(e.g. "What is the Ranson score cut-off for severe pancreatitis?").

This script replaces the naive chunking step with a surgical pipeline:

  1. **Sequential page tracking** — the current source file (data/harrison.md)
     contains 4 273 ``[p:1]`` markers (all identical, due to a known
     ``pymupdf4llm`` output bug in ``convert_pdf.py``).  Rather than trusting
     the embedded page number, this script counts the Nth occurrence of a
     ``[p:NNN]`` marker and assigns page number N to every chunk produced
     after that marker.  This faithfully reconstructs the 1–4 273 page range
     seen in the production index.

  2. **Section-aware splitting** — the document is first segmented at every
     Markdown heading boundary (``#`` … ``######``).  Each segment carries its
     full ancestor heading for clinical context.

  3. **Table-preserving chunking** — within each section, Markdown table blocks
     (identified by at least one ``|---|---| separator row``) are treated as
     atomic units and emitted as single chunks, prefixed with the heading.

  4. **Large-table row-split fallback** — when a table exceeds
     ``--max-table-chars`` (default 1 800, about 512 embedding-model tokens),
     it is split into row batches of ``--table-rows-per-batch`` (default 20).
     Every batch is prefixed with the **original header row + separator row**
     so column context is never lost.

  5. **Prose chunking** — non-table text is split at ``--chunk-size`` characters
     with ``--overlap`` character overlap, preferring sentence boundaries.

  6. **Dense embedding** — SentenceTransformers(``BAAI/bge-m3``) produces
     1024-dim L2-normalised vectors identical to the live pipeline.
     Apple Silicon MPS is used when available; falls back to CPU silently.

  7. **FAISS index** — ``IndexFlatL2`` for exact nearest-neighbour search,
     matching the production index type.

  8. **Production-compatible schema** — chunk registry is written as a JSON
     array of ``{page: int, text: str}`` objects only, matching the schema
     expected by ``backend/retrieval/rag.py``.  The list position IS the
     ``chunk_id``; it is never stored in the JSON (consistent with production).

Safety guarantees
-----------------
* Default output is ``artifacts/vectorstore_staging/`` — production is NEVER
  touched unless ``--promote`` is explicitly passed.
* With ``--promote``, the existing production files are first backed up to
  ``artifacts/vectorstore_backup/<ISO-timestamp>/`` before being replaced.
* ``--dry-run`` parses, chunks, and prints stats without writing any files.
* Re-running always overwrites the staging directory only (idempotent).

Usage
-----
    # 1. Dry-run: parse + stats, no files written
    python scripts/ingest_tables_aware.py --dry-run

    # 2. Stage (safe): write to artifacts/vectorstore_staging/
    python scripts/ingest_tables_aware.py

    # 3. Promote to production (backs up prod first)
    python scripts/ingest_tables_aware.py --promote

    # 4. Custom source directory (globs all *.md files)
    python scripts/ingest_tables_aware.py --source data/

    Optional flags:
      --source PATH             Markdown file or directory (default: data/harrison.md)
      --chunk-size N            Max chars per prose chunk (default: 512)
      --overlap N               Overlap chars between prose chunks (default: 64)
      --batch-size N            Embedding batch size (default: 64)
      --max-table-chars N       Max chars before a table is row-split (default: 1800)
      --table-rows-per-batch N  Rows per sub-table when splitting large tables (default: 20)
      --dry-run                 Parse + print stats without writing any files
      --promote                 After staging, copy outputs to production vectorstore
      --no-backup               Skip backup step when promoting (use with extreme care)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

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
log = logging.getLogger("ingest_tables_aware")

# ---------------------------------------------------------------------------
# Path constants
# (all resolved relative to the project root — the parent of scripts/)
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.config import EMBEDDING_DIM, EMBEDDING_MODEL

DEFAULT_SOURCE = _ROOT / "data" / "harrison.md"

# Staging output (default write target — never touches production)
STAGING_DIR    = _ROOT / "artifacts" / "vectorstore_staging"
STAGING_INDEX  = STAGING_DIR / "table_index.faiss"
STAGING_CHUNKS = STAGING_DIR / "table_chunks.json"

# Production paths — written ONLY when --promote is passed
PROD_DIR    = _ROOT / "artifacts" / "vectorstore"
PROD_INDEX  = PROD_DIR / "index.faiss"
PROD_CHUNKS = PROD_DIR / "chunks.json"

# Backup directory root — timestamped subdirs created per promotion
BACKUP_ROOT = _ROOT / "artifacts" / "vectorstore_backup"

# ---------------------------------------------------------------------------
# Chunking defaults
# ---------------------------------------------------------------------------

DEFAULT_CHUNK_SIZE           = 2800   # max chars per prose chunk (calibrated to production avg ~2806)
DEFAULT_OVERLAP              = 200    # overlap chars between prose chunks
DEFAULT_BATCH_SIZE           = 64    # SentenceTransformer encode() batch size
DEFAULT_MAX_TABLE_CHARS      = 1800  # chars above which a table is row-split
DEFAULT_TABLE_ROWS_PER_BATCH = 20   # rows per sub-table when row-splitting
DEFAULT_MIN_CHUNK_CHARS      = 20   # chunks shorter than this are silently dropped

# Maximum character distance between two consecutive [p:NNN] markers for them
# to be treated as DUPLICATES on the same page (i.e. count as only ONE tick).
# harrison.md has ~11 such back-to-back duplicate markers from the PDF renderer.
MARKER_DEDUP_DISTANCE        = 200  # chars

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Any Markdown heading line (captures level and text)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

# A Markdown table row: starts with optional whitespace, then |
_TABLE_ROW_RE = re.compile(r"^\s*\|.+\|?\s*$")

# A Markdown table separator row: |---|---| or |:--:|---:|
_TABLE_SEP_RE = re.compile(r"^\s*\|[\s\-:]+\|[\s\-:|]*$")

# Any [p:NNN] page marker (NNN can be any digits — we ignore the value and
# count occurrences sequentially because all markers in harrison.md are [p:1])
_PAGE_MARKER_RE = re.compile(r"\[p:\d+\]")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    """
    One unit of text ready for embedding and FAISS insertion.

    Fields
    ------
    chunk_id       : Monotonically increasing integer (used internally only;
                     NOT written to the JSON registry — the list position
                     serves as the chunk_id for rag.py compatibility).
    page           : Sequential page number derived from [p:NNN] marker order.
                     Matches the production range 1–4273.
    text           : The full text that will be embedded and returned to LLM.
    source_heading : Nearest ancestor Markdown heading (audit field only).
    chunk_type     : ``"table"`` | ``"prose"`` (audit field only).
    char_count     : Derived character count (internal, not serialised).
    """

    chunk_id:       int
    page:           int
    text:           str
    source_heading: str
    chunk_type:     str    # "table" | "prose"
    char_count:     int = field(init=False)

    def __post_init__(self) -> None:
        self.char_count = len(self.text)

    def to_dict(self) -> dict:
        """
        Serialise to the PRODUCTION schema: {page, text} only.

        rag.py accesses chunks via:
            chunk.get("page")
            chunk.get("text", "")
        and uses the list index as the chunk_id — no stored chunk_id needed.

        The audit fields (source_heading, chunk_type) are included as extra
        keys; rag.py ignores unknown keys via .get(), so they do not break
        the retrieval pipeline but remain available for debugging tools.
        """
        return {
            "page":           self.page,
            "text":           self.text,
            "source_heading": self.source_heading,
            "chunk_type":     self.chunk_type,
        }


# ---------------------------------------------------------------------------
# Page marker helpers
# ---------------------------------------------------------------------------

def count_markers_in(text: str) -> int:
    """
    Return the raw count of ``[p:NNN]`` markers in ``text``.

    Used only for pre-scan reporting and for the preamble / section-level
    header scan.  The actual per-chunk page advancement is handled inside
    ``build_chunks()`` via a globally-deduplicated iterator.
    """
    return len(_PAGE_MARKER_RE.findall(text))


def build_global_page_offsets(md: str) -> List[int]:
    """
    Pre-scan the full Markdown document and return a sorted list of character
    offsets for each **unique** page boundary.

    Deduplication
    -------------
    pymupdf4llm occasionally emits two consecutive ``[p:1]`` markers within
    ``MARKER_DEDUP_DISTANCE`` characters (the same PDF page boundary rendered
    twice, e.g. at the join between a header/footer and the page body).  We
    suppress the second occurrence so the page ordinal is incremented only once
    per physical page.

    The current harrison.md source has ~11 such duplicate pairs, which would
    otherwise produce a spurious page range of 1–4273 instead of 1–4262.

    Returns
    -------
    List[int]
        Character start positions of each unique page marker, in document order.
        ``len(result)`` equals the total number of logical pages in the document.
    """
    all_matches = list(_PAGE_MARKER_RE.finditer(md))
    if not all_matches:
        return []

    unique: List[int] = [all_matches[0].start()]
    for m in all_matches[1:]:
        if m.start() - unique[-1] > MARKER_DEDUP_DISTANCE:
            unique.append(m.start())

    return unique


def build_page_map(md: str) -> np.ndarray:
    """
    Build a character-level page-number array for the full Markdown document.

    Returns an ``int32`` numpy array of length ``len(md)`` where
    ``page_map[i]`` is the sequential page number that character ``md[i]``
    belongs to.  Page numbers start at 1 at the first unique ``[p:NNN]``
    marker; positions before the first marker get page 0 (front-matter).

    Strategy
    --------
    1. ``build_global_page_offsets`` identifies the character positions of all
       **unique** page boundaries (duplicates within MARKER_DEDUP_DISTANCE
       are suppressed globally).
    2. We fill the array in a single linear pass: starting from each boundary
       offset, all chars from that position onward are assigned the next page
       number.  Because numpy slice assignment is O(k) per boundary and there
       are ~4 262 boundaries, the total cost is O(N) in document length.

    The resulting array allows ``build_chunks`` to look up any chunk's page
    number in O(1) with ``int(page_map[char_offset])``.
    """
    unique_offsets = build_global_page_offsets(md)
    page_map = np.zeros(len(md), dtype=np.int32)

    for page_number, offset in enumerate(unique_offsets, start=1):
        page_map[offset:] = page_number

    return page_map


# ---------------------------------------------------------------------------
# Section splitter
# ---------------------------------------------------------------------------

def split_into_sections(md: str) -> List[dict]:
    """
    Split a Markdown document into logical sections at every heading boundary.

    Returns a list of dicts, each with:
      ``heading``  : The heading text that opens this section.
      ``level``    : Heading depth (1-6).
      ``body``     : All text between this heading and the next heading of
                     equal or higher level (lower number = higher level).
      ``marker_count`` : Number of [p:NNN] markers in heading + first 200 chars
                         of body (used by build_chunks to advance page counter).
    """
    sections: List[dict] = []
    lines = md.splitlines(keepends=True)

    heading_positions: List[Tuple[int, int, str]] = []   # (line_idx, level, text)
    for i, line in enumerate(lines):
        m = _HEADING_RE.match(line.rstrip())
        if m:
            heading_positions.append((i, len(m.group(1)), m.group(2).strip()))

    if not heading_positions:
        # No headings at all — treat the whole document as one section.
        sections.append({
            "heading":      "(no heading)",
            "level":        0,
            "body":         md,
            "marker_count": count_markers_in(md),
        })
        return sections

    # Pre-heading preamble (e.g. the very first [p:1] markers)
    first_heading_line = heading_positions[0][0]
    if first_heading_line > 0:
        preamble = "".join(lines[:first_heading_line])
        if preamble.strip():
            sections.append({
                "heading":      "(preamble)",
                "level":        0,
                "body":         preamble,
                "marker_count": count_markers_in(preamble),
            })

    for idx, (line_i, level, heading_text) in enumerate(heading_positions):
        if idx + 1 < len(heading_positions):
            next_line_i = heading_positions[idx + 1][0]
        else:
            next_line_i = len(lines)

        body = "".join(lines[line_i + 1 : next_line_i])

        # Count ALL markers in heading + body so build_chunks can advance the
        # page counter without re-scanning the full text.
        total_text = heading_text + "\n" + body
        sections.append({
            "heading":      heading_text,
            "level":        level,
            "body":         body,
            "marker_count": count_markers_in(total_text),
        })

    return sections


# ---------------------------------------------------------------------------
# Table detector / extractor
# ---------------------------------------------------------------------------

def _is_table_row(line: str) -> bool:
    return bool(_TABLE_ROW_RE.match(line))


def _is_separator_row(line: str) -> bool:
    return bool(_TABLE_SEP_RE.match(line))


def split_large_table(
    table_text: str,
    max_chars: int,
    rows_per_batch: int,
) -> List[str]:
    """
    Split an oversized Markdown table into sub-tables, each prefixed with
    the original header row and separator row.

    Parameters
    ----------
    table_text   : The full Markdown table string (header + sep + data rows).
    max_chars    : Character limit that triggered the split (used for logging).
    rows_per_batch : Maximum data rows per sub-table.

    Returns
    -------
    List of Markdown table strings, each self-contained with its own header.
    If the table cannot be parsed (< 3 lines), the original string is returned
    as a single-element list.

    Example
    -------
    A 60-row table with rows_per_batch=20 → 3 sub-tables of 20 rows each,
    every sub-table starting with: header_row \\n sep_row \\n data_rows...
    """
    lines = table_text.strip().splitlines()
    if len(lines) < 3:
        # Too short to identify header + separator; return as-is
        log.debug("split_large_table: fewer than 3 lines, returning as-is.")
        return [table_text]

    header   = lines[0]
    sep_row  = lines[1]
    data_rows = lines[2:]

    if not data_rows:
        return [table_text]

    batches: List[str] = []
    for i in range(0, len(data_rows), rows_per_batch):
        batch_rows = data_rows[i : i + rows_per_batch]
        sub_table = "\n".join([header, sep_row] + batch_rows)
        batches.append(sub_table)

    log.debug(
        "split_large_table: %d chars → %d sub-tables (%d rows/batch).",
        len(table_text), len(batches), rows_per_batch,
    )
    return batches if batches else [table_text]


def extract_table_blocks(
    body: str,
    max_table_chars: int,
    table_rows_per_batch: int,
) -> List[dict]:
    """
    Scan a section body and return an ordered list of blocks:

      ``kind`` : ``"table"`` | ``"prose"``
      ``text`` : The raw text of this block.

    Tables are identified by at least one separator row (``|---|---|``).
    The block includes all consecutive lines that form part of the table,
    plus any immediately-preceding caption line.

    When a table block exceeds ``max_table_chars``, ``split_large_table()``
    is invoked; the single large block is replaced by multiple smaller
    table blocks, each containing the header row for context.
    """
    lines = body.splitlines(keepends=True)
    blocks: List[dict] = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Detect start of a Markdown table: look-ahead for a separator row
        # within the next 5 lines (header row + separator is minimum 2 lines).
        if _is_table_row(line):
            lookahead = lines[i : i + 5]
            has_sep = any(_is_separator_row(l) for l in lookahead)

            if has_sep:
                table_lines: List[str] = []

                # ── Caption absorption ──────────────────────────────────
                # If the immediately preceding block is prose, pull its last
                # non-blank line into the table as a caption.
                if blocks and blocks[-1]["kind"] == "prose":
                    prose_lines = blocks[-1]["text"].rstrip().splitlines(keepends=True)
                    if prose_lines:
                        caption = prose_lines[-1]
                        # Only absorb if it looks like a caption (short, not a row)
                        if len(caption.strip()) < 120 and not _is_table_row(caption):
                            table_lines.append(caption)
                            blocks[-1]["text"] = "".join(prose_lines[:-1])
                            if not blocks[-1]["text"].strip():
                                blocks.pop()

                # ── Collect all consecutive table rows ───────────────────
                while i < len(lines) and _is_table_row(lines[i]):
                    table_lines.append(lines[i])
                    i += 1

                raw_table = "".join(table_lines)

                # ── Large-table row-split fallback ───────────────────────
                if len(raw_table) > max_table_chars:
                    log.debug(
                        "Large table detected (%d chars > %d limit); splitting into row batches.",
                        len(raw_table), max_table_chars,
                    )
                    sub_tables = split_large_table(raw_table, max_table_chars, table_rows_per_batch)
                    for sub in sub_tables:
                        blocks.append({"kind": "table", "text": sub})
                else:
                    blocks.append({"kind": "table", "text": raw_table})

                continue  # already advanced i past table rows

        # ── Non-table line: accumulate as prose ──────────────────────────
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
    characters.  Splits prefer sentence boundaries (". ", "! ", "? ",
    "\\n\\n", "\\n") or fall back to hard character splits.
    """
    text = text.strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            # Prefer a clean sentence boundary in the last 20% of the window
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

def split_into_sections_with_offsets(md: str) -> List[dict]:
    """
    Split a Markdown document into sections AND record the absolute character
    offset of each section's body within ``md``.

    Returns a list of dicts with all keys from ``split_into_sections`` plus:
      ``body_start`` : Absolute char offset of ``body[0]`` in ``md``.

    This lets ``build_chunks`` look up exact page numbers from a pre-built
    page_map array without any string searching.
    """
    sections: List[dict] = []
    lines = md.splitlines(keepends=True)

    # Build cumulative line start offsets for O(1) line→offset mapping
    line_starts: List[int] = []
    pos = 0
    for line in lines:
        line_starts.append(pos)
        pos += len(line)

    heading_positions: List[Tuple[int, int, str]] = []
    for i, line in enumerate(lines):
        m = _HEADING_RE.match(line.rstrip())
        if m:
            heading_positions.append((i, len(m.group(1)), m.group(2).strip()))

    if not heading_positions:
        sections.append({
            "heading":    "(no heading)",
            "level":      0,
            "body":       md,
            "body_start": 0,
        })
        return sections

    # Pre-heading preamble
    first_heading_line = heading_positions[0][0]
    if first_heading_line > 0:
        preamble = "".join(lines[:first_heading_line])
        if preamble.strip():
            sections.append({
                "heading":    "(preamble)",
                "level":      0,
                "body":       preamble,
                "body_start": 0,
            })

    for idx, (line_i, level, heading_text) in enumerate(heading_positions):
        if idx + 1 < len(heading_positions):
            next_line_i = heading_positions[idx + 1][0]
        else:
            next_line_i = len(lines)

        body = "".join(lines[line_i + 1 : next_line_i])
        # body starts at the line AFTER the heading line
        body_start = line_starts[line_i + 1] if line_i + 1 < len(lines) else len(md)

        sections.append({
            "heading":    heading_text,
            "level":      level,
            "body":       body,
            "body_start": body_start,
        })

    return sections


def build_chunks(
    md: str,
    chunk_size:           int = DEFAULT_CHUNK_SIZE,
    overlap:              int = DEFAULT_OVERLAP,
    max_table_chars:      int = DEFAULT_MAX_TABLE_CHARS,
    table_rows_per_batch: int = DEFAULT_TABLE_ROWS_PER_BATCH,
    min_chunk_chars:      int = DEFAULT_MIN_CHUNK_CHARS,
) -> List[Chunk]:
    """
    Full table-aware chunking pipeline.

    Pipeline
    --------
    1. Build a character-level page_map array from the full document.
    2. Split document into sections at heading boundaries (with body offsets).
    3. Within each section, separate table blocks from prose blocks.
    4. Table blocks → one Chunk per table (with heading prefix), or multiple
       Chunks if the table was row-split due to exceeding max_table_chars.
    5. Prose blocks → one or more overlap-split Chunks.
    6. Each chunk's page number is read from page_map at its body_start offset.

    Page Assignment via page_map
    ----------------------------
    ``build_page_map`` pre-scans the full document in O(N) time, building a
    numpy int32 array where ``page_map[i]`` = the sequential page number of
    character ``i``.  Deduplication of back-to-back ``[p:1]`` markers is
    handled globally and exactly by ``build_global_page_offsets``.

    For each chunk, we look up ``int(page_map[body_start])`` — the page
    number at the START of the block's body in the original document.  This
    is O(1), perfectly accurate, and completely immune to section-boundary
    dedup issues.

    Returns
    -------
    List[Chunk]
        Ready-to-embed chunks with correct sequential page numbers.
    """
    # ── Step 1: Build page_map (one pre-scan, all lookups are O(1)) ──────
    log.debug("Building character-level page_map…")
    page_map = build_page_map(md)   # shape: (len(md),), dtype: int32

    # ── Step 2: Split into sections with absolute body offsets ───────────
    sections = split_into_sections_with_offsets(md)
    log.info("Document split into %d sections.", len(sections))

    all_chunks:    List[Chunk] = []
    chunk_counter: int = 0
    table_count:   int = 0
    prose_count:   int = 0

    for sec in sections:
        heading    = sec["heading"]
        body       = sec["body"]
        body_start = sec["body_start"]   # absolute offset of body[0] in md

        # ── Extract table and prose blocks ───────────────────────────────
        blocks = extract_table_blocks(body, max_table_chars, table_rows_per_batch)

        body_cursor = 0   # local offset within body

        for block in blocks:
            # Absolute offset of this block's first character in md
            block_abs_start = body_start + body_cursor

            if block["kind"] == "table":
                # ── Table chunk ──────────────────────────────────────────
                # Page = page_map value at the block's absolute start
                page = int(page_map[min(block_abs_start, len(page_map) - 1)])
                table_text = f"## {heading}\n\n{block['text'].strip()}"

                all_chunks.append(Chunk(
                    chunk_id=chunk_counter,
                    page=page,
                    text=table_text,
                    source_heading=heading,
                    chunk_type="table",
                ))
                chunk_counter += 1
                table_count   += 1

            else:
                # ── Prose chunks ─────────────────────────────────────────
                prose_parts = chunk_prose(block["text"], chunk_size, overlap)
                local_cursor = 0
                for part in prose_parts:
                    stripped = part.strip()
                    if not stripped:
                        local_cursor += len(part)
                        continue
                    if len(stripped) < min_chunk_chars:
                        log.debug(
                            "Dropped stub prose chunk (%d chars): %r",
                            len(stripped), stripped[:40],
                        )
                        local_cursor += len(part)
                        continue
                    # Find the absolute offset of this prose sub-chunk
                    part_abs_start = block_abs_start + local_cursor
                    page = int(page_map[min(part_abs_start, len(page_map) - 1)])
                    all_chunks.append(Chunk(
                        chunk_id=chunk_counter,
                        page=page,
                        text=stripped,
                        source_heading=heading,
                        chunk_type="prose",
                    ))
                    chunk_counter += 1
                    prose_count   += 1
                    local_cursor  += len(part)

            body_cursor += len(block["text"])

    # ── Summary statistics ───────────────────────────────────────────────
    zero_page = sum(1 for c in all_chunks if c.page == 0)
    log.info(
        "Chunking complete: %d total (%d tables, %d prose). "
        "page=0 count: %d (front-matter before first [p:NNN] marker).",
        len(all_chunks), table_count, prose_count, zero_page,
    )
    if zero_page:
        log.info(
            "  → %d page=0 chunks are expected front-matter / title / TOC content.",
            zero_page,
        )

    return all_chunks


# ---------------------------------------------------------------------------
# Embedding pipeline
# ---------------------------------------------------------------------------

def load_embedding_model(device: Optional[str] = None) -> SentenceTransformer:
    """
    Load the SentenceTransformer model, preferring Apple Silicon MPS when
    available and falling back to CPU on other hardware.

    Parameters
    ----------
    device : Optional override.  Pass ``"cpu"`` to force CPU regardless of
             hardware.

    Returns
    -------
    A loaded SentenceTransformer instance ready for encode().
    """
    if device is None:
        device = "cpu"
    log.info(
        "Loading SentenceTransformer('%s') on device='%s'…",
        EMBEDDING_MODEL, device,
    )
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    log.info("Model loaded.  Device: %s", device.upper())
    return model


def embed_chunks(
    chunks: List[Chunk],
    model: SentenceTransformer,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    """
    Generate L2-normalised embeddings for all chunks.

    Returns an ``(N, EMBEDDING_DIM)`` float32 numpy array suitable for direct
    insertion into a FAISS ``IndexFlatL2`` index.
    """
    texts = [c.text for c in chunks]
    log.info("Embedding %d chunks in batches of %d…", len(texts), batch_size)

    t0 = time.perf_counter()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalise to match the live pipeline
    )
    elapsed = time.perf_counter() - t0
    log.info(
        "Embedding complete in %.1f s  (%.2f chunks/s).",
        elapsed, len(texts) / elapsed if elapsed > 0 else float("inf"),
    )
    return embeddings.astype("float32")


# ---------------------------------------------------------------------------
# FAISS index builder
# ---------------------------------------------------------------------------

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """
    Build a flat L2 FAISS index from the embedding matrix.

    ``IndexFlatL2`` provides exact nearest-neighbour search — appropriate for
    a textbook-sized corpus (~12 000–15 000 chunks) where query accuracy is
    more important than indexing speed.

    The index type matches ``backend/retrieval/rag.py`` exactly.
    """
    n, dim = embeddings.shape
    log.info("Building FAISS IndexFlatL2 (dim=%d, n=%d)…", dim, n)
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    log.info("FAISS index contains %d vectors.", index.ntotal)
    return index


# ---------------------------------------------------------------------------
# Persistence: backup, staging write, and production promotion
# ---------------------------------------------------------------------------

def backup_production() -> Optional[Path]:
    """
    Copy the existing production ``index.faiss`` and ``chunks.json`` to a
    timestamped backup directory under ``artifacts/vectorstore_backup/``.

    Returns the backup directory path, or ``None`` if there was nothing to
    back up (e.g. first-time ingestion with no existing production files).
    """
    if not PROD_INDEX.exists() and not PROD_CHUNKS.exists():
        log.info("No existing production files to back up.")
        return None

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_dir = BACKUP_ROOT / ts
    backup_dir.mkdir(parents=True, exist_ok=True)

    if PROD_INDEX.exists():
        shutil.copy2(PROD_INDEX, backup_dir / "index.faiss")
        log.info("Backed up production index  → %s", backup_dir / "index.faiss")

    if PROD_CHUNKS.exists():
        shutil.copy2(PROD_CHUNKS, backup_dir / "chunks.json")
        log.info("Backed up production chunks → %s", backup_dir / "chunks.json")

    return backup_dir


def promote_only(no_backup: bool = False) -> None:
    """
    Promote existing staging outputs directly to production without
    rebuilding, re-embedding, or loading any models.
    """
    if not STAGING_INDEX.exists():
        log.error("Staging index missing: %s", STAGING_INDEX)
        sys.exit(1)
    if not STAGING_CHUNKS.exists():
        log.error("Staging chunks missing: %s", STAGING_CHUNKS)
        sys.exit(1)

    log.info("=" * 60)
    log.info("PROMOTE-ONLY: copying staging outputs → production vectorstore…")

    if not no_backup:
        backup_dir = backup_production()
        if backup_dir:
            log.info("Production backup stored at: %s", backup_dir)
    else:
        log.warning(
            "--no-backup specified: existing production files will be "
            "OVERWRITTEN without a backup."
        )

    PROD_DIR.mkdir(parents=True, exist_ok=True)

    shutil.copy2(STAGING_INDEX, PROD_INDEX)
    log.info("Promoted FAISS index  → %s  (%d bytes)", PROD_INDEX, PROD_INDEX.stat().st_size)

    shutil.copy2(STAGING_CHUNKS, PROD_CHUNKS)
    log.info("Promoted chunk registry → %s  (%d bytes)", PROD_CHUNKS, PROD_CHUNKS.stat().st_size)

    # Post-promotion safety checks
    if not PROD_INDEX.exists():
        raise RuntimeError(f"FATAL: production index missing after promotion: {PROD_INDEX}")
    if not PROD_CHUNKS.exists():
        raise RuntimeError(f"FATAL: production chunks missing after promotion: {PROD_CHUNKS}")

    log.info("=" * 60)
    log.warning(
        "⚠️  IMPORTANT: The running uvicorn server may still have the old\n"
        "    index and chunks in memory and should be restarted after promotion\n"
        "    to activate the new table-aware index.\n\n"
        "    To restart, kill the current uvicorn process, then run:\n"
        "        .venv312/bin/python -m uvicorn backend.api.main:app "
        "--reload --host 127.0.0.1 --port 8000\n"
        "    Then verify: curl http://127.0.0.1:8000/health"
    )


def save_outputs(
    index:   faiss.IndexFlatL2,
    chunks:  List[Chunk],
    promote: bool = False,
    no_backup: bool = False,
    dry_run: bool = False,
) -> None:
    """
    Write the FAISS index and chunk registry to disk.

    Write flow
    ----------
    **Default (promote=False):**
        Writes to ``artifacts/vectorstore_staging/``.
        Production is never touched.

    **With promote=True:**
        1. (Unless no_backup) Backs up existing production files.
        2. Writes staging outputs.
        3. Copies staging → production:
               staging/table_index.faiss  →  vectorstore/index.faiss
               staging/table_chunks.json  →  vectorstore/chunks.json
        4. Emits a reminder to restart the uvicorn server.

    Safety assertions
    -----------------
    Verifies that the staging directory path never resolves to the production
    path.  Hard-aborts if it does (should never happen, but belt-and-suspenders).
    """
    assert STAGING_DIR.resolve() != PROD_DIR.resolve(), (
        "FATAL: staging path resolves to the production directory — aborting."
    )

    if dry_run:
        log.info("[dry-run] Would write:")
        log.info("  %s  (%d vectors)", STAGING_INDEX, index.ntotal if index else 0)
        log.info("  %s  (%d chunks)", STAGING_CHUNKS, len(chunks))
        if promote:
            log.info("  [--promote] Would then copy staging → %s", PROD_DIR)
        return

    # ── Write staging outputs ────────────────────────────────────────────
    STAGING_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(STAGING_INDEX))
    log.info(
        "FAISS index saved → %s  (%d bytes)",
        STAGING_INDEX, STAGING_INDEX.stat().st_size,
    )

    registry = [c.to_dict() for c in chunks]
    with open(STAGING_CHUNKS, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
    log.info(
        "Chunk registry saved → %s  (%d bytes)",
        STAGING_CHUNKS, STAGING_CHUNKS.stat().st_size,
    )

    if not promote:
        return

    # ── Promotion to production ──────────────────────────────────────────
    log.info("=" * 60)
    log.info("PROMOTE: copying staging outputs → production vectorstore…")

    if not no_backup:
        backup_dir = backup_production()
        if backup_dir:
            log.info("Production backup stored at: %s", backup_dir)
    else:
        log.warning(
            "--no-backup specified: existing production files will be "
            "OVERWRITTEN without a backup."
        )

    PROD_DIR.mkdir(parents=True, exist_ok=True)

    shutil.copy2(STAGING_INDEX, PROD_INDEX)
    log.info("Promoted FAISS index  → %s  (%d bytes)", PROD_INDEX, PROD_INDEX.stat().st_size)

    shutil.copy2(STAGING_CHUNKS, PROD_CHUNKS)
    log.info("Promoted chunk registry → %s  (%d bytes)", PROD_CHUNKS, PROD_CHUNKS.stat().st_size)

    # Post-promotion safety checks
    if not PROD_INDEX.exists():
        raise RuntimeError(f"FATAL: production index missing after promotion: {PROD_INDEX}")
    if not PROD_CHUNKS.exists():
        raise RuntimeError(f"FATAL: production chunks missing after promotion: {PROD_CHUNKS}")

    log.info("=" * 60)
    log.warning(
        "⚠️  IMPORTANT: The running uvicorn server may still have the old\n"
        "    index and chunks in memory and should be restarted after promotion\n"
        "    to activate the new table-aware index.\n\n"
        "    To restart, kill the current uvicorn process, then run:\n"
        "        .venv312/bin/python -m uvicorn backend.api.main:app "
        "--reload --host 127.0.0.1 --port 8000\n"
        "    Then verify: curl http://127.0.0.1:8000/health"
    )


# ---------------------------------------------------------------------------
# Source file loader (file OR directory)
# ---------------------------------------------------------------------------

def load_source(source: Path) -> str:
    """
    Load Markdown source text from a file or a directory.

    When ``source`` is a directory, all ``*.md`` files are discovered
    recursively (sorted by path for deterministic ordering) and concatenated
    with a blank line between each file.

    Returns the full Markdown string.
    """
    if source.is_file():
        log.info("Reading source file: %s  (%.1f MB)", source, source.stat().st_size / 1e6)
        return source.read_text(encoding="utf-8")

    if source.is_dir():
        md_files = sorted(source.rglob("*.md"))
        if not md_files:
            log.error("No .md files found in directory: %s", source)
            sys.exit(1)
        log.info(
            "Source directory: %s  (%d .md file(s) found)", source, len(md_files)
        )
        parts: List[str] = []
        for f in md_files:
            log.info("  Including: %s  (%.1f MB)", f.name, f.stat().st_size / 1e6)
            parts.append(f.read_text(encoding="utf-8"))
        return "\n\n".join(parts)

    log.error("Source not found (not a file or directory): %s", source)
    sys.exit(1)


# ---------------------------------------------------------------------------
# CLI argument parser
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
        metavar="PATH",
        help=(
            "Path to the Markdown source file OR a directory containing .md "
            f"files (default: data/harrison.md)"
        ),
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        metavar="N",
        help=(
            f"Max characters per prose chunk (default: {DEFAULT_CHUNK_SIZE}).  "
            f"Calibrated to match the production vectorstore's avg chunk size "
            f"of ~2806 chars."
        ),
    )
    p.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        metavar="N",
        help=f"Overlap characters between prose chunks (default: {DEFAULT_OVERLAP})",
    )
    p.add_argument(
        "--min-chunk-chars",
        type=int,
        default=DEFAULT_MIN_CHUNK_CHARS,
        metavar="N",
        help=(
            f"Minimum character length for a prose chunk to be kept.  "
            f"Shorter chunks (lone page markers, stray newlines) are dropped.  "
            f"(default: {DEFAULT_MIN_CHUNK_CHARS})"
        ),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        metavar="N",
        help=f"Embedding batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    p.add_argument(
        "--max-table-chars",
        type=int,
        default=DEFAULT_MAX_TABLE_CHARS,
        metavar="N",
        help=(
            f"Character threshold above which a Markdown table is split into "
            f"row batches (default: {DEFAULT_MAX_TABLE_CHARS}).  Tune this "
            f"to stay within the embedding model's token window "
            f"(~1 char ≈ 0.28 tokens)."
        ),
    )
    p.add_argument(
        "--table-rows-per-batch",
        type=int,
        default=DEFAULT_TABLE_ROWS_PER_BATCH,
        metavar="N",
        help=(
            f"Maximum data rows per sub-table when a large table is row-split "
            f"(default: {DEFAULT_TABLE_ROWS_PER_BATCH})"
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and print statistics without writing any output files.",
    )
    p.add_argument(
        "--promote",
        action="store_true",
        help=(
            "After writing staging outputs, promote them to the production "
            "vectorstore (artifacts/vectorstore/).  A timestamped backup of "
            "the existing production files is created automatically unless "
            "--no-backup is also passed."
        ),
    )
    p.add_argument(
        "--no-backup",
        action="store_true",
        help=(
            "Skip the automatic production backup when --promote is used.  "
            "Use with extreme caution — the existing index will be "
            "OVERWRITTEN without a recovery point."
        ),
    )
    p.add_argument(
        "--promote-only",
        action="store_true",
        help=(
            "Promote existing staging outputs directly to production without "
            "rebuilding, re-embedding, or loading any models."
        ),
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ── 0. Guard: staging and production must be different paths ─────────
    assert STAGING_INDEX.resolve() != PROD_INDEX.resolve(), (
        "FATAL: staging index path resolves to the production index — aborting."
    )
    assert STAGING_CHUNKS.resolve() != PROD_CHUNKS.resolve(), (
        "FATAL: staging chunks path resolves to the production chunks — aborting."
    )

    if args.promote_only:
        promote_only(no_backup=args.no_backup)
        return

    # ── 1. Load source text ───────────────────────────────────────────────
    source = args.source.resolve()
    md = load_source(source)
    total_markers = count_markers_in(md)
    unique_pages  = len(build_global_page_offsets(md))
    log.info(
        "Source loaded: %.1f MB  |  raw [p:NNN] markers: %d  "
        "|  unique pages (after dedup): %d  "
        "|  expected page range: 1–%d",
        len(md) / 1e6, total_markers, unique_pages, unique_pages,
    )

    # ── 2. Build table-aware chunks ───────────────────────────────────────
    log.info("-" * 60)
    log.info("Building table-aware chunks…")
    chunks = build_chunks(
        md,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        max_table_chars=args.max_table_chars,
        table_rows_per_batch=args.table_rows_per_batch,
        min_chunk_chars=args.min_chunk_chars,
    )
    if not chunks:
        log.error("No chunks produced — check that the source file has content.")
        sys.exit(1)

    # ── 3. Print chunk statistics ─────────────────────────────────────────
    table_chunks = [c for c in chunks if c.chunk_type == "table"]
    prose_chunks = [c for c in chunks if c.chunk_type == "prose"]
    avg_len      = sum(c.char_count for c in chunks) / len(chunks)
    max_len      = max(c.char_count for c in chunks)
    min_len      = min(c.char_count for c in chunks)

    # Detect how many table chunks look like sub-tables from a split
    # (multiple consecutive table chunks with the same heading)
    log.info("=" * 60)
    log.info("CHUNK STATISTICS")
    log.info("  Total chunks     : %d", len(chunks))
    log.info("  Table chunks     : %d  (intact Markdown tables)", len(table_chunks))
    log.info("  Prose chunks     : %d", len(prose_chunks))
    log.info("  Avg chunk length : %.0f chars", avg_len)
    log.info("  Min chunk length : %d chars", min_len)
    log.info("  Max chunk length : %d chars", max_len)
    log.info("  Page range       : %d – %d",
             min(c.page for c in chunks), max(c.page for c in chunks))
    log.info("=" * 60)

    if args.dry_run:
        # Print sample chunks and exit without writing files
        log.info("[dry-run] Sample TABLE chunk:")
        if table_chunks:
            print("\n" + "─" * 60)
            print(table_chunks[0].text[:800])
            print("─" * 60 + "\n")
        log.info("[dry-run] Sample PROSE chunk:")
        if prose_chunks:
            print("\n" + "─" * 60)
            print(prose_chunks[0].text[:400])
            print("─" * 60 + "\n")

        # Schema preview
        log.info("[dry-run] Output schema preview (first chunk):")
        print(json.dumps(chunks[0].to_dict(), indent=2, ensure_ascii=False))

        save_outputs(None, chunks, promote=False, dry_run=True)  # type: ignore[arg-type]
        log.info("[dry-run] Complete — no files written.")
        return

    # ── 4. Load embedding model (MPS on Apple Silicon, else CPU) ─────────
    log.info("-" * 60)
    model = load_embedding_model()

    # ── 5. Generate embeddings ────────────────────────────────────────────
    log.info("-" * 60)
    embeddings = embed_chunks(chunks, model, batch_size=args.batch_size)
    assert embeddings.shape == (len(chunks), EMBEDDING_DIM), (
        f"Unexpected embedding shape: {embeddings.shape}  "
        f"(expected ({len(chunks)}, {EMBEDDING_DIM}))"
    )

    # ── 6. Build FAISS index ──────────────────────────────────────────────
    log.info("-" * 60)
    index = build_faiss_index(embeddings)

    # ── 7. Save outputs (staging, and optionally promote) ─────────────────
    log.info("-" * 60)
    save_outputs(
        index,
        chunks,
        promote=args.promote,
        no_backup=args.no_backup,
        dry_run=False,
    )

    # ── 8. Final success summary ──────────────────────────────────────────
    log.info("=" * 60)
    log.info("✅  Ingestion complete.")
    log.info("   Staging index   : %s", STAGING_INDEX)
    log.info("   Staging chunks  : %s", STAGING_CHUNKS)
    log.info("   Vectors stored  : %d", index.ntotal)
    log.info("   Table chunks    : %d  (never split mid-row)", len(table_chunks))
    log.info("   Prose chunks    : %d", len(prose_chunks))
    log.info("   Page range      : %d – %d",
             min(c.page for c in chunks), max(c.page for c in chunks))

    if not args.promote:
        log.info("=" * 60)
        log.info(
            "ℹ️   Production index at artifacts/vectorstore/ is UNCHANGED.\n"
            "    To validate, load the staging index and run test queries.\n"
            "    When satisfied, run with --promote to go live:\n"
            "        python scripts/ingest_tables_aware.py --promote"
        )
    log.info("=" * 60)


if __name__ == "__main__":
    main()
