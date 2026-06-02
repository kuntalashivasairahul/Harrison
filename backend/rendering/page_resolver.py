# backend/rendering/page_resolver.py
"""
Page Rendering Subsystem — URL Resolver
========================================
Maps the ``sources`` list produced by ``extract_sources()`` (e.g. ``["p.2787",
"p.512"]``) to a pair of image URLs per page:

- **thumbnail_url** → ``/pages/small/page_2744_small.webp``
  Low-resolution preview; suitable for a sidebar or hover card.
- **full_url**       → ``/pages/full/page_2744_full.png``
  Full-resolution render; suitable for a modal/lightbox.

Both paths are served by the FastAPI StaticFiles mount at ``/pages`` (pointing
at ``storage/pages/`` on disk).  This module is purely a URL constructor — it
does **not** check whether the image files exist on disk.  Missing images will
simply return a 404 from FastAPI when the frontend requests them.

Index Drift Correction
----------------------
The FAISS index stores **absolute PDF page numbers** (including front-matter).
The pre-rendered images in ``storage/pages/`` are numbered from the first
content page of the textbook.  The difference is exactly ``FAISS_TO_IMAGE_OFFSET``
pages of front-matter (title page, prefaces, table of contents, etc.).

This offset is applied **only to URL construction** — the ``page_label`` key
in the output always preserves the original FAISS page number so that
in-text citations match the labels shown in the UI.

Example (offset = 43)::

    FAISS label : "p.2787"   ← stored in chunks.json, cited in the answer
    Image file  : page_2744_small.webp   (2787 − 43 = 2744)

URL format
----------
The page number is extracted from the ``"p.{N}"`` label using a simple regex.
Non-numeric or malformed labels are silently skipped and produce no entry in
the output list.

Usage
-----
    from backend.rendering.page_resolver import resolve_page_urls

    visual_context = resolve_page_urls(
        sources=["p.2787", "p.512"],
        base_url="http://127.0.0.1:8000",
    )
    # [
    #   {
    #     "page_label": "p.2787",                                  ← unchanged
    #     "thumbnail_url": ".../pages/small/page_2744_small.webp", ← offset applied
    #     "full_url":      ".../pages/full/page_2744_full.png",    ← offset applied
    #   },
    #   ...
    # ]
"""

from __future__ import annotations

import re
from typing import Dict, List

# ---------------------------------------------------------------------------
# Index Drift Offset
# ---------------------------------------------------------------------------
# The FAISS index was built from the raw PDF, which includes 43 pages of
# front-matter before the first numbered textbook page.  The pre-rendered
# images in storage/pages/ are numbered from the first content page, so
# every image filename is exactly FAISS_TO_IMAGE_OFFSET less than the
# corresponding FAISS page number.
#
# To update this value: measure the difference between the PDF page that
# FAISS labels as "p.1" and the image file named "page_1_*.webp/png".
# That delta is the new offset.
FAISS_TO_IMAGE_OFFSET: int = 43

# Compiled once at import time for efficiency.
_PAGE_LABEL_RE = re.compile(r"^p\.(\d+)$", re.IGNORECASE)


def resolve_page_urls(
    sources: List[str],
    base_url: str,
) -> List[Dict[str, str]]:
    """
    Convert source page labels into image URL dictionaries.

    Parameters
    ----------
    sources:
        List of page labels as returned by ``extract_sources()``.
        Expected format: ``"p.{integer}"`` (e.g. ``"p.142"``).
        Labels that do not match this pattern are silently skipped.
    base_url:
        Scheme + host (+ optional port) of the API server, with **no**
        trailing slash.  Example: ``"http://127.0.0.1:8000"`` or
        ``"https://harrisonqpt.example.com"``.

    Returns
    -------
    List[Dict[str, str]]
        One dictionary per valid source label, preserving input order.
        Each dictionary has the keys:
        - ``page_label``    : original label string (e.g. ``"p.142"``)
        - ``thumbnail_url`` : absolute URL to the small WebP thumbnail
        - ``full_url``      : absolute URL to the full-resolution PNG
    """
    base = base_url.rstrip("/")
    result: List[Dict[str, str]] = []

    for label in (sources or []):
        match = _PAGE_LABEL_RE.match((label or "").strip())
        if not match:
            # Malformed or non-numeric label — skip gracefully.
            continue

        # faiss_page: absolute PDF page number stored in chunks.json.
        faiss_page: int = int(match.group(1))

        # actual_image_page: filename index used in storage/pages/.
        # Guard against underflow — clamp to 1 if the offset would produce
        # a non-positive page number (should not happen in practice).
        actual_image_page: int = max(1, faiss_page - FAISS_TO_IMAGE_OFFSET)

        result.append(
            {
                # page_label preserves the original FAISS citation number
                # so in-text references remain consistent with the UI labels.
                "page_label":    label,
                "thumbnail_url": f"{base}/pages/small/page_{actual_image_page}_small.webp",
                "full_url":      f"{base}/pages/full/page_{actual_image_page}_full.png",
            }
        )

    return result
