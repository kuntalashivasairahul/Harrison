# backend/rendering/page_resolver.py
"""
Page Rendering Subsystem — URL Resolver
========================================
Maps the ``sources`` list produced by ``extract_sources()`` (e.g. ``["p.142",
"p.512"]``) to a pair of image URLs per page:

- **thumbnail_url** → ``/pages/small/page_142_small.webp``
  Low-resolution preview; suitable for a sidebar or hover card.
- **full_url**       → ``/pages/full/page_142_full.png``
  Full-resolution render; suitable for a modal/lightbox.

Both paths are served by the FastAPI StaticFiles mount at ``/pages`` (pointing
at ``storage/pages/`` on disk).  This module is purely a URL constructor — it
does **not** check whether the image files exist on disk.  Missing images will
simply return a 404 from FastAPI when the frontend requests them.

URL format
----------
The page number is extracted from the ``"p.{N}"`` label using a simple regex.
Non-numeric or malformed labels are silently skipped and produce no entry in
the output list.

Usage
-----
    from backend.rendering.page_resolver import resolve_page_urls

    visual_context = resolve_page_urls(
        sources=["p.142", "p.512"],
        base_url="http://127.0.0.1:8000",
    )
    # [
    #   {
    #     "page_label": "p.142",
    #     "thumbnail_url": "http://127.0.0.1:8000/pages/small/page_142_small.webp",
    #     "full_url":      "http://127.0.0.1:8000/pages/full/page_142_full.png",
    #   },
    #   ...
    # ]
"""

from __future__ import annotations

import re
from typing import Dict, List

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

        page_num = match.group(1)  # the raw integer string, e.g. "142"

        result.append(
            {
                "page_label": label,
                "thumbnail_url": f"{base}/pages/small/page_{page_num}_small.webp",
                "full_url": f"{base}/pages/full/page_{page_num}_full.png",
            }
        )

    return result
