"""
tests/test_page_resolver.py
===========================
Tests for the page-image URL resolver.

Why this file matters: the resolver translates an index page number into a
rendered page-image filename using a single hardcoded constant,
``FAISS_TO_IMAGE_OFFSET``.  If that constant drifts, the API cites p.2787 and
serves the image of a different page — silently, with no error anywhere.  Until
now nothing tested it.
"""
from __future__ import annotations

import unittest

from backend.rendering import page_resolver
from backend.rendering.page_resolver import FAISS_TO_IMAGE_OFFSET, resolve_page_urls


class TestResolvePageUrls(unittest.TestCase):
    BASE = "http://127.0.0.1:8000"

    def test_offset_is_applied_to_urls_but_not_to_the_label(self):
        [entry] = resolve_page_urls(["p.2787"], self.BASE)

        self.assertEqual(entry["page_label"], "p.2787")
        expected = 2787 - FAISS_TO_IMAGE_OFFSET
        self.assertEqual(entry["thumbnail_url"], f"{self.BASE}/pages/small/page_{expected}_small.webp")
        self.assertEqual(entry["full_url"], f"{self.BASE}/pages/full/page_{expected}_full.png")

    def test_offset_constant_has_not_drifted(self):
        """Pin the constant. If a re-render changes it, this fails loudly
        instead of the API quietly serving the wrong page image."""
        self.assertEqual(FAISS_TO_IMAGE_OFFSET, 43)

    def test_label_and_image_page_stay_in_lockstep(self):
        for label_page in (100, 512, 1000, 2787):
            [entry] = resolve_page_urls([f"p.{label_page}"], self.BASE)
            image_page = int(entry["thumbnail_url"].split("page_")[1].split("_small")[0])
            self.assertEqual(label_page - image_page, FAISS_TO_IMAGE_OFFSET)

    def test_full_url_falls_back_to_the_thumbnail_when_full_renders_are_absent(self):
        """A free-tier deploy ships only storage/pages/small.

        Without this, full_url points at a PNG that was never deployed and the
        lightbox opens a 404.  The fallback must keep all three keys present:
        the QueryResponse field set is frozen (RULE 3.2), so dropping full_url
        would break the contract rather than degrade it.
        """
        [entry] = resolve_page_urls(["p.2787"], self.BASE, full_res_available=False)

        self.assertEqual(set(entry), {"page_label", "thumbnail_url", "full_url"})
        self.assertEqual(entry["full_url"], entry["thumbnail_url"])
        self.assertNotIn("/pages/full/", entry["full_url"])

    def test_full_res_defaults_to_available_so_existing_callers_are_unchanged(self):
        [with_default] = resolve_page_urls(["p.2787"], self.BASE)
        [explicit] = resolve_page_urls(["p.2787"], self.BASE, full_res_available=True)
        self.assertEqual(with_default, explicit)
        self.assertIn("/pages/full/", with_default["full_url"])

    def test_input_order_is_preserved(self):
        entries = resolve_page_urls(["p.900", "p.100", "p.500"], self.BASE)
        self.assertEqual([e["page_label"] for e in entries], ["p.900", "p.100", "p.500"])

    def test_malformed_labels_are_skipped(self):
        entries = resolve_page_urls(
            ["p.142", "142", "p.abc", "page 142", "", None, "p.", "p.1.2"], self.BASE
        )
        self.assertEqual([e["page_label"] for e in entries], ["p.142"])

    def test_underflow_is_clamped_to_page_one(self):
        [entry] = resolve_page_urls(["p.1"], self.BASE)
        self.assertIn("page_1_small.webp", entry["thumbnail_url"])

    def test_trailing_slash_on_base_url_does_not_double(self):
        [entry] = resolve_page_urls(["p.142"], "http://127.0.0.1:8000/")
        self.assertNotIn("//pages", entry["thumbnail_url"].replace("http://", ""))

    def test_https_and_custom_host_are_honoured(self):
        [entry] = resolve_page_urls(["p.142"], "https://harrison.example.com")
        self.assertTrue(entry["full_url"].startswith("https://harrison.example.com/pages/full/"))

    def test_empty_and_none_sources_return_empty_list(self):
        self.assertEqual(resolve_page_urls([], self.BASE), [])
        self.assertEqual(resolve_page_urls(None, self.BASE), [])

    def test_case_insensitive_label_prefix(self):
        [entry] = resolve_page_urls(["P.142"], self.BASE)
        self.assertEqual(entry["page_label"], "P.142")

    def test_every_entry_has_the_frozen_contract_keys(self):
        for entry in resolve_page_urls(["p.1", "p.2"], self.BASE):
            self.assertEqual(set(entry), {"page_label", "thumbnail_url", "full_url"})
            for value in entry.values():
                self.assertIsInstance(value, str)

    def test_resolver_does_no_disk_io(self):
        """Documented contract: URL construction only, never a file check."""
        source = page_resolver.__file__
        with open(source, encoding="utf-8") as handle:
            body = handle.read()
        for forbidden in ("open(", "Path(", "os.path", "exists("):
            self.assertNotIn(forbidden, body.split('"""', 2)[-1])


if __name__ == "__main__":
    unittest.main()
