"""
tests/test_api_main.py
======================
Tests for the HTTP layer — previously the largest untested module in the repo
(496 lines, zero coverage, not one TestClient test).

Covers the pure helpers (cache signature, final_k selection, cache-save
eligibility), the confidence-cap rule table, the two early-exit paths, and the
frozen QueryResponse contract.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from tests._api_harness import import_main

main = import_main()


def _optimized(**overrides):
    payload = {
        "is_medical_query": True,
        "expanded_query": "acute pancreatitis diagnosis and management",
        "focus": "management",
        "complexity": "complex",
        "original_query": "acute pancreatitis",
        "optimizer_used": True,
    }
    payload.update(overrides)
    return payload


class _ApiTestCase(unittest.TestCase):
    """Isolates the module-level semantic cache from the real cache file."""

    def setUp(self):
        self._cache = MagicMock()
        self._cache.check_cache.return_value = None
        self._cache.size = 0
        self._patch = patch.object(main, "_cache", self._cache)
        self._patch.start()
        self.client = TestClient(main.app)

    def tearDown(self):
        self._patch.stop()


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

class TestFinalK(unittest.TestCase):
    def test_simple_queries_use_a_tight_context(self):
        self.assertEqual(main._final_k_for("qa", "simple"), 5)
        self.assertEqual(main._final_k_for("smart_summary", "simple"), 5)

    def test_complex_queries_widen_the_context(self):
        self.assertEqual(main._final_k_for("qa", "complex"), 12)

    def test_smart_summary_is_capped_by_its_configured_ceiling(self):
        with patch.object(main, "SMART_SUMMARY_FINAL_K", 8):
            self.assertEqual(main._final_k_for("smart_summary", "complex"), 8)

    def test_unknown_complexity_is_treated_as_complex(self):
        self.assertEqual(main._final_k_for("qa", "anything-else"), 12)


class TestShouldSaveToCache(unittest.TestCase):
    def test_only_complete_verified_answers_are_cached(self):
        self.assertTrue(
            main._should_save_to_cache(
                disable_verifier=False, returned_path="verified", was_truncated=False
            )
        )

    def test_truncated_verified_answers_are_not_cached(self):
        self.assertFalse(
            main._should_save_to_cache(
                disable_verifier=False, returned_path="verified", was_truncated=True
            )
        )

    def test_unverified_paths_are_not_cached(self):
        for path in ("draft_fallback", "graceful_fallback", "error_fallback"):
            self.assertFalse(
                main._should_save_to_cache(
                    disable_verifier=False, returned_path=path, was_truncated=False
                ),
                path,
            )

    def test_disabling_the_verifier_disables_caching(self):
        self.assertFalse(
            main._should_save_to_cache(
                disable_verifier=True, returned_path="verified", was_truncated=False
            )
        )


class TestCacheSignature(unittest.TestCase):
    def test_signature_pins_every_parameter_that_changes_the_answer(self):
        signature = main._cache_signature(mode="qa", disable_verifier=False, final_k=12)
        for key in (
            "schema", "mode", "disable_verifier", "embedding_model", "embedding_dim",
            "retrieval_k", "final_k", "rerank_pool", "rrf_k", "rerank_model",
            "rerank_score_threshold", "faiss_dim", "faiss_ntotal", "chunk_count",
        ):
            self.assertIn(key, signature)

    def test_mode_change_produces_a_different_signature(self):
        qa = main._cache_signature(mode="qa", disable_verifier=False, final_k=12)
        summary = main._cache_signature(mode="smart_summary", disable_verifier=False, final_k=12)
        self.assertNotEqual(qa, summary)

    def test_verifier_state_produces_a_different_signature(self):
        on = main._cache_signature(mode="qa", disable_verifier=False, final_k=12)
        off = main._cache_signature(mode="qa", disable_verifier=True, final_k=12)
        self.assertNotEqual(on, off)

    def test_signature_tracks_the_loaded_vectorstore(self):
        signature = main._cache_signature(mode="qa", disable_verifier=False, final_k=12)
        self.assertEqual(signature["faiss_dim"], 1024)
        self.assertEqual(signature["faiss_ntotal"], 16983)


# ---------------------------------------------------------------------------
# Endpoint behaviour
# ---------------------------------------------------------------------------

class TestAskEndpoint(_ApiTestCase):
    def test_non_medical_query_short_circuits_before_retrieval(self):
        with patch.object(main, "optimize_query", return_value=_optimized(is_medical_query=False)), \
             patch.object(main, "retrieve") as retrieve, \
             patch.object(main, "ask_llm") as ask_llm:
            response = self.client.post("/ask", json={"query": "capital of France"})

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("medical reference assistant", body["answer"])
        self.assertEqual(body["sources"], [])
        self.assertEqual(body["visual_context"], [])
        retrieve.assert_not_called()
        ask_llm.assert_not_called()

    def test_cache_hit_skips_retrieval_and_generation(self):
        self._cache.check_cache.return_value = {
            "answer": "cached answer",
            "confidence": "High",
            "sources": ["p.142"],
            "visual_context": [],
        }
        with patch.object(main, "optimize_query", return_value=_optimized()), \
             patch.object(main, "retrieve") as retrieve, \
             patch.object(main, "ask_llm") as ask_llm:
            response = self.client.post("/ask", json={"query": "acute pancreatitis"})

        self.assertEqual(response.json()["answer"], "cached answer")
        retrieve.assert_not_called()
        ask_llm.assert_not_called()

    def test_response_matches_the_frozen_contract(self):
        with patch.object(main, "optimize_query", return_value=_optimized()), \
             patch.object(main, "retrieve", return_value=[{"chunk_id": 1, "page": 142, "text": "t", "score": 0.5}]), \
             patch.object(main, "route_and_sort_context", side_effect=lambda c: c), \
             patch.object(main, "fuse_context", return_value="context " * 20), \
             patch.object(main, "extract_evidence", return_value=[]), \
             patch.object(main, "ask_llm", return_value=("final", "draft", False, "verified")), \
             patch.object(main, "calculate_confidence", return_value="High"):
            response = self.client.post("/ask", json={"query": "acute pancreatitis"})

        body = response.json()
        self.assertEqual(set(body), {"answer", "confidence", "sources", "visual_context", "timings"})
        self.assertEqual(body["sources"], ["p.142"])
        self.assertEqual(body["visual_context"][0]["page_label"], "p.142")
        self.assertIn("total_request", body["timings"])

    def test_visual_context_urls_use_the_requesting_host(self):
        with patch.object(main, "optimize_query", return_value=_optimized()), \
             patch.object(main, "retrieve", return_value=[{"chunk_id": 1, "page": 142, "text": "t", "score": 0.5}]), \
             patch.object(main, "route_and_sort_context", side_effect=lambda c: c), \
             patch.object(main, "fuse_context", return_value="context " * 20), \
             patch.object(main, "extract_evidence", return_value=[]), \
             patch.object(main, "ask_llm", return_value=("final", "draft", False, "verified")), \
             patch.object(main, "calculate_confidence", return_value="High"):
            response = self.client.post(
                "/ask", json={"query": "acute pancreatitis"}, headers={"host": "harrison.example.com"}
            )

        self.assertTrue(
            response.json()["visual_context"][0]["full_url"].startswith("http://harrison.example.com/pages/")
        )

    def test_invalid_mode_is_rejected(self):
        response = self.client.post("/ask", json={"query": "x", "mode": "freeform"})
        self.assertEqual(response.status_code, 422)


class TestConfidenceCaps(_ApiTestCase):
    """The rule table that stops a degraded answer presenting as confident."""

    def _ask(self, returned_path, was_truncated, scored="High"):
        with patch.object(main, "optimize_query", return_value=_optimized()), \
             patch.object(main, "retrieve", return_value=[{"chunk_id": 1, "page": 1, "text": "t", "score": 0.5}]), \
             patch.object(main, "route_and_sort_context", side_effect=lambda c: c), \
             patch.object(main, "fuse_context", return_value="context " * 20), \
             patch.object(main, "extract_evidence", return_value=[]), \
             patch.object(main, "ask_llm", return_value=("final", "draft", was_truncated, returned_path)), \
             patch.object(main, "calculate_confidence", return_value=scored):
            return self.client.post("/ask", json={"query": "q"}).json()["confidence"]

    def test_verified_and_complete_keeps_high(self):
        self.assertEqual(self._ask("verified", False), "High")

    def test_truncation_caps_high_at_medium(self):
        self.assertEqual(self._ask("verified", True), "Medium")

    def test_draft_fallback_caps_high_at_medium(self):
        self.assertEqual(self._ask("draft_fallback", False), "Medium")

    def test_total_failure_paths_are_forced_low(self):
        self.assertEqual(self._ask("graceful_fallback", False), "Low")
        self.assertEqual(self._ask("error_fallback", True), "Low")

    def test_caps_never_upgrade_a_low_score(self):
        self.assertEqual(self._ask("verified", False, scored="Low"), "Low")
        self.assertEqual(self._ask("draft_fallback", False, scored="Medium"), "Medium")


class TestHealthEndpoint(_ApiTestCase):
    def test_healthy_system_reports_ok(self):
        body = self.client.get("/health").json()
        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["embedding_index_dim_match"])
        self.assertEqual(body["faiss_dim"], 1024)

    def test_healthy_system_returns_200(self):
        self.assertEqual(self.client.get("/health").status_code, 200)

    def test_dimension_mismatch_is_reported_as_degraded_with_503(self):
        with patch.object(main, "embedding_dimension", return_value=384):
            response = self.client.get("/health")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["status"], "degraded")
        self.assertFalse(response.json()["embedding_index_dim_match"])

    def test_missing_keys_are_reported_as_degraded_with_503(self):
        with patch.object(main.key_manager, "has_keys", return_value=False):
            response = self.client.get("/health")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["status"], "degraded")


class TestCachedVisualContextIsHostIndependent(_ApiTestCase):
    """A cached entry served behind a different host must not hand back the
    URLs of the host that first populated it."""

    def test_cache_hit_rebuilds_urls_for_the_current_host(self):
        self._cache.check_cache.return_value = {
            "answer": "cached", "confidence": "High", "sources": ["p.142"],
        }
        with patch.object(main, "optimize_query", return_value=_optimized()):
            response = self.client.post(
                "/ask", json={"query": "q"}, headers={"host": "second-host.example.com"}
            )

        entry = response.json()["visual_context"][0]
        self.assertEqual(entry["page_label"], "p.142")
        self.assertTrue(entry["full_url"].startswith("http://second-host.example.com/pages/"))

    def test_cache_hit_without_sources_yields_empty_visual_context(self):
        self._cache.check_cache.return_value = {
            "answer": "cached", "confidence": "High", "sources": [],
        }
        with patch.object(main, "optimize_query", return_value=_optimized()):
            response = self.client.post("/ask", json={"query": "q"})
        self.assertEqual(response.json()["visual_context"], [])

    def test_saved_payload_omits_host_specific_urls(self):
        with patch.object(main, "optimize_query", return_value=_optimized()), \
             patch.object(main, "retrieve", return_value=[{"chunk_id": 1, "page": 142, "text": "t", "score": 0.5}]), \
             patch.object(main, "route_and_sort_context", side_effect=lambda c: c), \
             patch.object(main, "fuse_context", return_value="context " * 20), \
             patch.object(main, "extract_evidence", return_value=[]), \
             patch.object(main, "ask_llm", return_value=("final", "final", False, "verified")), \
             patch.object(main, "calculate_confidence", return_value="High"):
            self.client.post("/ask", json={"query": "q"})

        saved = self._cache.save_to_cache.call_args.kwargs["response_data"]
        self.assertNotIn("visual_context", saved)
        self.assertEqual(saved["sources"], ["p.142"])


class TestInputBounds(_ApiTestCase):
    def test_oversized_query_is_rejected_before_any_work(self):
        with patch.object(main, "optimize_query") as optimize:
            response = self.client.post("/ask", json={"query": "x" * (main.MAX_QUERY_CHARS + 1)})
        self.assertEqual(response.status_code, 422)
        optimize.assert_not_called()

    def test_empty_query_is_rejected(self):
        self.assertEqual(self.client.post("/ask", json={"query": ""}).status_code, 422)

    def test_query_at_the_limit_is_accepted(self):
        with patch.object(main, "optimize_query", return_value=_optimized(is_medical_query=False)):
            response = self.client.post("/ask", json={"query": "x" * main.MAX_QUERY_CHARS})
        self.assertEqual(response.status_code, 200)


class TestAdminAuth(_ApiTestCase):
    def test_admin_cache_is_closed_when_no_token_is_configured(self):
        with patch.object(main, "ADMIN_TOKEN", ""):
            response = self.client.delete("/admin/cache")
        self.assertEqual(response.status_code, 503)
        self._cache.clear.assert_not_called()

    def test_admin_cache_rejects_a_missing_token(self):
        with patch.object(main, "ADMIN_TOKEN", "s3cret"):
            response = self.client.delete("/admin/cache")
        self.assertEqual(response.status_code, 401)
        self._cache.clear.assert_not_called()

    def test_admin_cache_rejects_a_wrong_token(self):
        with patch.object(main, "ADMIN_TOKEN", "s3cret"):
            response = self.client.delete("/admin/cache", headers={"X-Admin-Token": "wrong"})
        self.assertEqual(response.status_code, 401)
        self._cache.clear.assert_not_called()

    def test_admin_cache_accepts_the_configured_token(self):
        with patch.object(main, "ADMIN_TOKEN", "s3cret"):
            response = self.client.delete("/admin/cache", headers={"X-Admin-Token": "s3cret"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "success")
        self._cache.clear.assert_called_once()


class TestRateLimit(_ApiTestCase):
    def setUp(self):
        super().setUp()
        main._rate_hits.clear()

    def tearDown(self):
        main._rate_hits.clear()
        super().tearDown()

    def test_requests_over_the_window_are_refused_with_429(self):
        with patch.object(main, "RATE_LIMIT_PER_MINUTE", 3), \
             patch.object(main, "optimize_query", return_value=_optimized(is_medical_query=False)):
            codes = [self.client.post("/ask", json={"query": "q"}).status_code for _ in range(5)]

        self.assertEqual(codes, [200, 200, 200, 429, 429])

    def test_refusal_carries_a_retry_after_header(self):
        with patch.object(main, "RATE_LIMIT_PER_MINUTE", 1), \
             patch.object(main, "optimize_query", return_value=_optimized(is_medical_query=False)):
            self.client.post("/ask", json={"query": "q"})
            response = self.client.post("/ask", json={"query": "q"})

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["Retry-After"], "60")

    def test_health_is_never_rate_limited(self):
        with patch.object(main, "RATE_LIMIT_PER_MINUTE", 1):
            codes = [self.client.get("/health").status_code for _ in range(5)]
        self.assertEqual(codes, [200] * 5)

    def test_limit_of_zero_disables_the_limiter(self):
        with patch.object(main, "RATE_LIMIT_PER_MINUTE", 0), \
             patch.object(main, "optimize_query", return_value=_optimized(is_medical_query=False)):
            codes = [self.client.post("/ask", json={"query": "q"}).status_code for _ in range(6)]
        self.assertEqual(codes, [200] * 6)


class TestExceptionBoundary(_ApiTestCase):
    def test_pipeline_failure_returns_a_generic_500_without_a_traceback(self):
        client = TestClient(main.app, raise_server_exceptions=False)
        with patch.object(main, "optimize_query", side_effect=RuntimeError("faiss segfault: /secret/path")):
            response = client.post("/ask", json={"query": "acute pancreatitis"})

        self.assertEqual(response.status_code, 500)
        body = response.text
        self.assertNotIn("faiss segfault", body)
        self.assertNotIn("/secret/path", body)
        self.assertNotIn("Traceback", body)
        self.assertIn("Internal server error", response.json()["detail"])


class TestCors(unittest.TestCase):
    def test_no_cors_middleware_is_installed_by_default(self):
        installed = [m.cls.__name__ for m in main.app.user_middleware]
        self.assertNotIn("CORSMiddleware", installed)


if __name__ == "__main__":
    unittest.main()
