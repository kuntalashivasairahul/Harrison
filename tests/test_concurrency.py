"""
tests/test_concurrency.py
=========================
Concurrency behaviour of the request path.

The audit flagged concurrency as "unverified" — the endpoints are sync `def`,
so FastAPI runs them in a threadpool, and several pieces of state are shared
across those threads: the semantic cache, the rate-limit table, the router's
cooldown map, and the request-id context variable.

These tests do not prove thread-safety in general; they exercise the specific
shared structures under parallel load and assert no cross-talk.
"""
from __future__ import annotations

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from backend.agents.semantic_cache import SemanticCache
from backend.observability import Metrics, request_id_var
from tests._api_harness import import_main

main = import_main()

WORKERS = 16
REQUESTS = 64


def _optimized():
    return {
        "is_medical_query": True,
        "expanded_query": "expanded",
        "focus": "management",
        "complexity": "complex",
        "original_query": "raw",
        "optimizer_used": True,
    }


class TestParallelRequests(unittest.TestCase):
    def setUp(self):
        self._cache = MagicMock()
        self._cache.check_cache.return_value = None
        self._patch = patch.object(main, "_cache", self._cache)
        self._patch.start()
        main._rate_hits.clear()
        self.client = TestClient(main.app)

    def tearDown(self):
        main._rate_hits.clear()
        self._patch.stop()

    def test_answers_are_not_crossed_between_concurrent_requests(self):
        """Each request must get back its own answer, not another's."""

        def fake_ask_llm(*, fused_context, question, **kwargs):
            return f"answer-for-{question}", "draft", False, "verified"

        def fake_retrieve(query, **kwargs):
            index = int(query.split("-")[-1])
            return [{"chunk_id": index, "page": index + 1, "text": "t", "score": 0.5}]

        with patch.object(main, "RATE_LIMIT_PER_MINUTE", 0), \
             patch.object(main, "optimize_query", side_effect=lambda q: {**_optimized(), "expanded_query": q}), \
             patch.object(main, "retrieve", side_effect=fake_retrieve), \
             patch.object(main, "route_and_sort_context", side_effect=lambda c: c), \
             patch.object(main, "fuse_context", return_value="context " * 20), \
             patch.object(main, "extract_evidence", return_value=[]), \
             patch.object(main, "ask_llm", side_effect=fake_ask_llm), \
             patch.object(main, "calculate_confidence", return_value="High"):

            def call(i: int):
                response = self.client.post("/ask", json={"query": f"query-{i}"})
                return i, response.json()

            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                results = list(pool.map(call, range(REQUESTS)))

        for i, body in results:
            self.assertEqual(body["answer"], f"answer-for-query-{i}")
            self.assertEqual(body["sources"], [f"p.{i + 1}"])

    def test_every_concurrent_request_gets_a_distinct_request_id(self):
        with patch.object(main, "RATE_LIMIT_PER_MINUTE", 0), \
             patch.object(main, "optimize_query", return_value={**_optimized(), "is_medical_query": False}):
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                responses = list(
                    pool.map(lambda _: self.client.post("/ask", json={"query": "q"}), range(REQUESTS))
                )

        ids = [r.headers["X-Request-ID"] for r in responses]
        self.assertEqual(len(set(ids)), REQUESTS)

    def test_supplied_request_id_is_echoed_back(self):
        with patch.object(main, "optimize_query", return_value={**_optimized(), "is_medical_query": False}):
            response = self.client.post(
                "/ask", json={"query": "q"}, headers={"X-Request-ID": "trace-me-123"}
            )
        self.assertEqual(response.headers["X-Request-ID"], "trace-me-123")

    def test_request_id_context_does_not_leak_between_requests(self):
        with patch.object(main, "optimize_query", return_value={**_optimized(), "is_medical_query": False}):
            self.client.post("/ask", json={"query": "q"}, headers={"X-Request-ID": "abc"})
        self.assertEqual(request_id_var.get(), "-")

    def test_rate_limiter_counts_exactly_under_parallel_load(self):
        """The limiter's window is shared mutable state across threads."""
        limit = 10
        with patch.object(main, "RATE_LIMIT_PER_MINUTE", limit), \
             patch.object(main, "optimize_query", return_value={**_optimized(), "is_medical_query": False}):
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                codes = list(
                    pool.map(lambda _: self.client.post("/ask", json={"query": "q"}).status_code, range(REQUESTS))
                )

        # Exactly `limit` requests may pass; no more, no fewer.
        self.assertEqual(codes.count(200), limit)
        self.assertEqual(codes.count(429), REQUESTS - limit)


class TestSemanticCacheUnderLoad(unittest.TestCase):
    def test_concurrent_reads_and_writes_do_not_corrupt_entries(self):
        cache = SemanticCache()
        cache._entries = []
        cache._flush_to_disk = lambda: None  # keep the test off disk

        errors: list[Exception] = []

        def writer(i: int):
            try:
                cache.save_to_cache(
                    query_embedding=[float(i % 7)] * 8,
                    response_data={"answer": f"a{i}", "confidence": "High", "sources": []},
                    metadata={"mode": "qa"},
                )
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        def reader(i: int):
            try:
                cache.check_cache([1.0] * 8, metadata={"mode": "qa"})
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for i in range(REQUESTS):
                pool.submit(writer, i)
                pool.submit(reader, i)

        self.assertEqual(errors, [])
        self.assertTrue(all(isinstance(e.get("embedding"), list) for e in cache._entries))
        self.assertTrue(all("response" in e for e in cache._entries))

    def test_eviction_cap_holds_under_parallel_writes(self):
        from backend.agents import semantic_cache as cache_mod

        cache = SemanticCache()
        cache._entries = []
        cache._flush_to_disk = lambda: None

        with patch.object(cache_mod, "MAX_CACHE_ENTRIES", 20):
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                list(pool.map(
                    lambda i: cache.save_to_cache(
                        query_embedding=[float(i)] * 8,
                        response_data={"answer": f"a{i}", "confidence": "High", "sources": []},
                    ),
                    range(200),
                ))

        self.assertLessEqual(len(cache._entries), 20)


class TestMetricsUnderLoad(unittest.TestCase):
    def test_counters_are_exact_under_parallel_increments(self):
        m = Metrics()
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            list(pool.map(lambda _: m.increment("hits"), range(2000)))
        self.assertEqual(m.snapshot()["counters"]["hits"], 2000)

    def test_timing_samples_stay_bounded(self):
        from backend.observability import MAX_SAMPLES

        m = Metrics()
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            list(pool.map(lambda i: m.observe("retrieval", i / 1000), range(MAX_SAMPLES * 2)))
        self.assertLessEqual(m.snapshot()["timings_seconds"]["retrieval"]["count"], MAX_SAMPLES)


class TestRouterCooldownUnderLoad(unittest.TestCase):
    def test_concurrent_cooldown_reads_and_writes(self):
        from backend.llm.contracts import LLMError, LLMErrorCategory
        from backend.llm.router import LLMRouter

        router = LLMRouter(MagicMock(), "prod", "backup")
        deployment = router._deployments["groq-optimizer"]
        error = LLMError(LLMErrorCategory.RATE_LIMITED, "429", retry_after_seconds=30)

        errors: list[Exception] = []
        stop = threading.Event()

        def reader():
            try:
                while not stop.is_set():
                    router._enabled(deployment)
                    router.status()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(4)]
        for t in threads:
            t.start()
        try:
            for _ in range(500):
                router._cooldown(deployment, error)
        finally:
            stop.set()
            for t in threads:
                t.join(timeout=5)

        self.assertEqual(errors, [])


if __name__ == "__main__":
    unittest.main()
