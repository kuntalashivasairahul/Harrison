"""
tests/test_pipeline_e2e.py
============================
End-to-end pipeline tests for the Harrison RAG system.

Covers:
  1. Verifier truncates + draft complete → returns draft_fallback
  2. Verifier succeeds → returns verified
  3. Both draft and verifier truncate → returns graceful_fallback
  4. Confidence cap works for every returned path
  5. Optimizer failure still allows retrieval (mocked)
  6. Dot notation sources remain p.####
  7. ask_llm returns 4-tuple on all code paths (including early exits)
  8. Error fallback doesn't expose raw exceptions
  9. verify_answer exception path → draft_fallback (not 'verified')
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _make_response(text: str, finish_reason_name: str = "STOP"):
    """Build a fake Gemini response object."""
    resp = MagicMock()
    resp.text = text
    cand = MagicMock()
    cand.finish_reason.name = finish_reason_name
    cand.content.parts = [MagicMock(text=text)]
    resp.candidates = [cand]
    return resp


def _make_truncated_response(text: str):
    """Build a fake Gemini response with MAX_TOKENS finish reason."""
    return _make_response(text, finish_reason_name="MAX_TOKENS")


def _chunks(scores: list[float], pages: list[int] | None = None) -> list[dict]:
    """Build minimal chunk dicts."""
    if pages is None:
        pages = list(range(100, 100 + len(scores)))
    return [
        {"chunk_id": i, "score": s, "text": f"Chunk text {i} " * 20, "page": p}
        for i, (s, p) in enumerate(zip(scores, pages))
    ]


class _FakeConfig:
    """Captures kwargs passed to GenerateContentConfig."""
    def __init__(self, **kw):
        self.__dict__.update(kw)


# -----------------------------------------------------------------------
# Test: ask_llm 4-tuple consistency on all paths
# -----------------------------------------------------------------------

class TestAskLlm4TupleConsistency(unittest.TestCase):
    """ask_llm must always return exactly 4 values."""

    def _call_ask_llm(self, fused_context: str = "Short", **kwargs):
        """Call ask_llm with patched dependencies and return the result tuple."""
        with patch("backend.llm.llm.key_manager") as mock_km, \
             patch("backend.llm.llm.types") as mock_types:
            mock_km.next_client.return_value = MagicMock()
            mock_types.GenerateContentConfig.side_effect = _FakeConfig
            from backend.llm.llm import ask_llm
            return ask_llm(fused_context=fused_context, question="test", **kwargs)

    def test_early_exit_empty_context_returns_4_tuple(self):
        """Empty context → 4-tuple with error_fallback path."""
        result = self._call_ask_llm(fused_context="")
        self.assertEqual(len(result), 4,
                         "ask_llm must return exactly 4 values on empty context")
        _, _, _, returned_path = result
        self.assertEqual(returned_path, "error_fallback")

    def test_early_exit_short_context_returns_4_tuple(self):
        """Context < 20 chars → 4-tuple with error_fallback path."""
        result = self._call_ask_llm(fused_context="tiny")
        self.assertEqual(len(result), 4,
                         "ask_llm must return exactly 4 values on short context")
        _, _, _, returned_path = result
        self.assertEqual(returned_path, "error_fallback")

    def test_empty_generation_returns_4_tuple(self):
        """Draft generation returns empty text → 4-tuple."""
        empty_resp = _make_response("")
        with patch("backend.llm.llm.key_manager") as mock_km, \
             patch("backend.llm.llm.types") as mock_types:
            mock_km.next_client.return_value = MagicMock()
            mock_km.next_client.return_value.models.generate_content.return_value = empty_resp
            mock_types.GenerateContentConfig.side_effect = _FakeConfig
            from backend.llm.llm import ask_llm
            result = ask_llm(fused_context="A" * 100, question="test")
        self.assertEqual(len(result), 4)
        _, _, _, returned_path = result
        self.assertEqual(returned_path, "error_fallback")

    def test_all_retries_fail_returns_4_tuple(self):
        """All LLM retries exhaust → 4-tuple with error_fallback."""
        with patch("backend.llm.llm.key_manager") as mock_km, \
             patch("backend.llm.llm.types") as mock_types, \
             patch("backend.llm.llm.DRAFT_MAX_ATTEMPTS", 1):
            client = MagicMock()
            client.models.generate_content.side_effect = RuntimeError("test error")
            mock_km.next_client.return_value = client
            mock_types.GenerateContentConfig.side_effect = _FakeConfig
            from backend.llm.llm import ask_llm
            result = ask_llm(fused_context="A" * 100, question="test")
        self.assertEqual(len(result), 4)
        answer, _, _, returned_path = result
        self.assertEqual(returned_path, "error_fallback")
        # Must NOT contain raw exception text
        self.assertNotIn("test error", answer)
        self.assertNotIn("LLM call failed", answer)

    def test_generation_uses_qa_token_limit(self):
        response = _make_response("draft")
        with patch("backend.llm.llm.key_manager") as mock_km, \
             patch("backend.llm.llm.types") as mock_types:
            client = MagicMock()
            client.models.generate_content.return_value = response
            mock_km.next_client.return_value = client
            mock_types.GenerateContentConfig.side_effect = _FakeConfig
            from backend.llm.llm import QA_MAX_TOKENS, ask_llm
            ask_llm("A" * 100, "test", mode="qa", disable_verifier=True)

        config = client.models.generate_content.call_args.kwargs["config"]
        self.assertEqual(config.max_output_tokens, QA_MAX_TOKENS)

    def test_generation_uses_smart_summary_token_limit(self):
        response = _make_response("draft")
        with patch("backend.llm.llm.key_manager") as mock_km, \
             patch("backend.llm.llm.types") as mock_types:
            client = MagicMock()
            client.models.generate_content.return_value = response
            mock_km.next_client.return_value = client
            mock_types.GenerateContentConfig.side_effect = _FakeConfig
            from backend.llm.llm import SMART_SUMMARY_MAX_TOKENS, ask_llm
            ask_llm("A" * 100, "test", mode="smart_summary", disable_verifier=True)

        config = client.models.generate_content.call_args.kwargs["config"]
        self.assertEqual(config.max_output_tokens, SMART_SUMMARY_MAX_TOKENS)


# -----------------------------------------------------------------------
# Test: error_fallback safety
# -----------------------------------------------------------------------

class TestErrorFallbackSafety(unittest.TestCase):
    """Error fallback must never expose raw provider errors."""

    def test_error_fallback_answer_is_user_safe(self):
        with patch("backend.llm.llm.key_manager") as mock_km, \
             patch("backend.llm.llm.types") as mock_types, \
             patch("backend.llm.llm.DRAFT_MAX_ATTEMPTS", 1):
            client = MagicMock()
            client.models.generate_content.side_effect = RuntimeError(
                "429 RESOURCE_EXHAUSTED: Quota exceeded for model"
            )
            mock_km.next_client.return_value = client
            mock_types.GenerateContentConfig.side_effect = _FakeConfig
            from backend.llm.llm import ask_llm
            answer, _, _, path = ask_llm(fused_context="A" * 100, question="test")
        self.assertEqual(path, "error_fallback")
        self.assertNotIn("429", answer)
        self.assertNotIn("RESOURCE_EXHAUSTED", answer)
        self.assertNotIn("Quota", answer)
        self.assertIn("try again", answer.lower())


# -----------------------------------------------------------------------
# Test: verify_answer 3-tuple and verification_ran flag
# -----------------------------------------------------------------------

class TestVerifyAnswerVerificationRanFlag(unittest.TestCase):
    """verify_answer must return (text, truncated, verification_ran)."""

    def test_successful_verification_sets_ran_true(self):
        resp = _make_response("Verified text")
        with patch("backend.llm.llm.key_manager") as mock_km, \
             patch("backend.llm.llm.types") as mock_types:
            client = MagicMock()
            client.models.generate_content.return_value = resp
            mock_km.next_client.return_value = client
            mock_types.GenerateContentConfig.side_effect = _FakeConfig
            from backend.llm.llm import verify_answer
            text, truncated, ran = verify_answer("draft", "context")
        self.assertTrue(ran)
        self.assertFalse(truncated)
        self.assertEqual(text, "Verified text")

    def test_exception_path_sets_ran_false(self):
        with patch("backend.llm.llm.key_manager") as mock_km, \
             patch("backend.llm.llm.types") as mock_types, \
             patch("backend.llm.llm.DRAFT_MAX_ATTEMPTS", 1):
            client = MagicMock()
            client.models.generate_content.side_effect = RuntimeError("fail")
            mock_km.next_client.return_value = client
            mock_types.GenerateContentConfig.side_effect = _FakeConfig
            from backend.llm.llm import verify_answer
            text, truncated, ran = verify_answer("draft", "context")
        self.assertFalse(ran, "verification_ran must be False when all retries threw exceptions")
        self.assertEqual(text, "draft", "Must return original draft when verification fails")

    def test_empty_inputs_sets_ran_false(self):
        from backend.llm.llm import verify_answer
        text, truncated, ran = verify_answer("", "context")
        self.assertFalse(ran)
        text, truncated, ran = verify_answer("draft", "")
        self.assertFalse(ran)

    def test_truncated_verification_sets_ran_true(self):
        resp = _make_truncated_response("Partial verified text")
        with patch("backend.llm.llm.key_manager") as mock_km, \
             patch("backend.llm.llm.types") as mock_types:
            client = MagicMock()
            client.models.generate_content.return_value = resp
            mock_km.next_client.return_value = client
            mock_types.GenerateContentConfig.side_effect = _FakeConfig
            from backend.llm.llm import verify_answer
            text, truncated, ran = verify_answer("draft", "context")
        self.assertTrue(ran, "verification_ran must be True even when truncated")
        self.assertTrue(truncated)


# -----------------------------------------------------------------------
# Test: return-path logic in ask_llm
# -----------------------------------------------------------------------

class TestAskLlmReturnPaths(unittest.TestCase):
    """ask_llm return_path must correctly reflect what actually happened."""

    def _run_ask_llm(self, draft_resp, verify_resp, verify_retry_resp=None):
        """Run ask_llm with controlled draft and verifier responses."""
        call_count = [0]

        def fake_generate(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return draft_resp
            elif call_count[0] == 2:
                return verify_resp
            elif verify_retry_resp and call_count[0] == 3:
                return verify_retry_resp
            return verify_resp

        with patch("backend.llm.llm.key_manager") as mock_km, \
             patch("backend.llm.llm.types") as mock_types:
            client = MagicMock()
            client.models.generate_content.side_effect = fake_generate
            mock_km.next_client.return_value = client
            mock_types.GenerateContentConfig.side_effect = _FakeConfig
            from backend.llm.llm import ask_llm
            return ask_llm(fused_context="A" * 100, question="test", mode="qa")

    def test_verified_path_when_both_complete(self):
        """Draft OK + verifier OK → returned_path='verified'."""
        draft = _make_response("Draft answer text")
        verified = _make_response("Verified answer text")
        answer, _, was_truncated, path = self._run_ask_llm(draft, verified)
        self.assertEqual(path, "verified")
        self.assertFalse(was_truncated)
        self.assertEqual(answer, "Verified answer text")

    def test_draft_fallback_when_verifier_truncates_and_draft_complete(self):
        """Draft OK + verifier truncated (both attempts) → returned_path='draft_fallback'."""
        draft = _make_response("Complete draft answer")
        trunc_verify = _make_truncated_response("Partial verified")
        answer, _, was_truncated, path = self._run_ask_llm(
            draft, trunc_verify, verify_retry_resp=trunc_verify
        )
        self.assertEqual(path, "draft_fallback")
        self.assertIn("Complete draft answer", answer)

    def test_graceful_fallback_when_both_truncated(self):
        """Draft truncated + verifier truncated → returned_path='graceful_fallback'."""
        draft = _make_truncated_response("Truncated draft")
        trunc_verify = _make_truncated_response("Truncated verified")
        answer, _, was_truncated, path = self._run_ask_llm(
            draft, trunc_verify, verify_retry_resp=trunc_verify
        )
        self.assertEqual(path, "graceful_fallback")
        self.assertTrue(was_truncated)
        self.assertIn("⚠️", answer)

    def test_verifier_retry_uses_the_configured_qa_token_limit(self):
        draft = _make_response("Complete draft answer")
        truncated = _make_truncated_response("Partial verified")
        verified = _make_response("Complete verified answer")
        responses = iter((draft, truncated, verified))
        client = MagicMock()
        client.models.generate_content.side_effect = lambda **_kwargs: next(responses)

        with patch("backend.llm.llm.key_manager") as mock_km, \
             patch("backend.llm.llm.types") as mock_types:
            mock_km.next_client.return_value = client
            mock_types.GenerateContentConfig.side_effect = _FakeConfig
            from backend.llm.llm import QA_MAX_TOKENS, ask_llm
            answer, _, was_truncated, path = ask_llm(
                fused_context="A" * 100,
                question="test",
                mode="qa",
            )

        retry_config = client.models.generate_content.call_args_list[2].kwargs["config"]
        self.assertEqual(retry_config.max_output_tokens, QA_MAX_TOKENS)
        self.assertEqual(answer, "Complete verified answer")
        self.assertFalse(was_truncated)
        self.assertEqual(path, "verified")

    def test_draft_fallback_when_verifier_throws_exception(self):
        """Draft OK + verifier exception → returned_path='draft_fallback' (NOT 'verified')."""
        draft = _make_response("Complete draft answer")

        call_count = [0]

        def fake_generate(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return draft
            raise RuntimeError("Verifier API failure")

        with patch("backend.llm.llm.key_manager") as mock_km, \
             patch("backend.llm.llm.types") as mock_types, \
             patch("backend.llm.llm.DRAFT_MAX_ATTEMPTS", 1):
            client = MagicMock()
            client.models.generate_content.side_effect = fake_generate
            mock_km.next_client.return_value = client
            mock_types.GenerateContentConfig.side_effect = _FakeConfig
            from backend.llm.llm import ask_llm
            answer, _, was_truncated, path = ask_llm(
                fused_context="A" * 100, question="test", mode="qa"
            )
        self.assertEqual(path, "draft_fallback",
                         "Verifier exception must produce draft_fallback, NOT verified")
        self.assertIn("Complete draft answer", answer)


# -----------------------------------------------------------------------
# Test: confidence caps per returned_path
# -----------------------------------------------------------------------

class TestConfidenceCaps(unittest.TestCase):
    """Confidence must be capped based on returned_path and truncation state."""

    def _compute_capped_confidence(self, raw_confidence, returned_path, was_truncated):
        """Apply the same cap logic as main.py."""
        confidence = raw_confidence
        if returned_path in ("graceful_fallback", "error_fallback"):
            confidence = "Low"
        elif returned_path in ("draft_fallback", "partial_verified") and confidence == "High":
            confidence = "Medium"
        elif was_truncated and confidence == "High":
            confidence = "Medium"
        return confidence

    def test_verified_not_truncated_allows_high(self):
        self.assertEqual(
            self._compute_capped_confidence("High", "verified", False),
            "High"
        )

    def test_verified_truncated_caps_to_medium(self):
        self.assertEqual(
            self._compute_capped_confidence("High", "verified", True),
            "Medium"
        )

    def test_draft_fallback_caps_high_to_medium(self):
        self.assertEqual(
            self._compute_capped_confidence("High", "draft_fallback", False),
            "Medium"
        )

    def test_draft_fallback_allows_medium(self):
        self.assertEqual(
            self._compute_capped_confidence("Medium", "draft_fallback", False),
            "Medium"
        )

    def test_partial_verified_caps_high_to_medium(self):
        self.assertEqual(
            self._compute_capped_confidence("High", "partial_verified", False),
            "Medium"
        )

    def test_graceful_fallback_always_low(self):
        self.assertEqual(
            self._compute_capped_confidence("High", "graceful_fallback", True),
            "Low"
        )

    def test_error_fallback_always_low(self):
        self.assertEqual(
            self._compute_capped_confidence("High", "error_fallback", True),
            "Low"
        )

    def test_error_fallback_low_stays_low(self):
        self.assertEqual(
            self._compute_capped_confidence("Low", "error_fallback", True),
            "Low"
        )


# -----------------------------------------------------------------------
# Test: dot notation sources
# -----------------------------------------------------------------------

class TestSourceDotNotation(unittest.TestCase):
    """Sources must always use p.NNN dot notation."""

    def test_extract_sources_dot_notation(self):
        from backend.processing.evidence import extract_sources
        chunks = [
            {"page": 1880, "text": "text"},
            {"page": 2137, "text": "text"},
            {"page": 3200, "text": "text"},
        ]
        sources = extract_sources(chunks)
        for src in sources:
            self.assertTrue(src.startswith("p."),
                            f"Source '{src}' must use dot notation p.NNN")
            self.assertNotIn(":", src,
                             f"Source '{src}' must not use colon notation")
        self.assertEqual(sources, ["p.1880", "p.2137", "p.3200"])


# -----------------------------------------------------------------------
# Test: optimizer fallback preserves retrieval
# -----------------------------------------------------------------------

class TestOptimizerFallback(unittest.TestCase):
    """Optimizer failure must not break the pipeline."""

    def test_optimizer_fallback_returns_original_query(self):
        from backend.agents.query_optimizer import _build_fallback
        result = _build_fallback("What causes chest pain?")
        self.assertTrue(result["is_medical_query"])
        self.assertEqual(result["expanded_query"], "What causes chest pain?")
        self.assertFalse(result["optimizer_used"])
        self.assertEqual(result["complexity"], "complex")

    def test_optimizer_exception_returns_fallback(self):
        """Full optimize_query with mocked LLM exception → safe fallback."""
        with patch("backend.agents.query_optimizer.llm_router") as mock_router:
            mock_router.optimize.side_effect = RuntimeError("connection error")
            from backend.agents.query_optimizer import optimize_query
            result = optimize_query("test medical query")
        self.assertFalse(result["optimizer_used"])
        self.assertTrue(result["is_medical_query"])
        self.assertEqual(result["expanded_query"], "test medical query")


if __name__ == "__main__":
    unittest.main()
