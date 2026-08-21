"""No-network tests for Groq response and error normalization."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from backend.llm.contracts import LLMError, LLMErrorCategory, LLMRequest, LLMStage
from backend.llm.groq_provider import GroqProvider


class TestGroqProvider(unittest.TestCase):
    def _request(self) -> LLMRequest:
        return LLMRequest("prompt", "system", "groq-optimizer", 0.0, 32, 5.0, LLMStage.OPTIMIZER)

    def test_missing_key_is_auth_error(self) -> None:
        with patch.dict("os.environ", {"GROQ_API_KEY": ""}, clear=False):
            provider = GroqProvider()
        with self.assertRaises(LLMError) as context:
            provider.generate(self._request(), "model")
        self.assertEqual(context.exception.category, LLMErrorCategory.AUTH)

    def test_completion_is_normalized(self) -> None:
        response = MagicMock()
        response.id = "req-1"
        response.usage.prompt_tokens = 12
        response.usage.completion_tokens = 5
        response.choices = [MagicMock(message=MagicMock(content="{}"), finish_reason="stop")]
        client = MagicMock()
        client.chat.completions.create.return_value = response
        with patch.dict("os.environ", {"GROQ_API_KEY": "key"}, clear=False), patch("backend.llm.groq_provider.Groq", return_value=client):
            result = GroqProvider().generate(self._request(), "model")
        self.assertEqual(result.text, "{}")
        self.assertEqual(result.finish_reason, "STOP")
        self.assertEqual(result.input_tokens, 12)

    def test_429_is_rate_limited(self) -> None:
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("429 retry-after: 7")
        with patch.dict("os.environ", {"GROQ_API_KEY": "key"}, clear=False), patch("backend.llm.groq_provider.Groq", return_value=client):
            with self.assertRaises(LLMError) as context:
                GroqProvider().generate(self._request(), "model")
        self.assertEqual(context.exception.category, LLMErrorCategory.RATE_LIMITED)
        self.assertEqual(context.exception.retry_after_seconds, 7.0)


class TestReasoningModelHandling(unittest.TestCase):
    """The approved optimizer model is a reasoning model.

    Left unbounded it spends the whole max_tokens budget on its internal chain
    and returns finish_reason="length" with empty content — which is how the
    optimizer silently fell back to its local path on every query.
    """

    def _request(self, max_tokens: int = 512) -> LLMRequest:
        return LLMRequest("prompt", "system", "groq-optimizer", 0.0, max_tokens, 5.0, LLMStage.OPTIMIZER)

    def _client(self):
        response = MagicMock()
        response.id = "req"
        response.usage.prompt_tokens = 1
        response.usage.completion_tokens = 1
        response.choices = [MagicMock(message=MagicMock(content="{}"), finish_reason="stop")]
        client = MagicMock()
        client.chat.completions.create.return_value = response
        return client

    def _call(self, model: str, env: dict | None = None):
        client = self._client()
        environ = {"GROQ_API_KEY": "key"}
        environ.update(env or {})
        with patch.dict("os.environ", environ, clear=False), \
             patch("backend.llm.groq_provider.Groq", return_value=client):
            GroqProvider().generate(self._request(), model)
        return client.chat.completions.create.call_args.kwargs

    def test_reasoning_effort_is_sent_for_gpt_oss(self):
        self.assertEqual(self._call("openai/gpt-oss-20b").get("reasoning_effort"), "low")

    def test_reasoning_effort_is_sent_for_qwen3(self):
        self.assertEqual(self._call("qwen/qwen3.6-27b").get("reasoning_effort"), "low")

    def test_reasoning_effort_is_absent_for_non_reasoning_models(self):
        self.assertNotIn("reasoning_effort", self._call("llama-3.3-70b-versatile"))

    def test_reasoning_effort_is_configurable(self):
        kwargs = self._call("openai/gpt-oss-20b", {"GROQ_REASONING_EFFORT": "medium"})
        self.assertEqual(kwargs["reasoning_effort"], "medium")

    def test_invalid_reasoning_effort_falls_back_to_low(self):
        kwargs = self._call("openai/gpt-oss-20b", {"GROQ_REASONING_EFFORT": "banana"})
        self.assertEqual(kwargs["reasoning_effort"], "low")


class TestOptimizerModelIsLive(unittest.TestCase):
    def test_registry_does_not_pin_the_decommissioned_llama_model(self):
        """llama-3.1-8b-instant was retired by Groq and 404s on every call."""
        from backend.llm.router import load_registry

        model = load_registry()["groq-optimizer"].model
        self.assertNotIn("llama-3.1-8b-instant", model)

    def test_registry_budget_fits_the_optimizer_token_request(self):
        """The router clamps with min(); too small a registry cap silently
        truncates the reasoning model back into the broken state."""
        from backend.agents.query_optimizer import _MAX_TOKENS
        from backend.llm.router import load_registry

        self.assertGreaterEqual(load_registry()["groq-optimizer"].max_output_tokens, _MAX_TOKENS)


class TestGroqTokenPerMinuteRejection(unittest.TestCase):
    """Groq answers an over-budget prompt with HTTP 413 code=rate_limit_exceeded.
    That spelling matched none of the spaced rate-limit markers, so it
    normalized to UNKNOWN and the draft stage stopped instead of failing over."""

    def test_413_rate_limit_exceeded_is_categorized_as_rate_limited(self) -> None:
        from backend.llm.gemini_provider import normalize_provider_error

        exc = Exception(
            "Error code: 413 - {'error': {'message': 'Request too large for model "
            "`openai/gpt-oss-120b` ... on tokens per minute (TPM): Limit 8000, "
            "Requested 11554', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}"
        )
        normalized = normalize_provider_error(exc, "groq")
        self.assertEqual(normalized.category, LLMErrorCategory.RATE_LIMITED)
