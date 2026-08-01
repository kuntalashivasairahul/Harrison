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
