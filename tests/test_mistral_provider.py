"""No-network tests for the Mistral adapter.

Every HTTP call is stubbed at ``urllib.request.urlopen``; nothing here touches
api.mistral.ai.
"""

from __future__ import annotations

import io
import json
import unittest
import urllib.error
from unittest.mock import patch

from backend.llm.contracts import LLMError, LLMErrorCategory, LLMRequest, LLMStage
from backend.llm.mistral_provider import MistralProvider


def _request(prompt: str = "prompt") -> LLMRequest:
    return LLMRequest(prompt, "system", "mistral-draft", 0.0, 512, 30.0, LLMStage.DRAFT)


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


def _ok_body(text: str = "draft text") -> _Response:
    return _Response(json.dumps({
        "id": "req-1",
        "model": "mistral-large-2411",
        "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }).encode("utf-8"))


def _http_error(code: int, body: dict, headers: dict | None = None) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://api.mistral.ai/v1/chat/completions", code, "err",
        headers or {}, io.BytesIO(json.dumps(body).encode("utf-8")),
    )


class TestMistralProvider(unittest.TestCase):
    def setUp(self) -> None:
        self.provider = MistralProvider()
        self.provider._api_key = "test-key"

    def test_unconfigured_provider_raises_auth_without_calling_out(self) -> None:
        provider = MistralProvider()
        provider._api_key = ""
        self.assertFalse(provider.configured)
        with patch("urllib.request.urlopen") as urlopen:
            with self.assertRaises(LLMError) as caught:
                provider.generate(_request(), "mistral-large-latest")
        urlopen.assert_not_called()
        self.assertEqual(caught.exception.category, LLMErrorCategory.AUTH)

    def test_successful_response_is_mapped_to_the_shared_result_contract(self) -> None:
        with patch("urllib.request.urlopen", return_value=_ok_body()) as urlopen:
            result = self.provider.generate(_request(), "mistral-large-latest")
        self.assertEqual(result.text, "draft text")
        self.assertEqual(result.provider, "mistral")
        self.assertEqual(result.finish_reason, "STOP")
        self.assertEqual((result.input_tokens, result.output_tokens), (100, 20))
        # Prefer the model the response reports over the one we asked for.
        # Live, Mistral echoes the alias back ("mistral-large-latest"), but a
        # response that does name the resolved build is what the logs want, and
        # "-latest" otherwise hides which build actually answered.
        self.assertEqual(result.model, "mistral-large-2411")
        sent = json.loads(urlopen.call_args.args[0].data)
        self.assertEqual(sent["messages"][0]["role"], "system")
        self.assertEqual(sent["max_tokens"], 512)

    def test_openai_style_length_reads_as_truncated(self) -> None:
        """Mistral says "length"; only Gemini says MAX_TOKENS.

        Every consumer compared finish_reason against the literal "MAX_TOKENS",
        so a Mistral draft cut at the ceiling arrived with truncated=False: no
        truncation notice, no confidence cap, and cached and re-served as High.
        """
        for reason in ("length", "model_length"):
            body = _Response(json.dumps({
                "id": "req-1",
                "model": "mistral-large-2411",
                "choices": [{"message": {"content": "cut off mid-"}, "finish_reason": reason}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 512},
            }).encode("utf-8"))
            with self.subTest(reason=reason), patch("urllib.request.urlopen", return_value=body):
                result = self.provider.generate(_request(), "mistral-large-latest")
            self.assertTrue(result.truncated)

    def test_a_natural_stop_is_not_truncated(self) -> None:
        with patch("urllib.request.urlopen", return_value=_ok_body()):
            result = self.provider.generate(_request(), "mistral-large-latest")
        self.assertFalse(result.truncated)

    def test_api_key_is_sent_as_a_bearer_token(self) -> None:
        with patch("urllib.request.urlopen", return_value=_ok_body()) as urlopen:
            self.provider.generate(_request(), "mistral-large-latest")
        headers = urlopen.call_args.args[0].headers
        self.assertEqual(headers["Authorization"], "Bearer test-key")

    def test_429_is_rate_limited_and_carries_retry_after(self) -> None:
        error = _http_error(429, {"message": "Requests rate limit exceeded"}, {"Retry-After": "12"})
        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(LLMError) as caught:
                self.provider.generate(_request(), "mistral-large-latest")
        self.assertEqual(caught.exception.category, LLMErrorCategory.RATE_LIMITED)
        self.assertEqual(caught.exception.retry_after_seconds, 12.0)

    def test_401_is_auth_and_is_not_retried_elsewhere(self) -> None:
        with patch("urllib.request.urlopen", side_effect=_http_error(401, {"message": "Unauthorized"})):
            with self.assertRaises(LLMError) as caught:
                self.provider.generate(_request(), "mistral-large-latest")
        self.assertEqual(caught.exception.category, LLMErrorCategory.AUTH)

    def test_retired_model_is_not_found_so_the_stage_can_fall_over(self) -> None:
        """Mistral answers an unknown model with 400, which normalizes to
        INVALID_REQUEST and stops the stage. The body names the real cause, so
        the NOT_FOUND markers have to win — a retired model is the single most
        common way a deployment in this project dies."""
        error = _http_error(400, {"message": "Invalid model: mistral-tiny does not exist"})
        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(LLMError) as caught:
                self.provider.generate(_request(), "mistral-tiny")
        self.assertEqual(caught.exception.category, LLMErrorCategory.NOT_FOUND)

    def test_socket_timeout_is_categorized_as_timeout(self) -> None:
        with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
            with self.assertRaises(LLMError) as caught:
                self.provider.generate(_request(), "mistral-large-latest")
        self.assertEqual(caught.exception.category, LLMErrorCategory.TIMEOUT)


if __name__ == "__main__":
    unittest.main()
