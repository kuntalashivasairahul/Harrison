"""
tests/test_evaluate_rag.py
==========================
Unit tests for helpers in scripts/evaluate_rag.py.

Covers the three real-world judge output shapes that previously caused crashes:
  1. Clean JSON string          — fast path (json.loads succeeds directly)
  2. Markdown-fenced JSON       — slow path (regex extracts {...} block)
  3. Conversational preamble    — slow path (regex extracts {...} block)
  4. Completely unparseable     — returns None (caller raises ValueError)
"""
from __future__ import annotations

import importlib
import sys
import types
import unittest
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: scripts/ is not a package; import evaluate_rag as a module spec.
# We only need _extract_json, so we stub out heavy dependencies.
# ---------------------------------------------------------------------------

def _load_evaluate_rag_module():
    """
    Import scripts/evaluate_rag.py without executing the
    full module (which requires a running server and API keys).
    We do this by temporarily monkeypatching the missing imports.
    """
    root = Path(__file__).resolve().parents[1]
    scripts_dir = root / "scripts"

    # evaluate_rag.py and backend.llm.llm both use the new google.genai SDK.
    # Stub that path on a single google namespace module.
    #
    # The stub MUST be undone.  This function runs at import time, and leaving
    # a bare google.genai.types in sys.modules poisoned every test collected
    # afterwards: anything that later resolved a real SDK symbol (e.g.
    # types.ThinkingConfig) silently got the stub instead and took a
    # feature-missing code path.
    saved = {
        name: sys.modules.get(name)
        for name in ("google", "google.genai", "google.genai.types")
    }
    google_mod = sys.modules.get("google") or types.ModuleType("google")
    saved_genai_attr = getattr(google_mod, "genai", None)

    # Stub google.genai (new SDK used by llm.py)
    new_genai = types.ModuleType("google.genai")
    new_genai.Client = lambda **kw: object()
    new_genai_types = types.ModuleType("google.genai.types")

    class GenerateContentConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    new_genai_types.GenerateContentConfig = GenerateContentConfig
    new_genai.types = new_genai_types
    google_mod.genai = new_genai
    sys.modules["google.genai"] = new_genai
    sys.modules["google.genai.types"] = new_genai_types

    sys.modules["google"] = google_mod

    spec = importlib.util.spec_from_file_location(
        "evaluate_rag", scripts_dir / "evaluate_rag.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        # Module-level side-effects (server calls, @dataclass quirk on
        # Python 3.14) may raise after _extract_json is already defined.
        pass
    finally:
        # Put the real SDK back before anything else imports it.
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        if saved_genai_attr is not None:
            google_mod.genai = saved_genai_attr
        elif hasattr(google_mod, "genai"):
            del google_mod.genai

    if not hasattr(mod, "_extract_json"):
        raise ImportError("_extract_json not found in evaluate_rag")
    return mod


evaluate_rag = _load_evaluate_rag_module()
_extract_json = evaluate_rag._extract_json


class TestExtractJson(unittest.TestCase):
    """_extract_json must handle every judge output shape without crashing."""

    # ------------------------------------------------------------------
    # Shape 1: clean JSON — fast path
    # ------------------------------------------------------------------
    def test_clean_json(self):
        raw = '{"score": 4, "reasoning": "Covers key DKA criteria."}'
        result = _extract_json(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result["score"], 4)
        self.assertEqual(result["reasoning"], "Covers key DKA criteria.")

    # ------------------------------------------------------------------
    # Shape 2: markdown-fenced JSON — slow path
    # ------------------------------------------------------------------
    def test_markdown_fenced_json(self):
        raw = '```json\n{"score": 3, "reasoning": "Partially correct."}\n```'
        result = _extract_json(raw)
        self.assertIsNotNone(result, "Should extract JSON from inside ```json fence")
        self.assertEqual(result["score"], 3)

    def test_markdown_fenced_no_lang_tag(self):
        raw = '```\n{"score": 2, "reasoning": "Missing fluids dosage."}\n```'
        result = _extract_json(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result["score"], 2)

    # ------------------------------------------------------------------
    # Shape 3: conversational preamble — slow path
    # ------------------------------------------------------------------
    def test_preamble_before_json(self):
        raw = (
            'Sure! Here is the structured evaluation:\n\n'
            '{"score": 5, "reasoning": "Comprehensive and accurate."}'
        )
        result = _extract_json(raw)
        self.assertIsNotNone(result, "Should extract JSON block after preamble text")
        self.assertEqual(result["score"], 5)

    def test_preamble_with_markdown_fence(self):
        raw = (
            "After reviewing the answer, here's my evaluation:\n\n"
            "```json\n{\"score\": 1, \"reasoning\": \"No clinical detail.\"}\n```"
        )
        result = _extract_json(raw)
        self.assertIsNotNone(result)
        self.assertEqual(result["score"], 1)

    # ------------------------------------------------------------------
    # Shape 4: completely unparseable — must return None, not raise
    # ------------------------------------------------------------------
    def test_completely_unparseable_returns_none(self):
        raw = "The answer looks great to me! Very comprehensive."
        result = _extract_json(raw)
        self.assertIsNone(result, "_extract_json must return None, not raise, on garbage input")

    def test_empty_string_returns_none(self):
        result = _extract_json("")
        self.assertIsNone(result)

    # ------------------------------------------------------------------
    # Regression: old markdown-fence stripping only handled startswith("```")
    # which missed mid-string fences.
    # ------------------------------------------------------------------
    def test_json_embedded_mid_string(self):
        raw = 'Here is the score. {"score": 3, "reasoning": "Good coverage."} That is all.'
        result = _extract_json(raw)
        self.assertIsNotNone(result, "Should find JSON block embedded mid-string")
        self.assertEqual(result["score"], 3)


class TestJudgeAnswer(unittest.TestCase):
    """judge_answer must use the google-genai client path and safe key handling."""

    def test_judge_answer_uses_next_client_and_generate_content(self):
        calls = []

        class FakeModels:
            def generate_content(self, **kwargs):
                calls.append(kwargs)
                return types.SimpleNamespace(
                    text='{"score": 4, "reasoning": "Covers the core points."}'
                )

        class FakeClient:
            models = FakeModels()

        class FakeKeyManager:
            def next_client(self):
                calls.append({"next_client": True})
                return FakeClient()

            def mark_rate_limited(self):
                calls.append({"mark_rate_limited": True})

            def rotate(self):
                calls.append({"rotate": True})

        old_key_manager = evaluate_rag.key_manager
        old_model = evaluate_rag.JUDGE_MODEL
        try:
            evaluate_rag.key_manager = FakeKeyManager()
            evaluate_rag.JUDGE_MODEL = "test-judge-model"

            result = evaluate_rag.judge_answer("query", "expected", "answer")

            self.assertEqual(result["score"], 4)
            self.assertTrue(any(c.get("next_client") for c in calls if isinstance(c, dict)))
            generate_call = next(c for c in calls if isinstance(c, dict) and c.get("model"))
            self.assertEqual(generate_call["model"], "test-judge-model")
            self.assertEqual(generate_call["contents"], "QUERY: query\n\nEXPECTED FOCUS: expected\n\nGENERATED ANSWER:\nanswer\n\nJSON evaluation:")
            self.assertEqual(generate_call["config"].kwargs["temperature"], 0.0)
            self.assertEqual(generate_call["config"].kwargs["max_output_tokens"], 256)
            self.assertIn("system_instruction", generate_call["config"].kwargs)
            self.assertFalse(any(c.get("rotate") for c in calls if isinstance(c, dict)))
        finally:
            evaluate_rag.key_manager = old_key_manager
            evaluate_rag.JUDGE_MODEL = old_model

    def test_judge_rate_limit_cools_down_key_without_rotate(self):
        calls = []

        class FakeModels:
            count = 0

            def generate_content(self, **kwargs):
                self.count += 1
                if self.count == 1:
                    raise RuntimeError("429 quota exceeded")
                return types.SimpleNamespace(
                    text='{"score": 5, "reasoning": "Recovered after retry."}'
                )

        class FakeClient:
            models = FakeModels()

        class FakeKeyManager:
            def next_client(self):
                calls.append({"next_client": True})
                return FakeClient()

            def mark_rate_limited(self):
                calls.append({"mark_rate_limited": True})

            def rotate(self):
                calls.append({"rotate": True})

        old_key_manager = evaluate_rag.key_manager
        old_sleep = evaluate_rag.time.sleep
        old_attempts = evaluate_rag.RETRY_MAX_ATTEMPTS
        try:
            evaluate_rag.key_manager = FakeKeyManager()
            evaluate_rag.time.sleep = lambda _seconds: None
            evaluate_rag.RETRY_MAX_ATTEMPTS = 2

            result = evaluate_rag.judge_answer("query", "expected", "answer")

            self.assertEqual(result["score"], 5)
            self.assertEqual(
                sum(1 for c in calls if isinstance(c, dict) and c.get("mark_rate_limited")),
                1,
            )
            self.assertFalse(any(c.get("rotate") for c in calls if isinstance(c, dict)))
        finally:
            evaluate_rag.key_manager = old_key_manager
            evaluate_rag.time.sleep = old_sleep
            evaluate_rag.RETRY_MAX_ATTEMPTS = old_attempts


if __name__ == "__main__":
    unittest.main()
