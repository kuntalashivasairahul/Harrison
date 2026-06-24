"""
tests/test_evaluate_rag.py
==========================
Unit tests for the _extract_json helper added to scripts/evaluate_rag.py.

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

def _load_extract_json():
    """
    Import _extract_json from scripts/evaluate_rag.py without executing the
    full module (which requires a running server and API keys).
    We do this by temporarily monkeypatching the missing imports.
    """
    root = Path(__file__).resolve().parents[1]
    scripts_dir = root / "scripts"

    # Fully stub google.generativeai so module-level calls don't fail.
    genai_stub = types.ModuleType("google.generativeai")
    genai_stub.GenerativeModel = object
    genai_stub.configure = lambda **kw: None
    genai_stub.list_models = lambda: []
    genai_stub.types = types.SimpleNamespace(GenerationConfig=object)
    sys.modules.setdefault("google", types.ModuleType("google"))
    sys.modules["google.generativeai"] = genai_stub

    spec = importlib.util.spec_from_file_location(
        "evaluate_rag", scripts_dir / "evaluate_rag.py"
    )
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        # Module-level side-effects (server calls, @dataclass quirk on
        # Python 3.14) may raise after _extract_json is already defined.
        pass

    if not hasattr(mod, "_extract_json"):
        raise ImportError(
            "_extract_json not found in evaluate_rag — "
            "check that Fix 3 was applied correctly."
        )
    return mod._extract_json


_extract_json = _load_extract_json()


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


if __name__ == "__main__":
    unittest.main()
