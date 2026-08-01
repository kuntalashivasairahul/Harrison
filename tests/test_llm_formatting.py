"""Regression tests for mutually exclusive QA and Smart Summary prompts."""

from __future__ import annotations

import unittest

from backend.llm.llm import (
    QA_FORMAT_INSTRUCTIONS,
    SMART_SUMMARY_ACK,
    SMART_SUMMARY_FORMAT_INSTRUCTIONS,
    _formatting_instructions,
)


class TestFormattingInstructions(unittest.TestCase):
    def test_qa_mode_excludes_smart_summary_contract(self) -> None:
        instructions = _formatting_instructions("qa")
        self.assertEqual(instructions, QA_FORMAT_INSTRUCTIONS)
        self.assertNotIn(SMART_SUMMARY_ACK, instructions)
        self.assertNotIn("Quick Revision", instructions)

    def test_smart_summary_mode_excludes_qa_contract(self) -> None:
        instructions = _formatting_instructions("smart_summary")
        self.assertEqual(instructions, SMART_SUMMARY_FORMAT_INSTRUCTIONS)
        self.assertIn(SMART_SUMMARY_ACK, instructions)


if __name__ == "__main__":
    unittest.main()
