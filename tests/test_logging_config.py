"""
tests/test_logging_config.py
============================
Regression tests for backend logger visibility.

The failure these guard against: uvicorn configures only its own loggers and
leaves root bare at WARNING, so every ``log.info(...)`` under ``backend.*``
was silently discarded when the app ran under the documented command.
"""
from __future__ import annotations

import logging
import logging.config
import unittest

from backend.logging_config import ROOT_LOGGER_NAME, configure_logging


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


class TestLoggingConfig(unittest.TestCase):
    def setUp(self) -> None:
        self._backend = logging.getLogger(ROOT_LOGGER_NAME)
        self._saved = (list(self._backend.handlers), self._backend.level, self._backend.propagate)

    def tearDown(self) -> None:
        handlers, level, propagate = self._saved
        self._backend.handlers = handlers
        self._backend.setLevel(level)
        self._backend.propagate = propagate

    def test_backend_child_loggers_emit_info_under_uvicorn_config(self) -> None:
        """The exact scenario that was broken: uvicorn's dictConfig, then INFO."""
        from uvicorn.config import LOGGING_CONFIG

        logging.config.dictConfig(LOGGING_CONFIG)
        configure_logging(force=True)

        capture = _Capture()
        self._backend.addHandler(capture)

        for name in (
            "backend.llm.llm",
            "backend.agents.query_optimizer",
            "backend.agents.semantic_cache",
            "backend.llm.router",
            "backend.api.main",
        ):
            logging.getLogger(name).info("diagnostic from %s", name)

        emitted = {record.name for record in capture.records}
        self.assertEqual(
            emitted,
            {
                "backend.llm.llm",
                "backend.agents.query_optimizer",
                "backend.agents.semantic_cache",
                "backend.llm.router",
                "backend.api.main",
            },
        )

    def test_configure_logging_is_idempotent(self) -> None:
        configure_logging(force=True)
        first = len(self._backend.handlers)
        configure_logging()
        configure_logging()
        self.assertEqual(len(self._backend.handlers), first)

    def test_level_is_configurable(self) -> None:
        configure_logging(level="WARNING", force=True)
        self.assertEqual(self._backend.level, logging.WARNING)
        configure_logging(level="INFO", force=True)
        self.assertEqual(self._backend.level, logging.INFO)

    def test_unknown_level_falls_back_to_info(self) -> None:
        configure_logging(level="NOT_A_LEVEL", force=True)
        self.assertEqual(self._backend.level, logging.INFO)

    def test_no_module_reaches_for_the_uvicorn_logger(self) -> None:
        """Modules must use __name__; grabbing uvicorn.error re-hides the bug."""
        from pathlib import Path

        backend_dir = Path(__file__).resolve().parents[1] / "backend"
        offenders = [
            str(path.relative_to(backend_dir))
            for path in backend_dir.rglob("*.py")
            if 'getLogger("uvicorn' in path.read_text(encoding="utf-8")
        ]
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
