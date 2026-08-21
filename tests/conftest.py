"""Suite-wide hermeticity guards.

The suite is hermetic by design (no network, no weights, no index), but
`backend/llm/llm.py` calls `load_dotenv()` at import, so the developer's real
`backend/.env` lands in `os.environ` before any test runs. That made routing
tests depend on whether the machine running them happened to have Groq enabled:
with `GROQ_ENABLED=true` in `.env`, `groq-draft` silently joined the draft
stage's deployment list and failover tests attempted one more provider than
they had scripted an outcome for.

Provider-enabling switches are therefore pinned off, and a test that wants one
on opts in explicitly with `patch.dict(os.environ, {...})`.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _disable_optional_providers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_ENABLED", "false")
    monkeypatch.setenv("MISTRAL_ENABLED", "false")
