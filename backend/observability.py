"""Request correlation IDs and in-process pipeline metrics.

Two gaps this closes:

- **Correlation.** Every stage logged independently, so under any concurrency
  the lines from different requests interleaved with nothing to tie them
  together. A request id is generated per request, attached to the response as
  ``X-Request-ID``, and injected into every ``backend.*`` log record.

- **Aggregates.** The per-request ``timings`` dict was returned to the caller
  and then discarded. The same numbers now accumulate here, so ``/metrics``
  can answer "how often does the verifier fall back" and "what is p95
  retrieval" without replaying logs.

Deliberately dependency-free and in-process: one uvicorn worker, no Prometheus
client, no push gateway. If this ever runs multi-worker, export from here into
a real collector rather than reading these counters directly.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from contextvars import ContextVar
from typing import Any

#: Set per request by the middleware; read by the log filter.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


def new_request_id() -> str:
    return uuid.uuid4().hex[:12]


#: Absolute monotonic deadline for the whole request, set by /ask.
#: Stage deadlines are 60s each and every one of them retries, so nothing
#: bounded a single request end to end -- the deadline structure alone
#: permitted several minutes.  This is the ceiling all of them clamp against.
request_deadline_var: ContextVar[float | None] = ContextVar(
    "request_deadline", default=None
)


def start_request_budget(seconds: float) -> None:
    """Open a wall-clock budget for the current request. 0 or less disables it."""
    request_deadline_var.set(time.monotonic() + seconds if seconds > 0 else None)


def remaining_budget() -> float | None:
    """Seconds left in the request budget, or None when no budget is set."""
    deadline = request_deadline_var.get()
    return None if deadline is None else deadline - time.monotonic()


class RequestIdFilter(logging.Filter):
    """Attach the current request id to every record as ``%(request_id)s``."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

#: Cap on retained samples per timing stage — bounded memory, recent-window p95.
MAX_SAMPLES: int = 1000


class Metrics:
    """Counters and timing samples for the /ask pipeline."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[str, int] = {}
        self._timings: dict[str, list[float]] = {}

    def increment(self, name: str, value: int = 1) -> None:
        with self._lock:
            self._counters[name] = self._counters.get(name, 0) + value

    def observe(self, stage: str, seconds: float) -> None:
        with self._lock:
            samples = self._timings.setdefault(stage, [])
            samples.append(float(seconds))
            if len(samples) > MAX_SAMPLES:
                del samples[: len(samples) - MAX_SAMPLES]

    def observe_timings(self, timings: dict[str, float]) -> None:
        for stage, seconds in (timings or {}).items():
            try:
                self.observe(stage, float(seconds))
            except (TypeError, ValueError):
                continue

    @staticmethod
    def _percentile(ordered: list[float], fraction: float) -> float:
        if not ordered:
            return 0.0
        index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
        return ordered[index]

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            counters = dict(self._counters)
            timings = {stage: sorted(samples) for stage, samples in self._timings.items()}

        return {
            "counters": counters,
            "timings_seconds": {
                stage: {
                    "count": len(samples),
                    "mean": round(sum(samples) / len(samples), 4),
                    "p50": round(self._percentile(samples, 0.50), 4),
                    "p95": round(self._percentile(samples, 0.95), 4),
                    "max": round(samples[-1], 4),
                }
                for stage, samples in timings.items()
                if samples
            },
        }

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._timings.clear()


#: Process-wide metrics registry.
metrics = Metrics()
