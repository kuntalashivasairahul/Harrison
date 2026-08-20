"""Application logging configuration.

Why this module exists
----------------------
Uvicorn configures handlers for ``uvicorn``, ``uvicorn.error`` and
``uvicorn.access`` only.  It leaves the root logger bare, so a module logger
obtained with ``logging.getLogger(__name__)`` resolves to the root's default
level (WARNING) with no handler attached — and every ``log.info(...)`` in
``backend.*`` is silently discarded when the app runs under the documented
uvicorn command.

``configure_logging()`` attaches one handler to the ``backend`` logger so all
``backend.*`` module loggers become visible, using a format that matches
uvicorn's own output.  Modules keep using ``logging.getLogger(__name__)``;
they must not reach for ``uvicorn.error`` to get around this.

Call this once, before importing the rest of ``backend``, so diagnostics
emitted during module import are captured too.
"""

from __future__ import annotations

import logging
import os
import sys

#: Logger every ``backend.*`` module logger propagates to.
ROOT_LOGGER_NAME = "backend"

#: Matches uvicorn's default line shape, plus the request id so lines from
#: concurrent requests can be told apart and grepped as a group.
LOG_FORMAT = "%(levelname)-8s [%(request_id)s] %(name)s: %(message)s"

_configured = False


def configure_logging(level: str | int | None = None, *, force: bool = False) -> logging.Logger:
    """Attach a stream handler to the ``backend`` logger.

    Parameters
    ----------
    level:
        Log level for ``backend.*``.  Defaults to the ``HARRISON_LOG_LEVEL``
        environment variable, or ``INFO``.
    force:
        Re-configure even if this function has already run.  Used by tests.

    Returns
    -------
    logging.Logger
        The configured ``backend`` logger.
    """
    global _configured

    logger = logging.getLogger(ROOT_LOGGER_NAME)

    if _configured and not force:
        return logger

    if level is None:
        level = os.getenv("HARRISON_LOG_LEVEL", "INFO")
    if isinstance(level, str):
        level = logging.getLevelName(level.strip().upper())
        if not isinstance(level, int):
            level = logging.INFO

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    from backend.observability import RequestIdFilter

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    # On the handler, not the logger: filters on a logger do not apply to
    # records propagated from its children.
    handler.addFilter(RequestIdFilter())
    logger.addHandler(handler)
    logger.setLevel(level)

    # The handler above is the only one that should emit these records; without
    # this, anything that later configures root would print them a second time.
    logger.propagate = False

    _configured = True
    return logger
