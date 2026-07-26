"""Structured JSON logging (PRD Module 15).

One `configure_logging()` call, made by the API factory and by the CLI, so a log
line has the same shape whether it came from a request handler, a nightly cron
job or an interactive backtest. JSON in production because a log aggregator
parses fields, not prose; a human-readable console renderer in development
because nobody reads JSON at a terminal by choice.

**Request ids.** Every log line emitted while handling a request carries the
same ``request_id``, bound into a :mod:`contextvars` context rather than passed
down through call signatures. That matters here specifically: a backtest logs
from inside the engine, several layers below the route that triggered it, and
threading an id through `engine -> ledger -> cost model` to make the trail
joinable would be absurd. The id is also returned to the caller as the
``X-Request-ID`` response header, so a user reporting "the holdout endpoint
refused me" can hand over one token that finds every line.
"""

from __future__ import annotations

import logging
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager

import structlog
from structlog.contextvars import (
    bind_contextvars,
    clear_contextvars,
    merge_contextvars,
)

from quantlab.config import get_settings

#: Header carrying the id in and out. Inbound values are honoured so a trace can
#: span the frontend, the API and a worker; a missing or malformed one is
#: replaced rather than trusted.
REQUEST_ID_HEADER = "X-Request-ID"

_configured = False


def configure_logging(*, json_logs: bool | None = None, force: bool = False) -> None:
    """Configure structlog and the stdlib root logger.

    Idempotent: repeated calls are ignored unless ``force``, because the API
    factory may be invoked more than once in a test session and reconfiguring
    would stack duplicate processors.

    ``json_logs`` defaults to "JSON unless a TTY is attached" — containers and CI
    get machine-readable output, an interactive shell gets a readable console.
    """

    global _configured
    if _configured and not force:
        return

    settings = get_settings()
    use_json = json_logs if json_logs is not None else not sys.stderr.isatty()

    shared: list[structlog.typing.Processor] = [
        merge_contextvars,  # pulls request_id (and anything else bound) into every line
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer()
        if use_json
        # exc_info is formatted by the renderer in dev; in JSON mode
        # format_exc_info below turns it into a string field instead.
        else structlog.dev.ConsoleRenderer(colors=False)
    )
    if use_json:
        shared.append(structlog.processors.format_exc_info)

    structlog.configure(
        processors=[*shared, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping().get(settings.log_level.upper(), logging.INFO)
        ),
        # stdlib, not PrintLogger: `add_logger_name` reads `logger.name`, which
        # only a stdlib logger has, and routing through stdlib means uvicorn's and
        # SQLAlchemy's records share this one handler instead of bypassing it.
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # uvicorn/sqlalchemy log through the stdlib; keep them at the same level so
    # one setting governs everything rather than two half-configured systems.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=settings.log_level.upper(),
        force=True,
    )
    _configured = True


def new_request_id() -> str:
    """A short, log-greppable id. UUID4 truncated — collisions do not matter for
    correlation within a retention window, and 32 hex characters in every line is
    noise."""

    return uuid.uuid4().hex[:16]


@contextmanager
def request_context(request_id: str | None = None) -> Iterator[str]:
    """Bind ``request_id`` for the duration of the block.

    Used by the API middleware and by the CLI, so a nightly `quantlab ingest`
    run produces a correlatable trail exactly as a request does.
    """

    resolved = request_id or new_request_id()
    bind_contextvars(request_id=resolved)
    try:
        yield resolved
    finally:
        clear_contextvars()
