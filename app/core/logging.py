"""
app/core/logging.py
────────────────────
Centralised logger factory — identical pattern to ChatPDF.

All modules call `get_logger(__name__)` instead of `logging.getLogger`.
`configure_logging()` is called ONCE at app startup inside the lifespan.
Outputs structured JSON when python-json-logger is available (production),
falls back to clean plain-text for local dev without extra installs.
"""

from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "info") -> None:
    """
    Configure the root logger once at application startup.
    JSON output is compatible with log aggregators (Datadog, Cloud Logging, Loki).
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)

    try:
        from pythonjsonlogger.json import JsonFormatter

        formatter = JsonFormatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    except ImportError:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )

    handler.setFormatter(formatter)
    root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger. Usage: logger = get_logger(__name__)"""
    return logging.getLogger(name)
