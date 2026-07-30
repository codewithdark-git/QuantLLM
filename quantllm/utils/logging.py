"""
Production logging system for QuantLLM.

Provides structured logging with JSON support, file rotation,
and environment variable configuration -- all without external
dependencies beyond the standard library.

Environment variables:
    QUANTLLM_LOG_LEVEL      - One of DEBUG, INFO, WARNING, ERROR (default: WARNING)
    QUANTLLM_LOG_FORMAT     - "text" or "json" (default: "text")
    QUANTLLM_LOG_FILE       - Path to log file (default: none, stderr only)
    QUANTLLM_LOG_MAX_BYTES  - Max bytes per log file before rotation (default: 10485760)
    QUANTLLM_LOG_BACKUP     - Number of backup files to keep (default: 3)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_LOG_TEXT_FORMAT = "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s"
_QUANTLLM_LOGGER_NAME = "quantllm"

# ---------------------------------------------------------------------------
# JSON formatter for machine-parseable output
# ---------------------------------------------------------------------------


class _JSONFormatter(logging.Formatter):
    """Format log records as newline-delimited JSON."""

    def format(self, record: logging.LogRecord) -> str:
        return json.dumps(
            {
                "timestamp": self.formatTime(record, _LOG_DATE_FORMAT),
                "level": record.levelname,
                "name": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
            }
        )


# ---------------------------------------------------------------------------
# Stderr handler with flush-on-every-write semantics
# ---------------------------------------------------------------------------


class _StderrHandler(logging.Handler):
    """Write formatted records to stderr, flushing after each line.

    This avoids the line-buffering issues that ``StreamHandler`` can
    exhibit when stderr is redirected to a pipe or file.
    """

    def __init__(self, formatter: Optional[logging.Formatter] = None) -> None:
        super().__init__()
        self._formatter = formatter or logging.Formatter(
            _LOG_TEXT_FORMAT, datefmt=_LOG_DATE_FORMAT
        )
        self.setFormatter(self._formatter)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            sys.stderr.write(self.format(record) + "\n")
            sys.stderr.flush()
        except Exception:
            self.handleError(record)


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def _env(key: str, default: str) -> str:
    return os.environ.get(f"QUANTLLM_LOG_{key}", default)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def configure_logging(
    level: Optional[str] = None,
    fmt: Optional[str] = None,
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 3,
) -> logging.Logger:
    """Configure the QuantLLM logger.

    Parameters
    ----------
    level:
        One of ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``.  Falls back to
        the ``QUANTLLM_LOG_LEVEL`` environment variable, then ``WARNING``.
    fmt:
        ``"text"`` (human-readable) or ``"json"`` (newline-delimited JSON).
        Falls back to ``QUANTLLM_LOG_FORMAT``, then ``"text"``.
    log_file:
        Path to a log file.  Falls back to ``QUANTLLM_LOG_FILE``.
        When ``None`` (or unset), output goes to stderr only.
    max_bytes:
        Maximum size of a single log file before rotation.  Falls back to
        ``QUANTLLM_LOG_MAX_BYTES``.  Default 10 MB.
    backup_count:
        Number of old log files to retain.  Falls back to
        ``QUANTLLM_LOG_BACKUP``.  Default 3.

    Safe to call multiple times -- previous handlers are removed first.
    """
    resolved_level = (level or _env("LEVEL", "WARNING")).upper()
    resolved_fmt = (fmt or _env("FORMAT", "text")).lower()
    resolved_file = log_file or _env("FILE", "") or None

    logger = logging.getLogger(_QUANTLLM_LOGGER_NAME)
    logger.handlers.clear()
    logger.setLevel(getattr(logging, resolved_level, logging.WARNING))
    logger.propagate = False

    text_fmt = logging.Formatter(_LOG_TEXT_FORMAT, datefmt=_LOG_DATE_FORMAT)
    json_fmt = _JSONFormatter()

    # Stderr handler (always present)
    stderr_handler = _StderrHandler(json_fmt if resolved_fmt == "json" else text_fmt)
    logger.addHandler(stderr_handler)

    # Optional rotating file handler
    if resolved_file:
        resolved_max = int(_env("MAX_BYTES", str(max_bytes)))
        resolved_backup = int(_env("BACKUP", str(backup_count)))
        file_fmt = json_fmt if resolved_fmt == "json" else text_fmt
        file_handler = RotatingFileHandler(
            resolved_file, maxBytes=resolved_max, backupCount=resolved_backup
        )
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = _QUANTLLM_LOGGER_NAME) -> logging.Logger:
    """Return a logger for the qualified *name*.

    Child loggers such as ``"quantllm.server"`` inherit the parent's
    configuration automatically (handlers, level, propagation).
    """
    return logging.getLogger(name)


logger: logging.Logger = get_logger()


def enable_logging(level: str = "INFO") -> None:
    """Alias for :func:`configure_logging` (backwards compatibility)."""
    configure_logging(level)
