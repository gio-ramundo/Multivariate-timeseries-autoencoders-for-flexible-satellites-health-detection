"""Shared logger used by all pipeline scripts."""

from __future__ import annotations

import logging
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Create (or retrieve) a logger with a console handler and, if specified, a file handler.

    Repeated calls with the same `name` do not duplicate handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file is not None:
        log_file = Path(log_file).resolve()
        current_file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
        if not any(Path(h.baseFilename) == log_file for h in current_file_handlers):
            # `logging.getLogger(name)` is a process-wide registry: since several
            # call sites reuse the same logger name across architectures/datasets
            # (e.g. "bayesian_optimizer"), a call with a new log_file must replace
            # the stale handler, otherwise logging silently keeps writing to the
            # first path it was ever given.
            for h in current_file_handlers:
                logger.removeHandler(h)
                h.close()
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    logger.propagate = False
    return logger
