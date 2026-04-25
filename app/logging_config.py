from __future__ import annotations

import logging
import os
import sys

DEFAULT_LOG_LEVEL = "INFO"
LOG_LEVEL_ENV_VAR = "DIPLOMA_LOG_LEVEL"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_MESSAGE_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_HANDLER_MARK = "_diploma_console_handler"


def _resolve_log_level() -> int:
    level_name = os.getenv(LOG_LEVEL_ENV_VAR, DEFAULT_LOG_LEVEL).strip().upper()
    return int(getattr(logging, level_name, logging.INFO))


def configure_logging() -> None:
    """Configure process-wide console logging once for the project."""
    root_logger = logging.getLogger()

    has_project_handler = any(
        getattr(handler, _HANDLER_MARK, False) for handler in root_logger.handlers
    )
    if not has_project_handler:
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(logging.Formatter(fmt=_MESSAGE_FORMAT, datefmt=_DATE_FORMAT))
        setattr(handler, _HANDLER_MARK, True)
        root_logger.addHandler(handler)

    root_logger.setLevel(_resolve_log_level())


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
