"""Logging utilities for pico-agent.

All pico-agent loggers live under the ``pico_agent`` namespace.  Use
``get_logger()`` to obtain a namespaced logger and ``configure_logging()``
to set the level and handler for the entire library.
"""

import logging
import sys
from typing import Optional

DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
"""str: Default log format used by ``configure_logging``."""


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the pico_agent namespace.

    Args:
        name: Logger name. If not prefixed with 'pico_agent', it will be added.

    Returns:
        A configured Logger instance.
    """
    if not name.startswith("pico_agent"):
        name = f"pico_agent.{name}"
    return logging.getLogger(name)


def configure_logging(level: int = logging.INFO, handler: Optional[logging.Handler] = None) -> None:
    """Configure logging for the pico_agent library.

    Args:
        level: Logging level (default: INFO)
        handler: Custom handler. If None, uses StreamHandler to stderr.
    """
    root_logger = logging.getLogger("pico_agent")
    root_logger.setLevel(level)

    if not root_logger.handlers:
        if handler is None:
            handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(DEFAULT_FORMAT))
        root_logger.addHandler(handler)
