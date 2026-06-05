"""Logging configuration for the rehabilitation system."""

import logging


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the 'rehabilitation' namespace."""
    return logging.getLogger(f"rehabilitation.{name}")


def configure_logging(level: int = logging.INFO) -> None:
    """Set up the root rehabilitation logger with a console handler.

    Safe to call multiple times — won't add duplicate handlers.
    """
    root = logging.getLogger("rehabilitation")
    if root.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(levelname)s [%(name)s] %(message)s")
    )
    root.addHandler(handler)
    root.setLevel(level)
