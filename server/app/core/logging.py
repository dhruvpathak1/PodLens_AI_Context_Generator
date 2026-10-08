"""Logging setup shared by the API process."""

from __future__ import annotations

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging once (uvicorn adds its own handlers on top)."""
    logging.basicConfig(level=level, format="%(levelname)s:%(name)s: %(message)s")
