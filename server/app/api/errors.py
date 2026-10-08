"""Shared HTTP error helpers for routes."""

from __future__ import annotations

from fastapi import HTTPException

#: Shown when an optional AI dependency (LangChain) is missing or broken.
INSTALL_HINT = "Run: venv/bin/python -m pip install -r server/requirements.txt"


def feature_unavailable(feature: str, error: Exception) -> HTTPException:
    """500 error used when an optional feature module cannot be imported."""
    return HTTPException(status_code=500, detail=f"{feature} unavailable ({error}). {INSTALL_HINT}")


def missing_dependency(error: Exception) -> HTTPException:
    """500 error used when a dependency import fails while the feature is running."""
    return HTTPException(status_code=500, detail=f"{error}. {INSTALL_HINT}")
