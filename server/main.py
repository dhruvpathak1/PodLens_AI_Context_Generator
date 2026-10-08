"""ASGI entry point. Run from the `server/` directory:

    uvicorn main:app --reload --port 8000

The application itself lives in the `app` package (see `app/main.py`).
"""

from app.main import app

__all__ = ["app"]
