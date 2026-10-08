"""FastAPI application factory.

`create_app()` wires middleware, routers and (optionally) the built frontend. The module-level
`app` is what uvicorn serves (`server/main.py` re-exports it so `uvicorn main:app` keeps working).
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import ALL_ROUTERS
from app.core.config import settings
from app.core.logging import configure_logging

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Build the PodLens API application."""
    configure_logging()
    application = FastAPI(title="PodLens API", version="1.0.0")

    # Browsers on another origin (Vite dev server, GitHub Pages) may call the API.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    for router in ALL_ROUTERS:
        application.include_router(router)

    # Live microphone mode is parked (future scope). Its routes are only mounted on request.
    if settings.enable_live_mode:
        from app.live import build_live_router

        application.include_router(build_live_router())
        logger.info("Live mode enabled: /api/live and /api/transcribe-chunk mounted")

    _mount_frontend(application)
    return application


def _mount_frontend(application: FastAPI) -> None:
    """Serve the built React app from `STATIC_DIST_DIR` (used by the Docker image).

    Mounted last so `/api/*` routes take precedence over the catch-all static handler.
    """
    if not settings.static_dist_dir:
        return
    static_dir = Path(settings.static_dist_dir)
    if static_dir.is_dir():
        from fastapi.staticfiles import StaticFiles

        application.mount("/", StaticFiles(directory=str(static_dir), html=True), name="spa")


app = create_app()
