"""FastAPI routers, one module per feature. `ALL_ROUTERS` is mounted by `app.main`."""

from app.api.routes import ask, entities, health, timeline, transcription

ALL_ROUTERS = [
    health.router,
    transcription.router,
    entities.router,
    timeline.router,
    ask.router,
]
