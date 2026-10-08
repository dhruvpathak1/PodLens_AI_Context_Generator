"""`GET /api/health`: liveness plus a summary of which optional features are configured."""

from __future__ import annotations

from fastapi import APIRouter

from app.core.config import settings
from app.services.entities import resolve_entity_backend

router = APIRouter(tags=["health"])


@router.get("/api/health")
def health() -> dict:
    """Report server status and feature configuration (never exposes secret values)."""
    return {
        "ok": True,
        "model": settings.whisper_model,
        "entity_backend": resolve_entity_backend(None),
        "unsplash_configured": bool(settings.unsplash_access_key),
        "openai_configured": settings.openai_configured,
        "live_mode": settings.enable_live_mode,
        # Kept for the parked live-mode frontend, which checks this before streaming audio.
        "realtime_transcription": settings.enable_live_mode and settings.openai_configured,
    }
