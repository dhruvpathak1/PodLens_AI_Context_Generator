"""PARKED: live microphone transcription (future scope).

Nothing in this package is imported by the running API unless `ENABLE_LIVE_MODE=true`.
See `README.md` in this folder for what it contains and how to re-enable it.
"""

from __future__ import annotations

from fastapi import APIRouter, WebSocket


def build_live_router() -> APIRouter:
    """Router with the two live endpoints. Imported lazily so the parked code costs nothing."""
    from app.live.chunk_transcription import router as chunk_router
    from app.live.realtime_relay import handle_live_socket

    router = APIRouter(tags=["live (parked)"])
    router.include_router(chunk_router)

    @router.websocket("/api/live")
    async def live_socket(ws: WebSocket) -> None:
        """Live mic -> OpenAI Realtime relay (PCM16 mono 24 kHz binary frames in, JSON events out)."""
        await handle_live_socket(ws)

    return router
