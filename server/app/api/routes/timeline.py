"""`POST /api/timeline`: chronological timeline of everything mentioned (LangChain + OpenAI)."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.api.errors import feature_unavailable, missing_dependency
from app.api.schemas import TimelineRequest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ai"])


@router.post("/api/timeline")
async def timeline(body: TimelineRequest) -> dict:
    """Generate the episode timeline from transcript segments and extracted entities.

    Response: `{model, events[], windows, failed_windows}`.
    Errors: 400 empty input, 413 transcript too long, 503 no OpenAI key, 502 model failure.
    """
    if not body.segments:
        raise HTTPException(status_code=400, detail="segments must be non-empty")
    try:
        # Imported here so a missing/broken LangChain install can never stop the API from starting.
        from app.services.timeline import TimelineNotConfigured, build_timeline
    except Exception as e:
        logger.exception("timeline module failed to import")
        raise feature_unavailable("Timeline", e) from e

    try:
        return await build_timeline(
            [s.model_dump() for s in body.segments],
            [e.model_dump() for e in body.entities],
            body.source_label,
        )
    except TimelineNotConfigured as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except ImportError as e:
        raise missing_dependency(e) from e
    except ValueError as e:
        raise HTTPException(status_code=413, detail=str(e)) from e
    except Exception as e:
        logger.exception("timeline generation failed")
        raise HTTPException(status_code=502, detail=f"Timeline generation failed: {e}") from e
