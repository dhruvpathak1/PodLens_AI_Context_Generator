"""`POST /api/ask`: grounded Q&A over the episode with validated timestamp citations."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from app.api.errors import feature_unavailable, missing_dependency
from app.api.schemas import AskRequest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["ai"])


@router.post("/api/ask")
async def ask(body: AskRequest) -> dict:
    """Answer a question using only the transcript.

    Response: `{answer, found, citations[{start_sec, quote}], mode, model, unverified}`.
    Errors: 400 bad question/transcript, 503 no OpenAI key, 502 model failure.
    """
    try:
        # Imported here so a missing/broken LangChain install can never stop the API from starting.
        from app.services.ask import AskNotConfigured, answer_question
    except Exception as e:
        logger.exception("ask module failed to import")
        raise feature_unavailable("Ask", e) from e

    try:
        return await answer_question(
            body.question,
            [s.model_dump() for s in body.segments],
            [e.model_dump() for e in body.entities],
            [t.model_dump() for t in body.history],
            body.source_label,
        )
    except AskNotConfigured as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except ImportError as e:
        raise missing_dependency(e) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("ask failed")
        raise HTTPException(status_code=502, detail=f"Could not answer: {e}") from e
