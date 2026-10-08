"""`POST /api/transcribe`: upload an audio file, get back transcript segments and entities."""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.services.episode import process_upload

logger = logging.getLogger(__name__)

router = APIRouter(tags=["transcription"])


@router.post("/api/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    language: str | None = Form(None),
    extract_entities: bool = Form(True),
    entity_backend: str | None = Form(None),
) -> dict:
    """Transcribe an uploaded file with Whisper, then tag entities.

    Form fields:
        audio: the audio file (mp3, wav, m4a, webm, ogg, flac, mp4).
        language: optional ISO code, skips language detection.
        extract_entities: set false to return the transcript only.
        entity_backend: "spacy" | "claude" (server default when omitted).

    Response: `{transcript, segments[], saved_path, document, entity_saved_path, entity_error}`.
    """
    if not audio.filename:
        raise HTTPException(status_code=400, detail="Missing file name")
    contents = await audio.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")
    try:
        return process_upload(
            contents,
            audio.filename,
            language=language,
            extract_entities=extract_entities,
            entity_backend=entity_backend,
        )
    except Exception as e:
        logger.exception("transcribe failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
