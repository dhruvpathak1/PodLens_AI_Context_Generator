"""PARKED: `POST /api/transcribe-chunk`, the legacy live mode (10 s WebM slices -> Whisper).

Superseded by the OpenAI Realtime relay (`realtime_relay.py`) but kept as the fallback the
frontend uses when no OpenAI key is configured.
"""

from __future__ import annotations

import logging
import tempfile
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.core.config import env_float, settings
from app.services import storage
from app.services.audio import normalized_suffix, remove_quietly
from app.services.entities import extract_document, save_document
from app.services.transcription import placeholder_transcript, transcribe_file

logger = logging.getLogger(__name__)

#: Length of one live slice; used to size the placeholder when a slice cannot be decoded.
LIVE_CHUNK_WINDOW_SEC = env_float("LIVE_CHUNK_WINDOW_SEC", 10.0)

router = APIRouter()


def offset_segments(segments: list[dict[str, Any]], *, time_offset_sec: float, chunk_seq: int) -> list[dict[str, Any]]:
    """Shift slice-relative segments onto the session timeline.

    Ids become `chunk_seq * 1000 + index` so they stay unique across slices.
    """
    base = max(0, int(chunk_seq)) * 1000
    return [
        {
            "id": base + i,
            "start": float(seg["start"]) + float(time_offset_sec),
            "end": float(seg["end"]) + float(time_offset_sec),
            "text": seg["text"],
        }
        for i, seg in enumerate(segments)
    ]


@router.post("/api/transcribe-chunk")
async def transcribe_chunk(
    audio: UploadFile = File(...),
    time_offset_sec: float = Form(0.0),
    chunk_seq: int = Form(0),
    language: str | None = Form(None),
    extract_entities: bool = Form(True),
    entity_backend: str | None = Form(None),
    persist_transcript: bool = Form(False),
) -> dict:
    """Transcribe one timed slice of live mic audio and tag its entities."""
    if not audio.filename:
        raise HTTPException(status_code=400, detail="Missing file name")

    tmp_path: str | None = None
    try:
        contents = await audio.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")
        with tempfile.NamedTemporaryFile(suffix=normalized_suffix(audio.filename, ".webm"), delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(contents)

        decode_placeholder = False
        try:
            text, raw_segments = transcribe_file(tmp_path, language)
        except Exception as e:
            logger.warning("transcribe-chunk %s failed; using placeholder: %s", int(chunk_seq), e)
            text, raw_segments = placeholder_transcript(LIVE_CHUNK_WINDOW_SEC)
            decode_placeholder = True
        segments = offset_segments(raw_segments, time_offset_sec=time_offset_sec, chunk_seq=chunk_seq)

        stem = storage.safe_stem(audio.filename or "live_chunk")
        saved_path: str | None = None
        if persist_transcript and segments:
            saved_path = str(storage.save_transcript(settings.transcripts_dir, f"{stem}_chunk{chunk_seq}", text, segments))

        document: dict | None = None
        entity_saved_path: str | None = None
        entity_error: str | None = None
        if extract_entities and segments and not decode_placeholder:
            try:
                label = f"{audio.filename or 'live'}#chunk{chunk_seq}"
                document = extract_document(segments, source_label=label, backend=entity_backend)
                if persist_transcript:
                    entity_saved_path = str(save_document(document, settings.entity_json_dir, stem))
            except Exception as e:
                logger.exception("entity extraction after transcribe-chunk failed")
                entity_error = str(e)

        return {
            "transcript": text,
            "segments": segments,
            "saved_path": saved_path,
            "document": document,
            "entity_saved_path": entity_saved_path,
            "entity_error": entity_error,
            "chunk_seq": int(chunk_seq),
            "time_offset_sec": float(time_offset_sec),
            "decode_placeholder": decode_placeholder,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("transcribe-chunk failed")
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        remove_quietly(tmp_path)
