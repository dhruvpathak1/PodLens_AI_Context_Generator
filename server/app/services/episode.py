"""End-to-end processing of one uploaded episode: transcribe -> save -> tag entities -> save.

This is the work behind `POST /api/transcribe`. It is deliberately synchronous: Whisper and
spaCy are CPU-bound and not async-aware.

Note: the route currently calls this directly on the event loop, so the API cannot answer other
requests while an upload is processing. Wrapping the call in `asyncio.to_thread` (plus a lock
around the shared Whisper model) is the planned fix.
"""

from __future__ import annotations

import logging
import tempfile
from typing import Any

from app.core.config import settings
from app.services import storage
from app.services.audio import normalized_suffix, remove_quietly
from app.services.entities import extract_document, save_document
from app.services.transcription import placeholder_transcript, transcribe_file

logger = logging.getLogger(__name__)


def process_upload(
    contents: bytes,
    filename: str,
    *,
    language: str | None = None,
    extract_entities: bool = True,
    entity_backend: str | None = None,
) -> dict[str, Any]:
    """Transcribe an uploaded audio file and tag its entities.

    Decoding/transcription failures do not raise: the response carries a single
    "Unknown Sentence" placeholder so the UI can still render, and entity tagging is skipped.
    Entity failures are reported in `entity_error` while the transcript is still returned.

    Returns:
        `{transcript, segments, saved_path, document, entity_saved_path, entity_error}`.
    """
    tmp_path: str | None = None
    try:
        # 1) Write the upload to a temp file with a sensible extension for ffmpeg.
        with tempfile.NamedTemporaryFile(suffix=normalized_suffix(filename, ".wav"), delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(contents)

        # 2) Speech-to-text (falls back to a placeholder on failure).
        transcription_failed = False
        try:
            text, segments = transcribe_file(tmp_path, language)
        except Exception as e:
            logger.warning("transcribe/decode failed; using placeholder: %s", e)
            text, segments = placeholder_transcript()
            transcription_failed = True

        # 3) Keep a human-readable copy of the transcript on disk.
        stem = storage.safe_stem(filename)
        saved_path = storage.save_transcript(settings.transcripts_dir, stem, text, segments)

        # 4) Named-entity tagging (skipped for placeholders).
        document: dict[str, Any] | None = None
        entity_saved_path: str | None = None
        entity_error: str | None = None
        if extract_entities and segments and not transcription_failed:
            try:
                document = extract_document(segments, source_label=filename, backend=entity_backend)
                entity_saved_path = str(save_document(document, settings.entity_json_dir, stem))
            except Exception as e:
                logger.exception("entity extraction after transcribe failed")
                entity_error = str(e)

        return {
            "transcript": text,
            "segments": segments,
            "saved_path": str(saved_path),
            "document": document,
            "entity_saved_path": entity_saved_path,
            "entity_error": entity_error,
        }
    finally:
        remove_quietly(tmp_path)
