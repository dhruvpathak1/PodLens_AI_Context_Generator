"""Speech-to-text with OpenAI Whisper (runs locally, no API calls).

The Whisper model is loaded lazily on first use and cached for the life of the process.
`whisper` (and therefore PyTorch) is imported inside `get_model()` so the API can start,
and tests can run, without loading several hundred MB of ML libraries.
"""

from __future__ import annotations

import logging
from typing import Any

from app.core.config import settings
from app.services.audio import decode_to_whisper_wav, remove_quietly

logger = logging.getLogger(__name__)

#: Text shown in place of a transcript when audio could not be decoded or transcribed.
UNKNOWN_TRANSCRIPT_SENTENCE = "Unknown Sentence"

_model: Any = None


def get_model() -> Any:
    """Return the cached Whisper model, loading it (and downloading weights) on first call."""
    global _model
    if _model is None:
        import whisper  # heavy import: pulls in torch

        logger.info("Loading Whisper model %r (first request may take a while)", settings.whisper_model)
        settings.whisper_download_root.mkdir(parents=True, exist_ok=True)
        _model = whisper.load_model(settings.whisper_model, download_root=str(settings.whisper_download_root))
    return _model


def segments_from_result(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert Whisper's raw `segments` into `{id, start, end, text}` dicts, skipping empty text."""
    out: list[dict[str, Any]] = []
    for i, seg in enumerate(result.get("segments") or []):
        if not isinstance(seg, dict):
            continue
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        out.append(
            {
                "id": i,
                "start": float(seg.get("start") or 0.0),
                "end": float(seg.get("end") or 0.0),
                "text": text,
            }
        )
    return out


def transcribe_file(path: str, language: str | None = None) -> tuple[str, list[dict[str, Any]]]:
    """Transcribe an audio file on disk.

    Args:
        path: Any audio/video file ffmpeg can read.
        language: Optional ISO code to skip Whisper's language detection (e.g. "en").

    Returns:
        `(full_text, segments)` where each segment is `{id, start, end, text}` in seconds.

    Raises:
        RuntimeError: ffmpeg could not decode the file.
    """
    decoded = decode_to_whisper_wav(path)
    try:
        options: dict[str, Any] = {}
        if language and language.strip():
            options["language"] = language.strip()
        result = get_model().transcribe(decoded, **options)
        text = (result.get("text") or "").strip()
        return text, segments_from_result(result)
    finally:
        if decoded != path:
            remove_quietly(decoded)


def placeholder_transcript(duration_sec: float = 1.0) -> tuple[str, list[dict[str, Any]]]:
    """Return the `(text, segments)` pair used when transcription fails, so the UI still renders."""
    return UNKNOWN_TRANSCRIPT_SENTENCE, [
        {"id": 0, "start": 0.0, "end": float(duration_sec), "text": UNKNOWN_TRANSCRIPT_SENTENCE}
    ]
