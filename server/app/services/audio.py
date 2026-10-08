"""Audio decoding helpers built on ffmpeg."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

from app.core.config import settings

#: File extensions accepted for upload; anything else is saved with `default_suffix`.
SUPPORTED_SUFFIXES = frozenset({".mp3", ".wav", ".m4a", ".webm", ".ogg", ".flac", ".mp4", ".mpeg", ".mpga"})

#: How much of ffmpeg's stderr to surface in error messages.
_STDERR_TAIL_CHARS = 3500


def ffmpeg_binary() -> str:
    """Return the ffmpeg executable: `FFMPEG_PATH`, else the one on PATH, else plain `ffmpeg`."""
    return settings.ffmpeg_path or shutil.which("ffmpeg") or "ffmpeg"


def normalized_suffix(filename: str, default_suffix: str = ".wav") -> str:
    """Return the lower-cased extension of `filename` if supported, otherwise `default_suffix`."""
    suffix = os.path.splitext(filename)[1].lower()
    return suffix if suffix in SUPPORTED_SUFFIXES else default_suffix


def decode_to_whisper_wav(src_path: str) -> str:
    """Decode any browser/container audio file to mono 16 kHz PCM WAV and return the new path.

    Whisper's built-in loader pipes ffmpeg stdout, which fails on fragmented WebM produced by
    `MediaRecorder`. Decoding to a temporary file first is slower but far more reliable.
    The caller owns the returned file and must delete it.

    Raises:
        RuntimeError: ffmpeg exited non-zero (stderr tail included in the message).
    """
    fd, wav_path = tempfile.mkstemp(suffix=".whisper16k.wav")
    os.close(fd)
    cmd = [
        ffmpeg_binary(),
        "-hide_banner",
        "-loglevel", "warning",
        "-y",  # overwrite the temp file we just created
        "-nostdin",  # never block waiting for keyboard input
        "-threads", "0",
        "-fflags", "+genpts+discardcorrupt",  # tolerate broken timestamps / corrupt packets
        "-err_detect", "ignore_err",
        "-i", src_path,
        "-vn",  # drop any video stream (e.g. .mp4 uploads)
        "-ac", "1",  # mono
        "-ar", "16000",  # 16 kHz, Whisper's native rate
        "-acodec", "pcm_s16le",
        "-f", "wav",
        wav_path,
    ]  # fmt: skip
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        stderr = (result.stderr or b"").decode(errors="replace").strip()
        remove_quietly(wav_path)
        tail = stderr[-_STDERR_TAIL_CHARS:] if stderr else "(no stderr)"
        raise RuntimeError(f"ffmpeg could not decode audio (exit {result.returncode}): {tail}") from None
    return wav_path


def remove_quietly(path: str | None) -> None:
    """Delete a temp file, ignoring 'already gone' and permission errors."""
    if path and os.path.isfile(path):
        try:
            os.unlink(path)
        except OSError:
            pass
