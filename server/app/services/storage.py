"""Writing transcripts and entity exports to local disk.

Files are named `<safe-stem>_<UTC timestamp>.<ext>` so repeated runs never overwrite each other.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: Longest filename stem we keep (the timestamp and extension are added after).
_MAX_STEM_LEN = 80


def safe_stem(filename: str, fallback: str = "audio") -> str:
    """Turn an uploaded filename into a filesystem-safe stem.

    Example: "My Episode #12.mp3" -> "My_Episode__12".
    """
    stem = Path(filename).stem
    stem = re.sub(r"[^\w\-.]", "_", stem, flags=re.UNICODE)[:_MAX_STEM_LEN]
    return stem or fallback


def utc_timestamp() -> str:
    """Compact UTC timestamp used in output filenames (e.g. 20261008_174512)."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def format_clock(sec: float) -> str:
    """Format seconds as HH:MM:SS.mmm for human-readable transcript files."""
    sec = max(0.0, float(sec))
    ms = int(round((sec % 1) * 1000))
    total = int(sec)
    hours, rem = divmod(total, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{ms:03d}"


def transcript_file_body(text: str, segments: list[dict[str, Any]]) -> str:
    """Render a transcript as one `[start -> end] text` line per segment (plain text when untimed)."""
    if segments:
        lines = [f"[{format_clock(s['start'])} → {format_clock(s['end'])}] {s['text']}" for s in segments]
        return "\n".join(lines) + "\n"
    return text + ("\n" if text else "")


def save_transcript(directory: Path, stem: str, text: str, segments: list[dict[str, Any]]) -> Path:
    """Write a timestamped transcript `.txt` file and return its absolute path."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{stem}_{utc_timestamp()}.txt"
    path.write_text(transcript_file_body(text, segments), encoding="utf-8")
    return path.resolve()


def save_json(directory: Path, basename: str, payload: dict[str, Any]) -> Path:
    """Write `payload` as pretty-printed UTF-8 JSON and return its absolute path."""
    directory.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^\w\-.]", "_", basename, flags=re.UNICODE)[:_MAX_STEM_LEN] or "entities"
    path = directory / f"{safe}_{utc_timestamp()}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path.resolve()
