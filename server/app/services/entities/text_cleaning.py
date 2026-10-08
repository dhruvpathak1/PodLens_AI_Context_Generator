"""Transcript clean-up and span helpers shared by every NER backend."""

from __future__ import annotations

import re

# Filler words and verbal tics removed before NER (they confuse both spaCy and LLMs).
_FILLER_PATTERN = re.compile(
    r"\b("
    r"um|uh|er|ah|hmm|hm|mm|uhm|erm|"
    r"like(?=\s+(?:i|you|the|a|an|this|that|it)\b)|"  # "like" only as a filler, not the verb
    r"you know|i mean|i guess|sort of|kind of|"
    r"actually|basically|literally|obviously|honestly|"
    r"right\?|okay\?|ok\?"
    r")\b",
    re.IGNORECASE,
)

# Throwaway phrases (YouTube-style outros) that never contain useful entities.
_NOISE_PHRASES = re.compile(
    r"\b(thanks for watching|subscribe|like and subscribe|see you next time)\b",
    re.IGNORECASE,
)

#: Entity strings too generic to be worth a card, compared case-insensitively.
GENERIC_ENTITIES = frozenset(
    {
        "today", "tomorrow", "here", "there", "now",
        "something", "someone", "anything", "everything", "nothing",
        "people", "things", "stuff", "way", "thing", "lot", "bit",
    }
)  # fmt: skip


def filter_noise(text: str) -> str:
    """Strip filler words and throwaway phrases, then collapse whitespace."""
    if not text or not text.strip():
        return ""
    cleaned = _NOISE_PHRASES.sub("", text)
    cleaned = _FILLER_PATTERN.sub("", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def clean_entity_text(text: str) -> str | None:
    """Return a trimmed entity string, or None if it is too short or too generic to keep."""
    stripped = text.strip()
    if len(stripped) < 2 or stripped.lower() in GENERIC_ENTITIES:
        return None
    return stripped


def entity_time_span(
    chunk_start: float, chunk_end: float, chunk_text: str, ent_start_char: int, ent_end_char: int
) -> tuple[float, float]:
    """Estimate when an entity is spoken from its character position inside a timed chunk.

    Whisper only times whole segments, so we interpolate linearly: an entity halfway through
    the text is assumed to be spoken halfway through the segment.
    """
    n = max(len(chunk_text), 1)
    duration = max(chunk_end - chunk_start, 0.0)
    rel_start = max(0, min(ent_start_char, n)) / n
    rel_end = max(0, min(ent_end_char, n)) / n
    return chunk_start + rel_start * duration, chunk_start + rel_end * duration
