"""Chronological timeline of everything mentioned in an episode (LangChain + OpenAI structured output).

Flow
0. Runs AFTER transcription + NER: the entities PodLens already extracted are passed to the model as
   anchors, so the timeline is built around them (and spelled the same way).
1. Timestamped Whisper segments are packed into windows of ~TIMELINE_WINDOW_CHARS characters,
   each line prefixed with its start time: "[t=123.4] text".
2. One LangChain chain per window, run concurrently with `abatch`:
       ChatPromptTemplate | ChatOpenAI(...).with_structured_output(TimelineWindow, strict=True)
   The model returns validated Pydantic objects (OpenAI json_schema structured output),
   so the API never has to parse free-form JSON.
3. Events from all windows are merged (de-duplicated), their `mentioned_at_sec` is snapped to a
   real segment start, and each gets a stable id. The client sorts by history or by episode order.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.core.config import settings

logger = logging.getLogger(__name__)

# Module-level copies of the settings so tests can tweak them without rebuilding Settings.
TIMELINE_MODEL = settings.timeline_model
TIMELINE_WINDOW_CHARS = settings.timeline_window_chars
TIMELINE_MAX_CONCURRENCY = settings.timeline_max_concurrency
TIMELINE_MAX_TRANSCRIPT_CHARS = settings.timeline_max_transcript_chars

Category = Literal["COMPANY", "PERSON", "PRODUCT", "TECHNOLOGY", "EVENT", "PLACE", "DEAL", "OTHER"]


# ---------- Structured output schema (what the LLM must return) ----------
# Every field is required (nullable where optional) so the schema is valid in OpenAI strict mode.


class TimelineEvent(BaseModel):
    """One dated (or undated) thing mentioned in the episode, as returned by the model."""

    title: str = Field(description="Headline of the event, at most 10 words. No trailing period.")
    description: str = Field(
        description="One or two sentences on what happened and why it matters, based only on the transcript."
    )
    category: Category = Field(description="Best-fitting category for the event's main subject.")
    date_label: str | None = Field(
        description="When it happened in the real world, as precise as the transcript allows "
        "(e.g. '1993', 'March 2007', 'late 1990s', 'Q2 2021'). Null if the speakers give no date "
        "and it cannot be inferred from what they say."
    )
    year: int | None = Field(description="Best estimate of the real-world year, or null when undated.")
    month: int | None = Field(description="Month 1-12 if stated or clearly implied, else null.")
    mentioned_at_sec: float = Field(
        description="The t= value of the transcript line where this is first mentioned. Copy it exactly."
    )
    entities: list[str] = Field(
        description="Proper nouns involved (people, companies, products, places), canonical spelling."
    )
    importance: int = Field(description="1 = passing mention, 2 = notable, 3 = central to the episode.")


class TimelineWindow(BaseModel):
    """All events the model found in one transcript window."""

    events: list[TimelineEvent]


SYSTEM_PROMPT = """You build timelines from podcast transcripts.

Extract every distinct thing the speakers mention that can sit on a timeline: company foundings,
product launches, deals and acquisitions, funding rounds, IPOs, people's career moves, inventions,
historical events, regulatory decisions, and anecdotes tied to a time. Also include notable
people, companies and concepts that are discussed without a date (set date fields to null).

Rules:
- Use only what the transcript says or clearly implies. Do not add outside facts or dates the
  speakers do not support. If a date is fuzzy, keep it fuzzy in date_label and estimate the year.
- One event per real-world happening. Do not repeat the same happening within this excerpt.
- mentioned_at_sec must be copied from the [t=...] prefix of the line where it first comes up.
- Ignore ads, sponsor reads, housekeeping and small talk.
- Transcripts come from speech recognition: fix obvious misspellings of well-known names."""

HUMAN_PROMPT = """Episode: {source_label}
Excerpt {window_index} of {window_count} (timestamps are seconds from the start of the episode).

Entities already detected in this excerpt (use these spellings; every one that has a place on the
timeline should appear in at least one event's entities list):
{entities}

Transcript:
{transcript}"""

MAX_ENTITIES_PER_WINDOW = 150


class TimelineNotConfigured(RuntimeError):
    """Raised when `OPENAI_API_KEY` is missing (mapped to HTTP 503 by the API layer)."""


def timeline_configured() -> bool:
    """True when the OpenAI key needed for timelines is present."""
    return settings.openai_configured


# ---------- helpers ----------


def build_windows(segments: list[dict[str, Any]], max_chars: int | None = None) -> list[str]:
    """Pack '[t=..] text' lines into windows without splitting a segment."""
    max_chars = max_chars or TIMELINE_WINDOW_CHARS
    windows: list[str] = []
    current: list[str] = []
    size = 0
    for seg in segments:
        text = str(seg.get("text", "")).strip()
        if not text:
            continue
        line = f"[t={float(seg.get('start', 0.0)):.1f}] {text}"
        if current and size + len(line) + 1 > max_chars:
            windows.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        windows.append("\n".join(current))
    return windows


def _window_bounds(window: str) -> tuple[float, float]:
    """First and last `[t=..]` timestamps in a window (0, 0 when none)."""
    times = [float(m) for m in re.findall(r"^\[t=([0-9.]+)\]", window, flags=re.M)]
    return (min(times), max(times)) if times else (0.0, 0.0)


def entities_for_window(window: str, entities: list[dict[str, Any]], pad_sec: float = 30.0) -> str:
    """Unique 'TYPE: text' lines for entities whose mention falls inside this window."""
    lo, hi = _window_bounds(window)
    seen: set[tuple[str, str]] = set()
    lines: list[str] = []
    for e in entities:
        text = str(e.get("text", "")).strip()
        etype = str(e.get("type", "MISC")).strip().upper()
        try:
            t = float(e.get("start_sec", 0.0))
        except (TypeError, ValueError):
            t = 0.0
        if not text or not (lo - pad_sec <= t <= hi + pad_sec):
            continue
        key = (etype, text.lower())
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {etype}: {text}")
        if len(lines) >= MAX_ENTITIES_PER_WINDOW:
            break
    return "\n".join(lines) if lines else "(none detected)"


def _slug(text: str) -> str:
    """Lower-case alphanumeric key used to spot the same event title across windows."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _snap(t: float, starts: list[float]) -> float:
    """Clamp an LLM-provided timestamp to the closest real segment start."""
    if not starts:
        return max(0.0, t)
    return min(starts, key=lambda s: abs(s - t))


def merge_events(raw: list[TimelineEvent], segment_starts: list[float]) -> list[dict[str, Any]]:
    """De-duplicate across windows; keep the earliest mention and the richer description."""
    merged: dict[tuple, dict[str, Any]] = {}
    for ev in raw:
        e = ev.model_dump()
        month = e["month"] if isinstance(e["month"], int) and 1 <= e["month"] <= 12 else None
        e["month"] = month
        e["importance"] = min(3, max(1, int(e["importance"] or 1)))
        e["mentioned_at_sec"] = _snap(float(e["mentioned_at_sec"]), segment_starts)
        e["entities"] = sorted({x.strip() for x in e["entities"] if x and x.strip()})
        key = (e["year"], month, _slug(e["title"]))
        prev = merged.get(key)
        if prev is None:
            merged[key] = e
            continue
        prev["mentioned_at_sec"] = min(prev["mentioned_at_sec"], e["mentioned_at_sec"])
        prev["importance"] = max(prev["importance"], e["importance"])
        prev["entities"] = sorted(set(prev["entities"]) | set(e["entities"]))
        if len(e["description"]) > len(prev["description"]):
            prev["description"] = e["description"]
        if not prev["date_label"] and e["date_label"]:
            prev["date_label"] = e["date_label"]

    events = sorted(merged.values(), key=lambda e: (e["mentioned_at_sec"], e["title"]))
    for i, e in enumerate(events):
        e["id"] = f"ev{i:04d}"
    return events


def _build_chain():
    """Create `prompt | ChatOpenAI.with_structured_output(TimelineWindow)` (strict JSON schema)."""
    # Imported lazily so the API still starts if langchain-openai is not installed.
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model=TIMELINE_MODEL, max_retries=2, timeout=120)
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)])
    return prompt | llm.with_structured_output(TimelineWindow, method="json_schema", strict=True)


async def build_timeline(
    segments: list[dict[str, Any]],
    entities: list[dict[str, Any]] | None = None,
    source_label: str | None = None,
    chain: Any = None,
) -> dict[str, Any]:
    """Generate the episode timeline.

    Args:
        segments: Whisper segments `{id, start, end, text}`.
        entities: Entities already extracted for the episode (used as anchors in the prompt).
        source_label: Episode name shown to the model (usually the file name).
        chain: Injected LangChain runnable (tests pass a stub); built from settings when None.

    Returns:
        `{model, events, windows, failed_windows}`; events are sorted by first mention.

    Raises:
        TimelineNotConfigured: no OpenAI key.
        ValueError: transcript longer than `TIMELINE_MAX_TRANSCRIPT_CHARS`.
        RuntimeError: every window failed (the last error is included).
    """
    if chain is None:
        if not timeline_configured():
            raise TimelineNotConfigured("OPENAI_API_KEY is not set on the API server.")
        chain = _build_chain()

    total_chars = sum(len(str(s.get("text", ""))) for s in segments)
    if total_chars > TIMELINE_MAX_TRANSCRIPT_CHARS:
        raise ValueError(
            f"Transcript too long for a timeline ({total_chars} chars > {TIMELINE_MAX_TRANSCRIPT_CHARS})."
        )

    windows = build_windows(segments)
    if not windows:
        return {"model": TIMELINE_MODEL, "events": [], "windows": 0, "failed_windows": 0}

    label = (source_label or "podcast episode").strip()[:200]
    inputs = [
        {
            "source_label": label,
            "window_index": i + 1,
            "window_count": len(windows),
            "entities": entities_for_window(w, entities or []),
            "transcript": w,
        }
        for i, w in enumerate(windows)
    ]
    results = await chain.abatch(
        inputs, config={"max_concurrency": TIMELINE_MAX_CONCURRENCY}, return_exceptions=True
    )

    raw: list[TimelineEvent] = []
    failed = 0
    last_error: str | None = None
    for i, r in enumerate(results):
        if isinstance(r, BaseException):
            failed += 1
            last_error = str(r)
            logger.warning("timeline window %s failed: %s", i + 1, r)
        elif isinstance(r, TimelineWindow):
            raw.extend(r.events)
    if failed == len(windows):
        raise RuntimeError(f"Timeline generation failed: {last_error}")

    starts = sorted(float(s.get("start", 0.0)) for s in segments if str(s.get("text", "")).strip())
    return {
        "model": TIMELINE_MODEL,
        "events": merge_events(raw, starts),
        "windows": len(windows),
        "failed_windows": failed,
    }
