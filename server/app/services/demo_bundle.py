"""Build pre-processed "demo episodes" for the static public site (GitHub Pages).

A demo episode is everything the UI would normally fetch from the API, computed once and saved
as static files so the public site needs no server and no API keys:

    public/demo/
      index.json              list of demo episodes (picker)
      <slug>/episode.json     transcript, entities, source cards, timeline, preset Q&A
      <slug>/audio.mp3        compressed copy of the audio for playback

Run it through the CLI (`server/scripts/build_demo.py`), which needs the same setup as the API
(Whisper, ffmpeg, and optionally OpenAI/Anthropic/Unsplash keys in `.env`).
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from app.core.config import PROJECT_ROOT, settings
from app.services.audio import ffmpeg_binary

logger = logging.getLogger(__name__)

#: Bump when the episode.json shape changes; the frontend checks it.
SCHEMA_VERSION = 1

#: Default output folder: Vite copies `public/` into the built site as-is.
DEFAULT_OUT_DIR = PROJECT_ROOT / "public" / "demo"


@dataclass
class DemoOptions:
    """Inputs for one demo episode (see the CLI for descriptions)."""

    audio_path: Path
    title: str
    slug: str
    description: str = ""
    credit: str = ""
    questions: list[str] = field(default_factory=list)
    backend: str | None = None
    language: str | None = None
    bitrate: str = "64k"
    out_dir: Path = DEFAULT_OUT_DIR
    skip_timeline: bool = False
    skip_ask: bool = False
    #: LLM review of entities and cards (None = server default: on when OpenAI is configured).
    review: bool | None = None


# ---------------------------------------------------------------------------------------------
# Small pure helpers (unit tested)
# ---------------------------------------------------------------------------------------------


def slugify(text: str) -> str:
    """URL-safe folder name: lowercase words joined by '-', e.g. "Apollo 11 Landing" -> "apollo-11-landing"."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:60].rstrip("-") or "episode"


def default_questions(entities: list[dict[str, Any]]) -> list[str]:
    """Starter questions built from the most-mentioned entities.

    These get pre-computed answers. On the public demo, typed questions are matched to them
    loosely (src/features/demo/demoData.ts), and they are offered as follow-ups otherwise.
    """
    counts: Counter[tuple[str, str]] = Counter()
    display: dict[tuple[str, str], str] = {}
    for e in entities:
        text = str(e.get("text", "")).strip()
        if not text:
            continue
        key = (str(e.get("type", "")), text.lower())
        counts[key] += 1
        display.setdefault(key, text)
    # Stable ranking: most mentions first, ties keep first-seen order (like the TS sort).
    ranked = sorted(counts, key=lambda k: -counts[k])

    def pick(types: tuple[str, ...]) -> str | None:
        """Display text of the top-ranked entity whose type is in `types`."""
        return next((display[k] for k in ranked if k[0] in types), None)

    out = ["Summarize this episode in 3 points."]
    if person := pick(("PERSON",)):
        out.append(f"Who is {person} and why are they mentioned?")
    if org := pick(("COMPANY", "TECHNOLOGY")):
        out.append(f"What is said about {org}?")
    if (place := pick(("PLACE", "EVENT"))) and len(out) < 4:
        out.append(f"What happens around {place}?")
    return out[:4]


def episode_summary(episode: dict[str, Any]) -> dict[str, Any]:
    """The subset of an episode shown in the picker (`index.json` entry)."""
    entities = (episode.get("document") or {}).get("entities") or []
    return {
        "slug": episode["slug"],
        "title": episode["title"],
        "description": episode.get("description", ""),
        "credit": episode.get("credit", ""),
        "duration_sec": episode.get("duration_sec", 0),
        "entity_count": len({(e["type"], e["text"].lower()) for e in entities}),
        "has_timeline": episode.get("timeline") is not None,
        "question_count": len(episode.get("qa") or []),
    }


def upsert_index(index_path: Path, summary: dict[str, Any]) -> dict[str, Any]:
    """Add or replace one episode in `index.json` (keyed by slug) and write it back."""
    index: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "episodes": []}
    if index_path.exists():
        try:
            loaded = json.loads(index_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("episodes"), list):
                index["episodes"] = loaded["episodes"]
        except json.JSONDecodeError:
            logger.warning("index.json was invalid; recreating it")
    episodes = [e for e in index["episodes"] if e.get("slug") != summary["slug"]]
    episodes.append(summary)
    index["episodes"] = sorted(episodes, key=lambda e: e["title"].lower())
    write_json(index_path, index)
    return index


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write compact-but-readable UTF-8 JSON, creating parent folders."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def compress_audio(src: Path, dest: Path, bitrate: str = "64k") -> None:
    """Re-encode audio to mono MP3 at `bitrate` so the published file stays small (~0.5 MB/min at 64k).

    Raises:
        RuntimeError: ffmpeg failed (stderr tail included).
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_binary(), "-hide_banner", "-loglevel", "error", "-y", "-nostdin",
        "-i", str(src), "-vn", "-ac", "1", "-codec:a", "libmp3lame", "-b:a", bitrate, str(dest),
    ]  # fmt: skip
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        tail = (result.stderr or b"").decode(errors="replace").strip()[-400:]
        raise RuntimeError(f"ffmpeg could not compress audio: {tail}")


# ---------------------------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------------------------

#: Injectable steps, so tests can run the pipeline without Whisper, networks or API keys.
Transcribe = Callable[[str, str | None], tuple[str, list[dict[str, Any]]]]
Extract = Callable[..., dict[str, Any]]
Enrich = Callable[[list[dict[str, Any]]], Awaitable[dict[str, Any]]]
Timeline = Callable[..., Awaitable[dict[str, Any]]]
Answer = Callable[..., Awaitable[dict[str, Any]]]
Compress = Callable[[Path, Path, str], None]

#: Names of the pipeline steps that `build_demo_episode` expects.
STEP_NAMES = frozenset({"transcribe", "extract", "enrich", "timeline", "answer", "compress"})


def _default_steps() -> dict[str, Any]:
    """Real implementations, imported lazily (Whisper and LangChain are heavy)."""
    from app.services.ask import answer_question
    from app.services.source_cards import build_source_cards
    from app.services.entities import extract_document
    from app.services.timeline import build_timeline
    from app.services.transcription import transcribe_file

    return {
        "transcribe": transcribe_file,
        "extract": extract_document,
        "enrich": build_source_cards,
        "timeline": build_timeline,
        "answer": answer_question,
        "compress": compress_audio,
    }


async def build_demo_episode(
    opts: DemoOptions,
    *,
    steps: dict[str, Any] | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Run the full PodLens pipeline on one audio file and write the demo bundle.

    Steps: transcribe -> tag entities -> source cards -> timeline -> preset Q&A -> compress audio
    -> write `<slug>/episode.json` -> update `index.json`.

    The timeline and Q&A are skipped (with a warning) when no OpenAI key is configured, so a demo
    can be built with free, local tools only. Nothing machine-specific (file paths, keys) is
    written to the bundle, because it is published.

    Returns:
        The episode payload that was written.
    """
    s = dict(steps or {})
    if not STEP_NAMES <= s.keys():
        # Fill in any step not injected with the real implementation.
        s = {**_default_steps(), **s}
    episode_dir = opts.out_dir / opts.slug

    log(f"1/6 Transcribing {opts.audio_path.name} with Whisper ({settings.whisper_model})…")
    transcript, segments = s["transcribe"](str(opts.audio_path), opts.language)
    if not segments:
        raise RuntimeError("Whisper returned no segments; is the audio silent or unreadable?")

    log("2/6 Tagging entities…")
    document = s["extract"](segments, source_label=opts.title, backend=opts.backend)
    entities = document.get("entities") or []
    log(f"    {len(entities)} mentions via {document.get('backend')}")

    log("3/6 Reviewing entities and building source cards (Wikipedia, maps, photos)…")
    enriched = (
        await s["enrich"](entities, segments=segments, source_label=opts.title, review=opts.review)
        if entities
        else {"cards": [], "unsplash_enabled": False}
    )
    review = enriched.get("review")
    if enriched.get("entities") is not None:
        # The review corrected, merged or dropped entities: publish the reviewed list.
        entities = enriched["entities"]
        document = {**document, "entities": entities}
    if review and not review.get("error"):
        log(
            f"    review: {len(review['fixed'])} fixed, {len(review['merged'])} merged, "
            f"{len(review['dropped'])} dropped, {review.get('mismatches_removed', 0)} mismatched lookups removed"
        )
    elif review:
        log(f"    review failed, using unreviewed entities: {review['error']}")

    openai_ready = settings.openai_configured
    timeline: dict[str, Any] | None = None
    if opts.skip_timeline:
        log("4/6 Timeline skipped (--skip-timeline)")
    elif not openai_ready:
        log("4/6 Timeline skipped: OPENAI_API_KEY is not set")
    else:
        log("4/6 Building the timeline…")
        timeline = await s["timeline"](segments, entities, opts.title)

    qa: list[dict[str, Any]] = []
    questions = opts.questions or default_questions(entities)
    if opts.skip_ask:
        log("5/6 Q&A skipped (--skip-ask)")
    elif not openai_ready:
        log("5/6 Q&A skipped: OPENAI_API_KEY is not set")
    else:
        log(f"5/6 Answering {len(questions)} preset questions…")
        for q in questions:
            response = await s["answer"](q, segments, entities, None, opts.title)
            qa.append({"question": q, "response": response})

    log(f"6/6 Compressing audio ({opts.bitrate} mono MP3)…")
    s["compress"](opts.audio_path, episode_dir / "audio.mp3", opts.bitrate)

    episode = {
        "schema_version": SCHEMA_VERSION,
        "slug": opts.slug,
        "title": opts.title,
        "description": opts.description,
        "credit": opts.credit,
        "audio": "audio.mp3",
        "duration_sec": round(float(segments[-1]["end"]), 2),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "generated_with": {
            "whisper_model": settings.whisper_model,
            "entity_backend": document.get("backend"),
            "timeline_model": (timeline or {}).get("model"),
            "ask_model": qa[0]["response"].get("model") if qa else None,
        },
        "transcript": transcript,
        "segments": segments,
        "document": {**document, "source_label": opts.title},
        "cards": enriched.get("cards", []),
        "unsplash_enabled": bool(enriched.get("unsplash_enabled")),
        "review": review,
        "timeline": timeline,
        "qa": qa,
    }
    write_json(episode_dir / "episode.json", episode)
    upsert_index(opts.out_dir / "index.json", episode_summary(episode))
    log(f"Done: {episode_dir}")
    return episode
