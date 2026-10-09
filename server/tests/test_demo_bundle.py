"""Tests for the demo bundle builder (pipeline steps stubbed: no Whisper, network or keys)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from app.services import demo_bundle
from app.services.demo_bundle import DemoOptions, build_demo_episode, default_questions, slugify, upsert_index

SEGMENTS = [
    {"id": 0, "start": 0.0, "end": 5.0, "text": "Neil Armstrong flew Apollo 11 for NASA."},
    {"id": 1, "start": 5.0, "end": 9.5, "text": "They landed in the Sea of Tranquility in 1969."},
]
ENTITIES = [
    {"type": "PERSON", "text": "Neil Armstrong", "start_sec": 0.5, "end_sec": 1.5, "chunk_id": 0},
    {"type": "COMPANY", "text": "NASA", "start_sec": 4.0, "end_sec": 4.5, "chunk_id": 0},
    {"type": "PLACE", "text": "Sea of Tranquility", "start_sec": 6.0, "end_sec": 7.0, "chunk_id": 1},
    {"type": "COMPANY", "text": "NASA", "start_sec": 8.0, "end_sec": 8.5, "chunk_id": 1},
]


def _steps(calls: list[str]) -> dict:
    """Stub pipeline steps that record the order they were called in."""

    def transcribe(path, language):
        """Return the fixed test transcript."""
        calls.append("transcribe")
        return "Neil Armstrong flew Apollo 11.", SEGMENTS

    def extract(segments, *, source_label, backend):
        """Return a fixed entity document."""
        calls.append("extract")
        return {"backend": "spacy", "source_label": "/Users/me/secret/path.mp3", "chunks": segments, "entities": ENTITIES}

    async def enrich(entities):
        """Return one card per entity."""
        calls.append("enrich")
        return {"cards": [{"id": e["text"], **e} for e in entities], "unsplash_enabled": False}

    async def timeline(segments, entities, label):
        """Return a one-event timeline."""
        calls.append("timeline")
        return {"model": "m-tl", "events": [{"id": "ev0"}], "windows": 1, "failed_windows": 0}

    async def answer(q, segments, entities, history, label):
        """Echo the question back as an answer."""
        calls.append(f"answer:{q}")
        return {"answer": f"A: {q}", "found": True, "citations": [], "mode": "full", "model": "m-ask", "unverified": False}

    def compress(src, dest, bitrate):
        """Pretend to encode by writing a placeholder file."""
        calls.append("compress")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"ID3")

    return {"transcribe": transcribe, "extract": extract, "enrich": enrich, "timeline": timeline, "answer": answer, "compress": compress}


def _opts(tmp_path: Path, **kw) -> DemoOptions:
    """Options pointing at a temp output folder."""
    audio = tmp_path / "in.wav"
    audio.write_bytes(b"RIFF")
    return DemoOptions(audio_path=audio, title="Apollo 11", slug="apollo-11", out_dir=tmp_path / "demo", **kw)


def test_slugify() -> None:
    """Titles become short, URL-safe folder names."""
    assert slugify("Apollo 11: The Landing!") == "apollo-11-the-landing"
    assert slugify("???") == "episode"


def test_default_questions_match_frontend_rules() -> None:
    """Default questions: summary, top person, top company, top place."""
    assert default_questions(ENTITIES) == [
        "Summarize this episode in 3 points.",
        "Who is Neil Armstrong and why are they mentioned?",
        "What is said about NASA?",
        "What happens around Sea of Tranquility?",
    ]
    assert default_questions([]) == ["Summarize this episode in 3 points."]


def test_full_bundle_with_openai(tmp_path, monkeypatch) -> None:
    """With an OpenAI key the bundle has a timeline and preset answers, and no local paths."""
    monkeypatch.setattr(demo_bundle.settings.__class__, "openai_configured", property(lambda self: True))
    calls: list[str] = []
    ep = asyncio.run(build_demo_episode(_opts(tmp_path, questions=["Q1?"]), steps=_steps(calls), log=lambda _: None))

    assert calls[:4] == ["transcribe", "extract", "enrich", "timeline"]
    assert "answer:Q1?" in calls and calls[-1] == "compress"
    assert ep["timeline"]["model"] == "m-tl"
    assert ep["qa"] == [{"question": "Q1?", "response": ep["qa"][0]["response"]}]
    assert ep["duration_sec"] == 9.5
    assert ep["generated_with"]["ask_model"] == "m-ask"

    written = (tmp_path / "demo" / "apollo-11" / "episode.json").read_text()
    assert "/Users/me" not in written  # the machine-specific source label is replaced by the title
    assert (tmp_path / "demo" / "apollo-11" / "audio.mp3").exists()
    index = json.loads((tmp_path / "demo" / "index.json").read_text())
    assert index["episodes"][0] == {
        "slug": "apollo-11", "title": "Apollo 11", "description": "", "credit": "",
        "duration_sec": 9.5, "entity_count": 3, "has_timeline": True, "question_count": 1,
    }  # fmt: skip


def test_without_openai_skips_ai_steps(tmp_path, monkeypatch) -> None:
    """Without an OpenAI key the demo still builds, just without timeline and Q&A."""
    monkeypatch.setattr(demo_bundle.settings.__class__, "openai_configured", property(lambda self: False))
    calls: list[str] = []
    ep = asyncio.run(build_demo_episode(_opts(tmp_path), steps=_steps(calls), log=lambda _: None))
    assert "timeline" not in calls and not any(c.startswith("answer") for c in calls)
    assert ep["timeline"] is None and ep["qa"] == []


def test_empty_transcript_fails(tmp_path) -> None:
    """Silent or unreadable audio stops the build with a clear error."""
    steps = _steps([])
    steps["transcribe"] = lambda path, language: ("", [])
    with pytest.raises(RuntimeError, match="no segments"):
        asyncio.run(build_demo_episode(_opts(tmp_path), steps=steps, log=lambda _: None))


def test_upsert_index_replaces_by_slug(tmp_path) -> None:
    """Rebuilding an episode replaces its index entry; entries stay sorted by title."""
    path = tmp_path / "index.json"
    upsert_index(path, {"slug": "b", "title": "Beta"})
    upsert_index(path, {"slug": "a", "title": "Alpha"})
    upsert_index(path, {"slug": "b", "title": "Beta v2"})
    assert [e["title"] for e in json.loads(path.read_text())["episodes"]] == ["Alpha", "Beta v2"]
