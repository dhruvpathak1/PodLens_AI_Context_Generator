"""Timeline service: windowing, entity scoping, merging and failure handling (LLM stubbed)."""

from __future__ import annotations

import asyncio

import pytest
from langchain_core.runnables import RunnableLambda

from app.services import timeline as tl

SEGS = [{"id": i, "start": i * 5.0, "end": i * 5.0 + 5, "text": f"Sentence {i} about Nvidia and TSMC."} for i in range(400)]


def event(**kw) -> tl.TimelineEvent:
    """Build a minimal raw timeline event for tests."""
    base = dict(title="Nvidia founded", description="At a Denny's.", category="COMPANY", date_label="1993",
                year=1993, month=None, mentioned_at_sec=12.3, entities=["Nvidia"], importance=3)
    base.update(kw)
    return tl.TimelineEvent(**base)


def test_strict_schema_has_all_fields_required() -> None:
    """OpenAI strict mode needs every property listed as required."""
    from langchain_core.utils.function_calling import convert_to_openai_tool

    fn = convert_to_openai_tool(tl.TimelineWindow, strict=True)["function"]
    item = fn["parameters"]["properties"]["events"]["items"]
    assert fn["strict"] is True and set(item["required"]) == set(item["properties"])


def test_build_windows_respects_size() -> None:
    """Transcript windows never exceed the configured size."""
    windows = tl.build_windows(SEGS, 4000)
    assert len(windows) > 1 and max(map(len, windows)) <= 4000
    assert sum(w.count("\n") + 1 for w in windows) == len(SEGS)  # every segment exactly once
    assert windows[0].startswith("[t=0.0] Sentence 0")


def test_entities_scoped_to_window() -> None:
    """Each window only receives entities mentioned inside it."""
    ents = [{"type": "COMPANY", "text": "Nvidia", "start_sec": 10}, {"type": "COMPANY", "text": "nvidia", "start_sec": 20},
            {"type": "PERSON", "text": "Jensen Huang", "start_sec": 1900}]
    windows = tl.build_windows(SEGS, 4000)
    assert tl.entities_for_window(windows[0], ents) == "- COMPANY: Nvidia"
    assert tl.entities_for_window(windows[-1], ents) == "- PERSON: Jensen Huang"


def test_merge_dedupes_and_clamps(monkeypatch: pytest.MonkeyPatch) -> None:
    """Merging windows removes duplicate events and clamps bad values."""
    monkeypatch.setattr(tl, "TIMELINE_WINDOW_CHARS", 4000)
    calls = []

    def fake(inp):
        """Stand-in for the LLM call used by the test above."""
        calls.append(inp["window_index"])
        if inp["window_index"] == 3:
            raise RuntimeError("rate limited")
        return tl.TimelineWindow(events=[
            event(description="x" * inp["window_index"], mentioned_at_sec=12.3 + inp["window_index"] * 100),
            event(title="TSMC founded", date_label=None, year=1987, month=13, mentioned_at_sec=99999, importance=5),
        ])

    res = asyncio.run(tl.build_timeline(SEGS, [], "ep", chain=RunnableLambda(fake)))
    expected = len(tl.build_windows(SEGS, 4000))
    assert res["windows"] == expected and res["failed_windows"] == 1
    assert sorted(calls) == list(range(1, expected + 1))
    titles = {e["title"]: e for e in res["events"]}
    assert set(titles) == {"Nvidia founded", "TSMC founded"}
    assert titles["Nvidia founded"]["mentioned_at_sec"] == 110.0  # earliest mention, snapped to a segment
    assert len(titles["Nvidia founded"]["description"]) == expected  # richest description kept
    assert titles["TSMC founded"]["month"] is None and titles["TSMC founded"]["importance"] == 3


def test_all_windows_failing_raises() -> None:
    """A timeline with every window failing raises instead of returning empty."""
    def boom(_):
        """Stand-in LLM call that always fails."""
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        asyncio.run(tl.build_timeline(SEGS[:3], [], "x", chain=RunnableLambda(boom)))


def test_missing_key_raises() -> None:
    """Without OPENAI_API_KEY the timeline service refuses to run."""
    with pytest.raises(tl.TimelineNotConfigured):
        asyncio.run(tl.build_timeline(SEGS[:3], [], "x"))
