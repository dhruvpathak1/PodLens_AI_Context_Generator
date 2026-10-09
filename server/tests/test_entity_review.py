"""Tests for the LLM entity review and card check (model stubbed: no network or keys)."""

from __future__ import annotations

import asyncio

from app.services import entity_review, source_cards
from app.services.entity_review import (
    CardCheck,
    CardReview,
    EntityReview,
    EntityVerdict,
    apply_verdicts,
    group_entities,
    render_entities,
)

SEGMENTS = [
    {"id": 0, "start": 0.0, "end": 6.0, "text": "Welcome back to the Deep Dive. Today: San Francisco."},
    {"id": 1, "start": 6.0, "end": 12.0, "text": "SF began as Yerba Buena, a small Spanish settlement."},
    {"id": 2, "start": 12.0, "end": 18.0, "text": "Later the Traderus 8 founded Fairchild in the south bay of San."},
]
ENTITIES = [
    {"type": "MISC", "text": "Deep Dive", "start_sec": 1.0, "end_sec": 2.0, "chunk_id": 0},
    {"type": "PLACE", "text": "San Francisco", "start_sec": 4.0, "end_sec": 5.0, "chunk_id": 0},
    {"type": "PLACE", "text": "SF", "start_sec": 6.1, "end_sec": 6.4, "chunk_id": 1},
    {"type": "PLACE", "text": "Yerba Buena", "start_sec": 8.0, "end_sec": 9.0, "chunk_id": 1},
    {"type": "MISC", "text": "Traderus 8", "start_sec": 13.0, "end_sec": 14.0, "chunk_id": 2},
    {"type": "PLACE", "text": "San", "start_sec": 17.0, "end_sec": 17.3, "chunk_id": 2},
    {"type": "PLACE", "text": "San Francisco", "start_sec": 17.5, "end_sec": 18.0, "chunk_id": 2},
]


def verdict(id_, action, name, type_, merge_into=None, query=None):
    """Build an EntityVerdict with a short reason."""
    return EntityVerdict(id=id_, action=action, name=name, type=type_, merge_into=merge_into, search_query=query, reason="test")


# Groups (by first mention): e0 Deep Dive, e1 San Francisco, e2 SF, e3 Yerba Buena, e4 Traderus 8, e5 San
VERDICTS = [
    verdict("e0", "drop", "Deep Dive", "MISC"),
    verdict("e1", "keep", "San Francisco", "PLACE", query="San Francisco California"),
    verdict("e2", "merge", "SF", "PLACE", merge_into="e1"),
    verdict("e3", "keep", "Yerba Buena", "PLACE", query="Yerba Buena San Francisco history"),
    verdict("e4", "fix", "Traitorous Eight", "PERSON", query="Traitorous Eight Fairchild"),
    verdict("e5", "drop", "San", "PLACE"),
]


class StubChain:
    """Fake LangChain runnable: returns canned results and records the prompts it received."""

    def __init__(self, results):
        """Store the results to return from `abatch`."""
        self.results, self.inputs = results, []

    async def abatch(self, inputs, config=None, return_exceptions=False):
        """Return the canned results, one per input."""
        self.inputs.extend(inputs)
        return self.results[: len(inputs)]


def test_group_entities_orders_by_first_mention_and_counts() -> None:
    """Unique entities get ids in mention order and mention counts."""
    groups = group_entities(ENTITIES)
    assert [g["text"] for g in groups] == ["Deep Dive", "San Francisco", "SF", "Yerba Buena", "Traderus 8", "San"]
    assert groups[1]["count"] == 2


def test_prompt_includes_context_lines() -> None:
    """Each entity is sent with the transcript lines it appears in."""
    block = render_entities(group_entities(ENTITIES), SEGMENTS)
    assert "e3 | PLACE | Yerba Buena | 1 | SF began as Yerba Buena" in block


def test_apply_verdicts_fix_merge_drop() -> None:
    """Drops vanish, merges take the target's name, fixes rename and retype, hints are attached."""
    reviewed, report = apply_verdicts(ENTITIES, group_entities(ENTITIES), VERDICTS)
    names = [(m["text"], m["type"]) for m in reviewed]
    assert ("Deep Dive", "MISC") not in names and ("San", "PLACE") not in names
    assert names.count(("San Francisco", "PLACE")) == 3  # two originals + merged "SF"
    assert ("Traitorous Eight", "PERSON") in names
    yb = next(m for m in reviewed if m["text"] == "Yerba Buena")
    assert yb["search_query"] == "Yerba Buena San Francisco history"
    assert report["merged"] == [{"name": "SF", "into": "San Francisco"}]
    assert report["fixed"] == [{"name": "Traderus 8", "to": "Traitorous Eight", "type": "PERSON"}]
    assert {d["name"] for d in report["dropped"]} == {"Deep Dive", "San"}
    assert report["kept"] == 3


def test_missing_verdicts_and_bad_merge_targets_keep_entities() -> None:
    """The review fails open: no verdict, or a merge into an unknown id, keeps the entity."""
    reviewed, report = apply_verdicts(
        ENTITIES, group_entities(ENTITIES), [verdict("e2", "merge", "SF", "PLACE", merge_into="e99")]
    )
    assert len(reviewed) == len(ENTITIES)
    assert report["dropped"] == [] and report["merged"] == []


def test_review_entities_uses_chain_and_reports() -> None:
    """The review sends one batch and returns reviewed mentions plus a report."""
    chain = StubChain([EntityReview(verdicts=VERDICTS)])
    reviewed, report = asyncio.run(entity_review.review_entities(ENTITIES, SEGMENTS, "SF episode", chain=chain))
    assert len(chain.inputs) == 1 and "Welcome back" in chain.inputs[0]["topic"]
    assert report["failed_batches"] == 0 and len(reviewed) == 5


def test_failed_batch_keeps_entities() -> None:
    """If the model call fails, the batch's entities are kept unchanged."""
    chain = StubChain([RuntimeError("model down")])
    reviewed, report = asyncio.run(entity_review.review_entities(ENTITIES, SEGMENTS, chain=chain))
    assert len(reviewed) == len(ENTITIES) and report["failed_batches"] == 1


def test_verify_cards_removes_mismatched_sections() -> None:
    """A Wikipedia summary for the wrong sense (the plant) is removed; matching content stays."""
    cards = [
        {"id": "c1", "type": "PLACE", "text": "Yerba Buena", "wikipedia": {"title": "Yerba buena", "extract": "An aromatic plant"}, "unsplash": {"alt": "mint leaves"}},
        {"id": "c2", "type": "PLACE", "text": "San Francisco", "wikipedia": {"title": "San Francisco", "extract": "A city"}, "unsplash": None},
    ]
    chain = StubChain([CardReview(checks=[CardCheck(id="c1", wikipedia_matches=False, photo_matches=False), CardCheck(id="c2", wikipedia_matches=True, photo_matches=None)])])
    out, removed = asyncio.run(entity_review.verify_cards(cards, ENTITIES, SEGMENTS, chain=chain))
    assert removed == 2
    assert out[0]["wikipedia"] is None and out[0]["unsplash"] is None
    assert out[1]["wikipedia"]["title"] == "San Francisco"
    assert "Yerba Buena" in chain.inputs[0]["cards"] and "SF began as Yerba Buena" in chain.inputs[0]["cards"]


def test_build_source_cards_with_review(monkeypatch) -> None:
    """End to end: review -> lookups (with search hints) -> card check -> reviewed entities returned."""
    seen = {}

    async def fake_enrich(entities):
        """Record what is looked up and return one bare card per entity."""
        seen["entities"] = entities
        return {"cards": [{"id": f"c{i}", **e, "wikipedia": None, "unsplash": None} for i, e in enumerate(entities)], "count": len(entities), "unsplash_enabled": False}

    monkeypatch.setattr(source_cards, "enrich_entities_payload", fake_enrich)
    result = asyncio.run(source_cards.build_source_cards(
        ENTITIES, segments=SEGMENTS, source_label="SF", review=True,
        review_chain=StubChain([EntityReview(verdicts=VERDICTS)]), verify_chain=StubChain([]),
    ))  # fmt: skip
    assert {e["text"] for e in result["entities"]} == {"San Francisco", "Yerba Buena", "Traitorous Eight"}
    assert any(e.get("search_query") == "Traitorous Eight Fairchild" for e in seen["entities"])
    assert result["review"]["mismatches_removed"] == 0


def test_build_source_cards_falls_back_when_review_breaks(monkeypatch) -> None:
    """A broken reviewer never costs the cards: unreviewed entities are used and the error reported."""

    async def fake_enrich(entities):
        """Return an empty card set."""
        return {"cards": [], "count": 0, "unsplash_enabled": False}

    class Broken:
        """Runnable whose abatch raises."""

        async def abatch(self, *a, **k):
            """Always fail."""
            raise RuntimeError("boom")

    monkeypatch.setattr(source_cards, "enrich_entities_payload", fake_enrich)
    result = asyncio.run(source_cards.build_source_cards(ENTITIES, segments=SEGMENTS, review=True, review_chain=Broken()))
    assert result["review"] == {"error": "boom"} and "entities" not in result


def test_review_skipped_without_segments_or_when_disabled(monkeypatch) -> None:
    """No transcript (or review off) means plain enrichment, exactly as before."""

    async def fake_enrich(entities):
        """Return an empty card set."""
        return {"cards": [], "count": 0, "unsplash_enabled": False}

    monkeypatch.setattr(source_cards, "enrich_entities_payload", fake_enrich)
    assert "review" not in asyncio.run(source_cards.build_source_cards(ENTITIES, segments=[], review=True))
    assert "review" not in asyncio.run(source_cards.build_source_cards(ENTITIES, segments=SEGMENTS, review=False))


def test_schemas_are_strict_mode_compatible() -> None:
    """OpenAI strict mode needs every property listed as required, in every model."""
    for model in (EntityVerdict, EntityReview, CardCheck, CardReview):
        schema = model.model_json_schema()
        assert set(schema["required"]) == set(schema["properties"]), model.__name__
