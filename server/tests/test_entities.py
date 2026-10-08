"""Unit tests for transcript clean-up, span timing and homonym rules."""

from __future__ import annotations

from app.services.entities.disambiguation import apply_context_disambiguation
from app.services.entities.pipeline import normalize_chunks, resolve_entity_backend
from app.services.entities.text_cleaning import clean_entity_text, entity_time_span, filter_noise


def test_filter_noise_removes_fillers() -> None:
    """Filler words ("um", "you know") are stripped before NER."""
    assert filter_noise("Um, so basically Nvidia, you know, won.") == ", so Nvidia, , won."


def test_clean_entity_text_drops_generic_and_short() -> None:
    """Generic or too-short entity texts are discarded."""
    assert clean_entity_text("  Nvidia ") == "Nvidia"
    assert clean_entity_text("today") is None
    assert clean_entity_text("x") is None


def test_entity_time_span_interpolates() -> None:
    """Entity times are interpolated from their position in the segment."""
    assert entity_time_span(10.0, 20.0, "a" * 100, 50, 60) == (15.0, 16.0)


def test_apple_fruit_dropped_and_company_kept() -> None:
    """"Apple" is kept only when the context means the company."""
    apple = {"text": "Apple", "type": "PERSON"}
    assert apple_type(apple, "i baked an apple pie with fruit") is None
    assert apple_type(apple, "apple announced the iphone") == "COMPANY"


def apple_type(entity, chunk):
    """Return the type assigned to "Apple" in the given sentence."""
    result = apply_context_disambiguation(entity, chunk)
    return None if result is None else result["type"]


def test_amazon_river_becomes_place() -> None:
    """"Amazon River" is tagged PLACE, not COMPANY."""
    e = {"text": "Amazon", "type": "COMPANY"}
    assert apply_context_disambiguation(e, "the amazon river rainforest")["type"] == "PLACE"


def test_resolve_backend_defaults_to_spacy() -> None:
    """With no key or override, the NER backend is spaCy."""
    assert resolve_entity_backend(None) == "spacy"
    assert resolve_entity_backend("CLAUDE") == "claude"
    assert resolve_entity_backend("unknown") == "spacy"


def test_normalize_chunks_assigns_ids() -> None:
    """Chunks without ids get sequential ids."""
    out = normalize_chunks([{"start": 0, "end": 1, "text": "Um hello"}])
    assert out[0]["id"] == 0 and out[0]["text_clean"] == "hello"
