"""Unit tests for storage helpers and enrichment rules (no network)."""

from __future__ import annotations

from app.services.enrichment.geocoding import is_direct_location
from app.services.enrichment.service import dedupe_entities
from app.services.storage import format_clock, safe_stem, transcript_file_body


def test_safe_stem() -> None:
    """Upload names become filesystem-safe stems."""
    assert safe_stem("My Episode #12.mp3") == "My_Episode__12"
    assert safe_stem(".mp3") == ".mp3" or safe_stem("") == "audio"


def test_transcript_body_has_timestamps() -> None:
    """Saved transcript text includes [mm:ss] segment timestamps."""
    body = transcript_file_body("x", [{"start": 61.5, "end": 62, "text": "Hi"}])
    assert body == "[00:01:01.500 → 00:01:02.000] Hi\n"
    assert format_clock(3600) == "01:00:00.000"


def test_dedupe_entities_merges_span() -> None:
    """Repeated mentions collapse to one entity spanning all of them."""
    rows = dedupe_entities([
        {"type": "place", "text": "Paris", "start_sec": 5, "end_sec": 6},
        {"type": "PLACE", "text": "paris", "start_sec": 1, "end_sec": 9},
    ])
    assert len(rows) == 1 and rows[0]["start_sec"] == 1 and rows[0]["end_sec"] == 9


def test_geocoding_filters_businesses() -> None:
    """Geocoding keeps real places and skips shops/amenities."""
    assert is_direct_location({"class": "place", "type": "city"})
    assert is_direct_location({"class": "tourism", "type": "museum"})
    assert not is_direct_location({"class": "tourism", "type": "hotel"})
    assert not is_direct_location({"class": "shop", "type": "bakery"})
