"""Context rules that fix common homonyms after NER (Apple, Amazon, Oracle, Meta).

NER models label each span in isolation. These rules look at the *whole chunk* the entity came
from and either re-type it (Amazon the company vs. the river) or drop it (apple the fruit).
"""

from __future__ import annotations

import re
from typing import Any

# --- Context cues (matched against the lower-cased chunk text) -------------------------------

_APPLE_TECH = re.compile(
    r"\b(iphone|ipad|ipod|macbook|imac|ios|ipados|mac\s*os|app\s+store|cupertino|"
    r"tim\s+cook|steve\s+jobs|wwdc|airpods|apple\s+watch|icloud|m1\b|m2\b|m3\b|m4\b|"
    r"a\d+\s+bionic|vision\s+pro|apple\s+park|nasdaq|aapl|ecosystem|siri|facetime|"
    r"apple\s+music|apple\s+tv|mac\s+studio|ipad\s+pro|developer\s+conference|"
    r"silicon|osx|watchos|testflight|"
    r"apple\s+(announced|reported|reports|unveiled|released|launched|introduced|said|posted))\b",
    re.I,
)
_APPLE_FRUIT = re.compile(
    r"\b(fruit|apple\s+pie|apples\s+and|orchard|apple\s+juice|recipe|granny\s+smith|"
    r"cider|rotten\s+apple|red\s+delicious|gala\s+apple|honeycrisp|peel|crisp\s+apple|"
    r"\ban apple\b|\bthe apple\s+was\b|\bapples\s+(are|were|taste))\b",
    re.I,
)
_AMAZON_CORP = re.compile(
    r"\b(aws|amazon\s+prime|prime\s+video|kindle|alexa|bezos|e-?commerce|marketplace|"
    r"amazon\.com|fulfillment|amazon\s+web)\b",
    re.I,
)
_AMAZON_RIVER = re.compile(r"\b(rainforest|amazon\s+river|amazon\s+basin|manaus|peru|amazonia)\b", re.I)
_ORACLE_TECH = re.compile(
    r"\b(database|sql|java\b|oci\b|oracle\s+corp|larry\s+ellison|oracle\s+cloud|enterprise\s+software|erp)\b",
    re.I,
)
_ORACLE_MYTH = re.compile(r"\b(myth|prophecy|greek|delphi|ancient|pythia|apollo)\b", re.I)
_META_CORP = re.compile(
    r"\b(facebook|instagram|whatsapp|zuckerberg|meta\s+quest|threads|reality\s+labs|oculus|meta\s+platforms)\b",
    re.I,
)


def apply_context_disambiguation(entity: dict[str, Any], chunk_lower: str) -> dict[str, Any] | None:
    """Return a (possibly re-typed) copy of `entity`, or None to drop it.

    Args:
        entity: Entity dict with at least `text` and `type`.
        chunk_lower: Lower-cased text of the chunk the entity was found in.
    """
    e = dict(entity)
    low = (e.get("text") or "").strip().lower()
    typ = str(e.get("type", ""))

    if low == "apple":
        tech = _APPLE_TECH.search(chunk_lower)
        fruit = _APPLE_FRUIT.search(chunk_lower)
        if fruit and not tech:
            return None  # the fruit: not a useful card
        # Tech context, mixed context, or no context at all: in business podcasts a bare
        # "Apple" is almost always the company, whatever label NER gave it.
        e["type"] = "COMPANY"
        return e

    if low == "amazon":
        corp = _AMAZON_CORP.search(chunk_lower)
        river = _AMAZON_RIVER.search(chunk_lower)
        if corp and not river:
            e["type"] = "COMPANY"
        elif river and not corp and typ == "COMPANY":
            e["type"] = "PLACE"
        return e

    if low == "oracle":
        software = _ORACLE_TECH.search(chunk_lower)
        myth = _ORACLE_MYTH.search(chunk_lower)
        if software and not myth:
            e["type"] = "COMPANY"
        elif myth and not software:
            e["type"] = "EVENT"
        return e

    if low == "meta" and _META_CORP.search(chunk_lower):
        e["type"] = "COMPANY"

    return e


def refine_entities_with_chunk_context(
    chunks: list[dict[str, Any]], entities: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Run `apply_context_disambiguation` for every entity using the text of its source chunk."""
    by_id = {int(c["id"]): c for c in chunks if c.get("id") is not None}
    out: list[dict[str, Any]] = []
    for ent in entities:
        try:
            chunk_id = int(ent.get("chunk_id", -1))
        except (TypeError, ValueError):
            out.append(ent)  # cannot locate the chunk: keep the entity unchanged
            continue
        chunk = by_id.get(chunk_id)
        chunk_lower = (chunk.get("text_clean") or chunk.get("text") or "").lower() if chunk else ""
        refined = apply_context_disambiguation(ent, chunk_lower)
        if refined is not None:
            out.append(refined)
    return out
